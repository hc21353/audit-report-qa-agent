"""
SQLite chunks → KoE5 임베딩 → ChromaDB 벡터 저장소

모델 선택 근거:
  - KoE5 (nlpai-lab/KoE5): 한국어 특화 E5 계열, Recall@1 최고 성능
    → "정답을 정확하게 1위로 찾는 능력"이 핵심인 RAG QA에 최적
  - 대안: KURE-v1 (Recall@3/5 우수, 후보 다수 활용 시)
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import chromadb
import numpy as np
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # 루트에 있으니까 .parent 한 번만
DB_PATH = ROOT / "db" / "audit_reports.db"
VECTOR_DIR = ROOT / "db" / "vectorstore" / "chroma"

# ── 모델 설정 ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "nlpai-lab/KoE5"
COLLECTION_NAME = "samsung_audit"

# KoE5는 E5 계열로 query/passage prefix 필요
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


# ── 임베딩 모델 래퍼 ──────────────────────────────────────────────────────────
class KoE5Embedder:
    """
    KoE5 / E5 계열 모델 래퍼.
    - 색인 시: passage: <text>
    - 검색 시: query: <text>
    """
    def __init__(self, model_name: str = DEFAULT_MODEL):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  디바이스: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print(f"  모델 로드 완료: {model_name}")

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# ── ChromaDB 커스텀 임베딩 함수 ───────────────────────────────────────────────
class KoE5EmbeddingFunction:
    """ChromaDB EmbeddingFunction 인터페이스 구현."""
    def __init__(self, embedder: KoE5Embedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        vecs = self._embedder.encode_passages(input, batch_size=32)
        return vecs.tolist()


# ── SQLite에서 청크 로드 ──────────────────────────────────────────────────────
def load_chunks_from_db(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT chunk_uid, fiscal_year, period, section_path,
                  chunk_type, is_consolidated, content, table_ref, table_unit
           FROM chunks
           ORDER BY fiscal_year, id"""
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── 메타데이터 직렬화 ────────────────────────────────────────────────────────
def make_chroma_metadata(row: dict) -> dict:
    """
    ChromaDB metadata는 str/int/float/bool만 허용.
    None → "" 변환 필요.
    """
    return {
        "fiscal_year":    int(row["fiscal_year"]),
        "period":         row["period"] or "",
        "section_path":   row["section_path"] or "",
        "chunk_type":     row["chunk_type"] or "",
        "is_consolidated": bool(row["is_consolidated"]),
        "table_ref":      row["table_ref"] or "",
        "table_unit":     row["table_unit"] or "",
    }


# ── 배치 upsert ───────────────────────────────────────────────────────────────
def upsert_batch(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    embedder: KoE5Embedder,
    batch_size: int,
):
    for i in range(0, len(ids), batch_size):
        b_ids  = ids[i:i+batch_size]
        b_docs = documents[i:i+batch_size]
        b_meta = metadatas[i:i+batch_size]
        b_embs = embedder.encode_passages(b_docs, batch_size=batch_size)

        collection.upsert(
            ids=b_ids,
            documents=b_docs,
            embeddings=b_embs.tolist(),
            metadatas=b_meta,
        )
        print(f"  upsert [{i+1:,} ~ {min(i+batch_size, len(ids)):,} / {len(ids):,}]")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def build_vectordb(
    db_path: Path,
    vector_dir: Path,
    model_name: str,
    batch_size: int,
):
    vector_dir.mkdir(parents=True, exist_ok=True)

    print("=== KoE5 임베딩 + ChromaDB 구축 ===")
    print(f"  SQLite  : {db_path}")
    print(f"  VectorDB: {vector_dir}")

    # 1. 청크 로드
    print("\n[1/4] SQLite에서 청크 로드...")
    chunks = load_chunks_from_db(db_path)
    print(f"  총 {len(chunks):,}개 청크")

    # 2. 임베딩 모델 로드
    print(f"\n[2/4] 임베딩 모델 로드: {model_name}")
    embedder = KoE5Embedder(model_name)
    print(f"  임베딩 차원: {embedder.dim}")

    # 3. ChromaDB 컬렉션 생성
    print(f"\n[3/4] ChromaDB 컬렉션 설정...")
    client = chromadb.PersistentClient(
        path=str(vector_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # 기존 컬렉션 삭제 후 재생성 (재빌드 시)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  기존 컬렉션 '{COLLECTION_NAME}' 삭제 후 재생성")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"  컬렉션 생성: '{COLLECTION_NAME}'")

    # 4. 배치 임베딩 + Upsert
    print(f"\n[4/4] 임베딩 및 Upsert (배치={batch_size})...")
    t0 = time.time()

    ids       = [r["chunk_uid"] for r in chunks]
    documents = [r["content"]   for r in chunks]
    metadatas = [make_chroma_metadata(r) for r in chunks]

    upsert_batch(collection, ids, documents, metadatas, embedder, batch_size)

    elapsed = time.time() - t0
    count = collection.count()
    print(f"\n✅ 벡터 DB 완성")
    print(f"   저장된 벡터: {count:,}개")
    print(f"   소요 시간  : {elapsed/60:.1f}분")
    print(f"   저장 경로  : {vector_dir}")

    # 저장 요약 파일
    summary_path = vector_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "embedding_dim": embedder.dim,
            "total_vectors": count,
            "collection_name": COLLECTION_NAME,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
        }, f, ensure_ascii=False, indent=2)
    print(f"   요약 저장  : {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--vector-dir", type=Path, default=VECTOR_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="nlpai-lab/KoE5 (기본) 또는 nlpai-lab/KURE-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    build_vectordb(args.db, args.vector_dir, args.model, args.batch_size)