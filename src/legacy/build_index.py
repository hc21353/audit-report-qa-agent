"""
build_index.py - 벡터 인덱스 빌드

documents 테이블의 청크를 임베딩하여 FAISS 또는 Chroma 인덱스를 생성.
config에서 임베딩 모델, 벡터스토어 백엔드, 청킹 전략을 스위칭 가능.

사용법:
  python -m src.build_index                    # 기본 설정으로 빌드
  python -m src.build_index --embedding bge-m3 # 임베딩 모델 지정
  python -m src.build_index --backend chroma   # 벡터스토어 지정
  python -m src.build_index --rebuild          # 기존 인덱스 삭제 후 재빌드
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from src.config import load_config, Config
from src.db import AuditDB


# ─── 임베딩 모델 로더 ───────────────────────────────────────

def load_embedding_model(config: Config, model_name: Optional[str] = None):
    """
    config에 따라 임베딩 모델을 로드.

    Returns:
        (embed_fn, dimension) - embed_fn(texts: list[str]) -> list[list[float]]
    """
    name = model_name or config.active_embedding
    model_cfg = config.app.get("embedding", {}).get("models", {}).get(name, {})

    model_type = model_cfg.get("type", "sentence-transformers")
    model_id = model_cfg.get("name", name)
    dimension = model_cfg.get("dimension", 768)
    prefix_query = model_cfg.get("prefix_query", "")
    prefix_passage = model_cfg.get("prefix_passage", "")
    normalize = model_cfg.get("normalize", True)

    if model_type == "ollama":
        # Ollama 임베딩
        base_url = model_cfg.get("base_url", "http://localhost:11434")
        return _load_ollama_embedding(model_id, base_url, dimension)

    else:
        # sentence-transformers
        return _load_st_embedding(model_id, dimension, prefix_passage, normalize)


def _load_st_embedding(model_id: str, dimension: int, prefix: str, normalize: bool):
    """sentence-transformers 모델 로드"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    print(f"[Embedding] Loaded: {model_id} (dim={dimension})")

    def embed_fn(texts: list[str]) -> list[list[float]]:
        if prefix:
            texts = [prefix + t for t in texts]
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            batch_size=32,
        )
        return embeddings.tolist()

    return embed_fn, dimension


def _load_ollama_embedding(model_id: str, base_url: str, dimension: int):
    """Ollama 임베딩 모델 사용"""
    import requests

    print(f"[Embedding] Using Ollama: {model_id} at {base_url}")

    def embed_fn(texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            resp = requests.post(
                f"{base_url}/api/embeddings",
                json={"model": model_id, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings

    return embed_fn, dimension


# ─── FAISS 인덱스 ───────────────────────────────────────────

def build_faiss_index(
    chunks: list[dict],
    embed_fn,
    dimension: int,
    index_dir: str,
    index_name: str,
):
    """FAISS 인덱스 빌드 및 저장"""
    import faiss
    import numpy as np

    texts = [c["content"] for c in chunks]
    print(f"[FAISS] Embedding {len(texts)} chunks...")

    t0 = time.time()
    embeddings = embed_fn(texts)
    embed_time = time.time() - t0
    print(f"[FAISS] Embedding done in {embed_time:.1f}s")

    # numpy 배열로 변환
    vectors = np.array(embeddings, dtype="float32")

    # 인덱스 생성 (Inner Product for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    # 메타데이터 저장 (FAISS는 메타 저장 기능이 없으므로 별도 JSON)
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": chunk["id"],
            "year": chunk["year"],
            "section_path": chunk["section_path"],
            "content_type": chunk["content_type"],
            "content_length": chunk["content_length"],
        })

    # 저장
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    faiss_file = index_path / f"{index_name}.faiss"
    meta_file = index_path / f"{index_name}.meta.json"

    faiss.write_index(index, str(faiss_file))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"[FAISS] Index saved: {faiss_file} ({index.ntotal} vectors, dim={dimension})")
    return index, metadata


# ─── LangChain FAISS 래퍼 ───────────────────────────────────

def build_langchain_faiss(
    chunks: list[dict],
    embed_fn,
    dimension: int,
    index_dir: str,
    index_name: str,
):
    """LangChain FAISS 벡터스토어 빌드 (에이전트 연동용)"""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # LangChain용 Document 리스트 생성
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "chunk_id": chunk["id"],
                "year": chunk["year"],
                "section_path": chunk["section_path"],
                "content_type": chunk["content_type"],
                "section_h2": chunk.get("section_h2", ""),
                "section_h3": chunk.get("section_h3", ""),
            },
        )
        documents.append(doc)

    print(f"[FAISS] Building LangChain FAISS with {len(documents)} documents...")

    # 임베딩 모델 (LangChain 래퍼)
    # 여기서는 직접 embed_fn을 쓰지 않고 HuggingFaceEmbeddings 사용
    # config에서 모델명을 가져옴

    t0 = time.time()
    vectorstore = FAISS.from_documents(documents, embedding=_embedding_wrapper)
    build_time = time.time() - t0
    print(f"[FAISS] Built in {build_time:.1f}s")

    # 저장
    save_path = Path(index_dir) / index_name
    vectorstore.save_local(str(save_path))
    print(f"[FAISS] Saved to {save_path}")

    return vectorstore


def load_langchain_faiss(index_dir: str, index_name: str, embedding_model):
    """저장된 LangChain FAISS 인덱스 로드"""
    from langchain_community.vectorstores import FAISS

    load_path = Path(index_dir) / index_name
    if not load_path.exists():
        raise FileNotFoundError(f"Index not found: {load_path}")

    vectorstore = FAISS.load_local(
        str(load_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"[FAISS] Loaded from {load_path}")
    return vectorstore


# ─── 임베딩 래퍼 (LangChain 호환) ───────────────────────────

_embedding_wrapper = None


def create_embedding_wrapper(config: Config, model_name: Optional[str] = None):
    """LangChain 호환 임베딩 객체 생성"""
    global _embedding_wrapper

    name = model_name or config.active_embedding
    model_cfg = config.app.get("embedding", {}).get("models", {}).get(name, {})
    model_type = model_cfg.get("type", "sentence-transformers")
    model_id = model_cfg.get("name", name)

    if model_type == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        base_url = model_cfg.get("base_url", "http://localhost:11434")
        _embedding_wrapper = OllamaEmbeddings(
            model=model_id,
            base_url=base_url,
        )
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embedding_wrapper = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": model_cfg.get("normalize", True)},
        )

    print(f"[Embedding] Created LangChain wrapper: {model_id}")
    return _embedding_wrapper


# ─── 메인 ───────────────────────────────────────────────────

def build_index(
    config: Config,
    db: AuditDB,
    embedding_name: Optional[str] = None,
    backend_name: Optional[str] = None,
    rebuild: bool = False,
):
    """
    전체 벡터 인덱스 빌드 파이프라인.

    Args:
        config:         Config 객체
        db:             AuditDB 인스턴스
        embedding_name: 임베딩 모델 (None이면 config 기본값)
        backend_name:   벡터스토어 백엔드 (None이면 config 기본값)
        rebuild:        기존 인덱스 삭제 후 재빌드

    Returns:
        LangChain FAISS 벡터스토어 인스턴스
    """
    emb_name = embedding_name or config.active_embedding
    vs_backend = backend_name or config.app.get("vector_store", {}).get("active_backend", "faiss")
    chunking_strategy = config.active_chunking

    vs_config = config.app.get("vector_store", {}).get("backends", {}).get(vs_backend, {})
    index_dir = vs_config.get("index_dir", f"./db/vector_index/{vs_backend}")
    index_name_template = vs_config.get("index_name_template", "{embedding_model}_{chunking_strategy}")
    index_name = index_name_template.format(
        embedding_model=emb_name.replace("/", "_").replace(":", "_"),
        chunking_strategy=chunking_strategy,
    )

    # 기존 인덱스 확인
    index_path = Path(index_dir) / index_name
    if index_path.exists() and not rebuild:
        print(f"[Index] Found existing index: {index_path}")
        embedding_model = create_embedding_wrapper(config, emb_name)
        return load_langchain_faiss(index_dir, index_name, embedding_model)

    # DB에서 청크 로드
    chunks = db.get_documents()
    if not chunks:
        raise ValueError("No documents in database. Run loader first.")

    print(f"[Index] Building index: {index_name}")
    print(f"  Embedding: {emb_name}")
    print(f"  Backend:   {vs_backend}")
    print(f"  Chunks:    {len(chunks)}")

    # LangChain 임베딩 래퍼 생성
    embedding_model = create_embedding_wrapper(config, emb_name)

    # LangChain FAISS 빌드
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["content"],
            metadata={
                "chunk_id": chunk["id"],
                "year": chunk["year"],
                "section_path": chunk["section_path"],
                "content_type": chunk["content_type"],
                "section_h2": chunk.get("section_h2", ""),
                "section_h3": chunk.get("section_h3", ""),
            },
        )
        documents.append(doc)

    t0 = time.time()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    build_time = time.time() - t0
    print(f"[Index] Built in {build_time:.1f}s ({len(documents)} vectors)")

    # 저장
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"[Index] Saved to {index_path}")

    return vectorstore


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="벡터 인덱스 빌드")
    parser.add_argument("--embedding", type=str, help="임베딩 모델명")
    parser.add_argument("--backend", type=str, help="벡터스토어 백엔드 (faiss|chroma)")
    parser.add_argument("--rebuild", action="store_true", help="기존 인덱스 삭제 후 재빌드")
    parser.add_argument("--db", type=str, help="DB 경로")
    args = parser.parse_args()

    config = load_config()
    db_path = args.db or config.db_path
    db = AuditDB(db_path)

    vectorstore = build_index(
        config=config,
        db=db,
        embedding_name=args.embedding,
        backend_name=args.backend,
        rebuild=args.rebuild,
    )

    # 간단한 검색 테스트
    print("\n[Test] Quick search test...")
    results = vectorstore.similarity_search_with_score("총자산", k=3)
    for doc, score in results:
        year = doc.metadata.get("year", "?")
        path = doc.metadata.get("section_path", "?")
        print(f"  score={score:.4f} year={year} path={path}")
        print(f"  content: {doc.page_content[:100]}...")

    db.close()
    print("\n[Index] Done.")


if __name__ == "__main__":
    main()
