"""
check_db.py
──────────────
SQLite + ChromaDB 내용 검증 스크립트

실행:
  python check_db.py
  python check_db.py --db db/audit_reports.db --vector db/vectorstore/chroma
"""

import argparse
import json
import sqlite3
from pathlib import Path

# ── 경로 (필요 시 --db, --vector 인자로 오버라이드) ──────────────────────────
ROOT       = Path(__file__).resolve().parent
DB_PATH    = ROOT / "db" / "audit_reports.db"
VECTOR_DIR = ROOT / "db" / "vectorstore" / "chroma"

SEP = "=" * 60


# ── 1. SQLite 전체 요약 ───────────────────────────────────────────────────────
def check_sqlite(db_path: Path):
    print(f"\n{SEP}")
    print("[ SQLite 검증 ]")
    print(SEP)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # 테이블 목록
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    print(f"테이블 목록: {tables}")

    # ── chunks 테이블
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\n[chunks] 총 {total:,}개")

    print("  ▶ 연도별 청크 수:")
    for row in con.execute(
        "SELECT fiscal_year, COUNT(*) as cnt FROM chunks GROUP BY fiscal_year ORDER BY fiscal_year"
    ).fetchall():
        print(f"    {row['fiscal_year']}년: {row['cnt']:,}개")

    print("  ▶ chunk_type 분포:")
    for row in con.execute(
        "SELECT chunk_type, COUNT(*) as cnt FROM chunks GROUP BY chunk_type ORDER BY cnt DESC"
    ).fetchall():
        print(f"    {row['chunk_type']}: {row['cnt']:,}개")

    print("  ▶ 샘플 청크 (2020년, Table_Row 1개):")
    row = con.execute(
        "SELECT chunk_uid, fiscal_year, section_path, content FROM chunks "
        "WHERE fiscal_year=2020 AND chunk_type='Table_Row' LIMIT 1"
    ).fetchone()
    if row:
        print(f"    uid  : {row['chunk_uid']}")
        print(f"    섹션 : {row['section_path'][:70]}")
        print(f"    내용 : {row['content'][:120]}...")

    # ── tags 테이블
    tag_total = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"\n[tags] 총 {tag_total:,}개")

    print("  ▶ 상위 태그 10개:")
    for row in con.execute(
        "SELECT tag, COUNT(*) as cnt FROM tags GROUP BY tag ORDER BY cnt DESC LIMIT 10"
    ).fetchall():
        print(f"    {row['tag']}: {row['cnt']:,}회")

    # ── report_metadata 테이블
    print(f"\n[report_metadata]")
    for row in con.execute(
        "SELECT fiscal_year, doc_id, period, num_chunks FROM report_metadata ORDER BY fiscal_year"
    ).fetchall():
        print(f"  {row['fiscal_year']}년 | {row['doc_id']} | {row['period']} | {row['num_chunks']:,}청크")

    # ── FTS5 동작 확인
    print(f"\n[FTS5 검색 테스트]  쿼리: '영업이익'")
    try:
        rows = con.execute(
            """SELECT c.fiscal_year, c.chunk_type, c.content
               FROM fts_chunks
               JOIN chunks c ON fts_chunks.rowid = c.id
               WHERE fts_chunks MATCH '영업이익'
               ORDER BY bm25(fts_chunks) LIMIT 3"""
        ).fetchall()
        for r in rows:
            print(f"  [{r['fiscal_year']}년 / {r['chunk_type']}] {r['content'][:100]}...")
    except Exception as e:
        print(f"  FTS 오류: {e}")

    con.close()


# ── 2. ChromaDB 검증 ─────────────────────────────────────────────────────────
def check_chroma(vector_dir: Path):
    print(f"\n{SEP}")
    print("[ ChromaDB 검증 ]")
    print(SEP)

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("  chromadb 미설치 — pip install chromadb")
        return

    # build_summary
    summary_path = vector_dir / "build_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        print(f"  빌드 시각 : {s.get('built_at')}")
        print(f"  임베딩 모델: {s.get('model')}")
        print(f"  임베딩 차원: {s.get('embedding_dim')}")
        print(f"  총 벡터 수 : {s.get('total_vectors', s.get('total', 0)):,}개")
        print(f"  소요 시간  : {s.get('elapsed_sec')}초")
    else:
        print("  build_summary.json 없음")

    client = chromadb.PersistentClient(
        path=str(vector_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collections = client.list_collections()
    print(f"\n  컬렉션 목록: {[c.name for c in collections]}")

    col = client.get_collection("samsung_audit")
    print(f"  벡터 수    : {col.count():,}개")

    # 샘플 조회 (임베딩 없이 메타데이터만)
    sample = col.get(limit=3, include=["metadatas", "documents"])
    print(f"\n  샘플 3개:")
    for i, (uid, meta, doc) in enumerate(
        zip(sample["ids"], sample["metadatas"], sample["documents"]), 1
    ):
        print(f"  [{i}] uid={uid}")
        print(f"       연도={meta.get('fiscal_year')} | 타입={meta.get('chunk_type')}")
        print(f"       내용={doc[:80]}...")

    # 의미 검색 테스트 (모델 없이 — 임의 벡터 대신 get으로 첫 벡터 재사용)
    print(f"\n  ▶ 벡터 검색 테스트 (첫 벡터로 유사 문서 검색):")
    try:
        first_emb = col.get(limit=1, include=["embeddings"])
        if first_emb["embeddings"]:
            res = col.query(
                query_embeddings=[first_emb["embeddings"][0]],
                n_results=3,
                include=["metadatas", "distances"],
            )
            for uid, meta, dist in zip(
                res["ids"][0], res["metadatas"][0], res["distances"][0]
            ):
                print(f"    uid={uid} | 연도={meta.get('fiscal_year')} | cosine_dist={dist:.4f}")
    except Exception as e:
        print(f"  벡터 검색 오류: {e}")


# ── 3. 교차 검증: SQLite ↔ ChromaDB 수 일치 ──────────────────────────────────
def cross_check(db_path: Path, vector_dir: Path):
    print(f"\n{SEP}")
    print("[ 교차 검증: SQLite ↔ ChromaDB ]")
    print(SEP)

    try:
        import chromadb
        from chromadb.config import Settings
        con = sqlite3.connect(db_path)
        sqlite_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()

        client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection("samsung_audit")
        chroma_count = col.count()
        if chroma_count == 0:
            summary_path = vector_dir / "build_summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text())
                chroma_count = s.get("total_vectors", s.get("total", 0))

        print(f"  SQLite  청크 수: {sqlite_count:,}")
        print(f"  ChromaDB 벡터 수: {chroma_count:,}")

        if sqlite_count == chroma_count:
            print("  ✅ 수량 일치 — DB 정상")
        else:
            diff = abs(sqlite_count - chroma_count)
            print(f"  ⚠️  {diff:,}개 차이 — 재빌드 필요할 수 있음")
    except Exception as e:
        print(f"  교차 검증 실패: {e}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB 내용 검증")
    parser.add_argument("--db",     type=Path, default=DB_PATH)
    parser.add_argument("--vector", type=Path, default=VECTOR_DIR)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"❌ SQLite DB 없음: {args.db}")
        raise SystemExit(1)

    check_sqlite(args.db)
    check_chroma(args.vector)
    cross_check(args.db, args.vector)

    print(f"\n{SEP}")
    print("검증 완료")
    print(SEP)