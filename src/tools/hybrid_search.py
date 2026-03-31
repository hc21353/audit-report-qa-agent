"""
hybrid_search.py - 벡터 + BM25 하이브리드 검색

두 검색 결과를 Reciprocal Rank Fusion(RRF)으로 결합.
  - 벡터: 의미 유사도 (FAISS)
  - BM25: 키워드 매칭 (SQLite FTS5)

RRF score = Σ 1 / (k + rank_i)  for each retriever i
"""

import json
from langchain_core.tools import tool

_vector_store = None
_db = None
_rrf_k = 60  # RRF 하이퍼파라미터 (기본값 60)


def init_hybrid_search(vector_store, db, rrf_k: int = 60):
    global _vector_store, _db, _rrf_k
    _vector_store = vector_store
    _db = db
    _rrf_k = rrf_k


@tool
def hybrid_search(
    query: str,
    top_k: int = 10,
    year: int = 0,
    section_h2: str = "",
    mode: str = "hybrid",
) -> str:
    """벡터 유사도 + BM25 키워드 검색을 결합한 하이브리드 검색입니다.

    Args:
        query: 검색 쿼리
        top_k: 최종 반환할 결과 수
        year: 연도 필터 (0이면 전체)
        section_h2: h2 섹션 필터 (빈 문자열이면 전체. 예: '주석', '재무제표')
        mode: "hybrid" (벡터+BM25) | "vector" (벡터만) | "bm25" (키워드만)

    Returns:
        검색 결과 JSON. 각 결과에 vector_rank, bm25_rank, rrf_score 포함.
    """
    candidates = {}  # chunk_id → result dict

    # ─── 벡터 검색 ───────────────────────────────────────────
    if mode in ("hybrid", "vector") and _vector_store is not None:
        fetch_k = top_k * 3
        try:
            vec_results = _vector_store.similarity_search_with_score(query, k=fetch_k)

            rank = 0
            for doc, score in vec_results:
                meta = doc.metadata
                # 메타데이터 기반 필터링
                if year and meta.get("year") != year:
                    continue
                if section_h2 and meta.get("section_h2") != section_h2:
                    continue

                rank += 1
                cid = meta.get("chunk_id", id(doc))

                if cid not in candidates:
                    candidates[cid] = {
                        "chunk_id": cid,
                        "year": meta.get("year"),
                        "section_path": meta.get("section_path", ""),
                        "section_h2": meta.get("section_h2", ""),
                        "section_h3": meta.get("section_h3", ""),
                        "content_type": meta.get("content_type", "text"),
                        "content": doc.page_content,
                        "vector_score": round(float(score), 4),
                        "vector_rank": rank,
                        "bm25_rank": 0,
                        "rrf_score": 0.0,
                    }
                else:
                    candidates[cid]["vector_score"] = round(float(score), 4)
                    candidates[cid]["vector_rank"] = rank
        except Exception as e:
            if mode == "vector":
                return json.dumps({"error": f"Vector search failed: {e}"}, ensure_ascii=False)

    # ─── BM25 키워드 검색 ────────────────────────────────────
    if mode in ("hybrid", "bm25") and _db is not None:
        try:
            bm25_results = _db.bm25_search(
                query=query,
                top_k=top_k * 3,
                year=year if year else None,
                section_h2=section_h2 if section_h2 else None,
            )

            for rank, row in enumerate(bm25_results, 1):
                cid = row["id"]
                if cid not in candidates:
                    candidates[cid] = {
                        "chunk_id": cid,
                        "year": row["year"],
                        "section_path": row["section_path"],
                        "section_h2": row.get("section_h2", ""),
                        "section_h3": row.get("section_h3", ""),
                        "content_type": row["content_type"],
                        "content": row["content"],
                        "vector_score": 0.0,
                        "vector_rank": 0,
                        "bm25_rank": rank,
                        "rrf_score": 0.0,
                    }
                else:
                    candidates[cid]["bm25_rank"] = rank
        except Exception as e:
            if mode == "bm25":
                return json.dumps({"error": f"BM25 search failed: {e}"}, ensure_ascii=False)

    # ─── RRF 스코어 계산 ─────────────────────────────────────
    for cid, r in candidates.items():
        rrf = 0.0
        if r["vector_rank"] > 0:
            rrf += 1.0 / (_rrf_k + r["vector_rank"])
        if r["bm25_rank"] > 0:
            rrf += 1.0 / (_rrf_k + r["bm25_rank"])
        r["rrf_score"] = round(rrf, 6)

    # 정렬 및 상위 k개
    sorted_results = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    top_results = sorted_results[:top_k]

    # content 길이 제한 (LLM 토큰 절약)
    for r in top_results:
        if len(r["content"]) > 1500:
            r["content"] = r["content"][:1500] + "...[truncated]"

    return json.dumps(top_results, ensure_ascii=False, indent=2)
