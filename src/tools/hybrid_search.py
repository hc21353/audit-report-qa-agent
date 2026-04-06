"""
hybrid_search.py - 벡터 + BM25 하이브리드 검색

두 검색 결과를 Reciprocal Rank Fusion(RRF)으로 결합.
  - 벡터: 의미 유사도 (ChromaDB + KoE5)
  - BM25: 키워드 매칭 (SQLite fts_chunks)

RRF score = Σ 1 / (k + rank_i)  for each retriever i
"""

import json
from langchain_core.tools import tool

_collection = None   # ChromaDB collection
_embedder = None     # KoE5Embedder (encode_query 메서드 보유)
_db = None
_rrf_k = 60  # RRF 하이퍼파라미터


def init_hybrid_search(collection, embedder, db, rrf_k: int = 60):
    """
    Args:
        collection: chromadb.Collection 인스턴스 (samsung_audit)
        embedder:   KoE5Embedder — encode_query(str) → np.ndarray
        db:         AuditDB 인스턴스 (BM25용)
        rrf_k:      RRF 하이퍼파라미터
    """
    global _collection, _embedder, _db, _rrf_k
    _collection = collection
    _embedder = embedder
    _db = db
    _rrf_k = rrf_k


@tool
def hybrid_search(
    query: str,
    top_k: int = 10,
    year: int = 0,
    section_path_contains: str = "",
    chunk_type: str = "",
    mode: str = "hybrid",
) -> str:
    """벡터 유사도 + BM25 키워드 검색을 결합한 하이브리드 검색입니다.

    Args:
        query:                검색 쿼리
        top_k:                최종 반환할 결과 수
        year:                 연도 필터 (0이면 전체)
        section_path_contains: section_path 부분 문자열 필터
                              (예: "주석", "9. 종속", "재무제표")
        chunk_type:           청크 타입 필터 ("Narrative"|"Note"|"Table_Row"|"" 전체)
        mode:                 "hybrid" (벡터+BM25) | "vector" (벡터만) | "bm25" (키워드만)

    Returns:
        검색 결과 JSON. 각 결과에 chunk_uid, fiscal_year, section_path,
        chunk_type, vector_rank, bm25_rank, rrf_score 포함.
    """
    print(
        f"[HybridSearch] ▶ query='{query}', top_k={top_k}, year={year}, "
        f"section_path='{section_path_contains}', chunk_type='{chunk_type}', mode={mode}",
        flush=True,
    )
    candidates: dict = {}  # chunk_uid → result dict

    # ─── 벡터 검색 (ChromaDB) ────────────────────────────────
    if mode in ("hybrid", "vector") and _collection is not None and _embedder is not None:
        fetch_k = top_k * 3
        try:
            query_embedding = _embedder.encode_query(query)

            # fiscal_year 필터 (ChromaDB where절)
            where = {"fiscal_year": {"$eq": year}} if year else None

            results = _collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=fetch_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            ids_list      = results["ids"][0]
            docs_list     = results["documents"][0]
            metas_list    = results["metadatas"][0]
            dists_list    = results["distances"][0]  # cosine space: dist = 1 - cos_sim

            rank = 0
            for chunk_uid, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
                # section_path 후처리 필터 (ChromaDB where는 exact match만 지원)
                if section_path_contains and section_path_contains not in meta.get("section_path", ""):
                    continue
                if chunk_type and meta.get("chunk_type", "") != chunk_type:
                    continue

                rank += 1
                score = round(1.0 - float(dist), 4)

                candidates[chunk_uid] = {
                    "chunk_uid": chunk_uid,
                    "fiscal_year": meta.get("fiscal_year"),
                    "section_path": meta.get("section_path", ""),
                    "chunk_type": meta.get("chunk_type", ""),
                    "is_consolidated": meta.get("is_consolidated", False),
                    "content": doc,
                    "vector_score": score,
                    "vector_rank": rank,
                    "bm25_rank": 0,
                    "rrf_score": 0.0,
                }

        except Exception as e:
            if mode == "vector":
                return json.dumps({"error": f"Vector search failed: {e}"}, ensure_ascii=False)

    # ─── BM25 키워드 검색 ─────────────────────────────────────
    if mode in ("hybrid", "bm25") and _db is not None:
        try:
            bm25_results = _db.bm25_search(
                query=query,
                top_k=top_k * 3,
                year=year if year else None,
                section_path_contains=section_path_contains if section_path_contains else None,
                chunk_type=chunk_type if chunk_type else None,
            )

            for rank, row in enumerate(bm25_results, 1):
                cuid = row["chunk_uid"]
                if cuid not in candidates:
                    candidates[cuid] = {
                        "chunk_uid": cuid,
                        "fiscal_year": row["fiscal_year"],
                        "section_path": row["section_path"],
                        "chunk_type": row["chunk_type"],
                        "is_consolidated": bool(row["is_consolidated"]),
                        "content": row["content"],
                        "vector_score": 0.0,
                        "vector_rank": 0,
                        "bm25_rank": rank,
                        "rrf_score": 0.0,
                    }
                else:
                    candidates[cuid]["bm25_rank"] = rank

        except Exception as e:
            if mode == "bm25":
                return json.dumps({"error": f"BM25 search failed: {e}"}, ensure_ascii=False)

    # ─── RRF 스코어 계산 ──────────────────────────────────────
    for cuid, r in candidates.items():
        rrf = 0.0
        if r["vector_rank"] > 0:
            rrf += 1.0 / (_rrf_k + r["vector_rank"])
        if r["bm25_rank"] > 0:
            rrf += 1.0 / (_rrf_k + r["bm25_rank"])
        r["rrf_score"] = round(rrf, 6)

    sorted_results = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    top_results = sorted_results[:top_k]

    # content 길이 제한
    PREVIEW_LIMIT = 2000
    for r in top_results:
        content_len = len(r["content"])
        if content_len > PREVIEW_LIMIT:
            r["content"] = r["content"][:PREVIEW_LIMIT] + "...[이하 잘림 — get_full_content 사용]"
            r["is_truncated"] = True
        else:
            r["is_truncated"] = False

    vec_count = sum(1 for r in candidates.values() if r.get("vector_rank", 0) > 0)
    bm25_count = sum(1 for r in candidates.values() if r.get("bm25_rank", 0) > 0)
    print(
        f"[HybridSearch] ◀ total_candidates={len(candidates)} "
        f"(vector={vec_count}, bm25={bm25_count}) → top_k={len(top_results)}",
        flush=True,
    )
    return json.dumps(top_results, ensure_ascii=False, indent=2)
