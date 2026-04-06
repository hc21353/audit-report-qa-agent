"""
db_fetch.py - DB 청크 전체 내용 조회 Tool

structured_query/hybrid_search는 내용을 2000자로 잘라서 반환합니다.
이 도구는 is_truncated=true인 청크나 문맥 파악이 필요할 때
청크의 전체 내용과 전후 청크를 가져옵니다.

chunk_uid (예: "SAM_2024_ANNUAL_000064") 또는 integer id 모두 지원합니다.
"""

import json
from langchain_core.tools import tool

_db = None


def init_db_fetch(db):
    global _db
    _db = db


@tool
def get_full_content(
    chunk_ids: str,
    include_adjacent: bool = False,
) -> str:
    """특정 청크의 전체 내용을 가져옵니다.

    structured_query나 hybrid_search 결과에서 is_truncated=true인 청크의
    전체 내용이 필요할 때 사용하세요.

    Args:
        chunk_ids:        조회할 청크 ID (쉼표 구분)
                          chunk_uid 문자열 (예: "SAM_2024_ANNUAL_000064,SAM_2024_ANNUAL_000065")
                          또는 integer id (예: "42,43,44") 모두 지원
        include_adjacent: True이면 각 청크의 이전/다음 청크도 함께 반환

    Returns:
        청크 전체 내용 JSON [{chunk_uid, fiscal_year, section_path, chunk_type,
                             is_consolidated, content, table_ref?, prev_chunk?, next_chunk?}]
    """
    if _db is None:
        return json.dumps({"error": "Database not initialized"}, ensure_ascii=False)

    # chunk_ids 파싱 (chunk_uid 문자열 또는 integer)
    raw_ids = [x.strip() for x in chunk_ids.split(",") if x.strip()]
    if not raw_ids:
        return json.dumps({"error": "No chunk IDs provided"}, ensure_ascii=False)

    if len(raw_ids) > 20:
        return json.dumps({"error": "최대 20개 청크만 한 번에 조회 가능합니다"}, ensure_ascii=False)

    # integer 또는 string 판별
    try:
        ids = [int(x) for x in raw_ids]
    except ValueError:
        ids = raw_ids  # chunk_uid 문자열

    print(f"[GetFullContent] ▶ chunk_ids={raw_ids}, include_adjacent={include_adjacent}", flush=True)

    chunks = _db.get_chunks_by_ids(ids)

    if not chunks:
        return json.dumps({"error": f"No chunks found for IDs: {raw_ids}"}, ensure_ascii=False)

    output = []
    for chunk in chunks:
        entry = {
            "chunk_uid": chunk["chunk_uid"],
            "fiscal_year": chunk["fiscal_year"],
            "period": chunk.get("period", ""),
            "section_path": chunk["section_path"],
            "chunk_type": chunk["chunk_type"],
            "is_consolidated": bool(chunk["is_consolidated"]),
            "content": chunk["content"],
            "content_length": chunk["content_length"],
            "table_ref": chunk.get("table_ref", ""),
            "table_unit": chunk.get("table_unit", ""),
        }

        if chunk.get("row_json"):
            entry["row_data"] = chunk["row_json"]

        if include_adjacent:
            adj = _db.get_adjacent_chunks(chunk["id"])
            if adj["prev"]:
                entry["prev_chunk"] = {
                    "chunk_uid": adj["prev"]["chunk_uid"],
                    "section_path": adj["prev"]["section_path"],
                    "content": adj["prev"]["content"],
                }
            if adj["next"]:
                entry["next_chunk"] = {
                    "chunk_uid": adj["next"]["chunk_uid"],
                    "section_path": adj["next"]["section_path"],
                    "content": adj["next"]["content"],
                }

        output.append(entry)

    print(f"[GetFullContent] ◀ {len(output)} chunks returned", flush=True)
    return json.dumps(output, ensure_ascii=False, indent=2)
