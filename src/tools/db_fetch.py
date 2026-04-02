"""
db_fetch.py - DB 청크 전체 내용 조회 Tool

structured_query/hybrid_search는 내용을 1500자로 잘라서 반환합니다.
이 도구는 content_length > 1500인 청크나 문맥 파악이 필요할 때
청크의 전체 내용과 전후 청크를 가져옵니다.
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

    structured_query나 hybrid_search 결과에서 content가 잘렸거나
    (content_length가 content 길이보다 훨씬 크면 잘린 것),
    전후 맥락이 필요할 때 사용하세요.

    Args:
        chunk_ids:        조회할 청크 ID (쉼표 구분, 예: "42,43,44")
        include_adjacent: True이면 각 청크의 이전/다음 청크도 함께 반환

    Returns:
        청크 전체 내용 JSON [{chunk_id, year, section_path, chunk_index,
                              total_chunks, content, prev_chunk?, next_chunk?}]
    """
    if _db is None:
        return json.dumps({"error": "Database not initialized"}, ensure_ascii=False)

    # chunk_ids 파싱
    try:
        ids = [int(x.strip()) for x in chunk_ids.split(",") if x.strip()]
    except ValueError:
        return json.dumps({"error": f"Invalid chunk_ids format: {chunk_ids}"}, ensure_ascii=False)

    if not ids:
        return json.dumps({"error": "No chunk IDs provided"}, ensure_ascii=False)

    if len(ids) > 20:
        return json.dumps({"error": "최대 20개 청크만 한 번에 조회 가능합니다"}, ensure_ascii=False)

    print(f"[GetFullContent] ▶ chunk_ids={ids}, include_adjacent={include_adjacent}", flush=True)

    chunks = _db.get_chunks_by_ids(ids)

    if not chunks:
        return json.dumps({"error": f"No chunks found for IDs: {ids}"}, ensure_ascii=False)

    output = []
    for chunk in chunks:
        entry = {
            "chunk_id": chunk["id"],
            "year": chunk["year"],
            "section_h2": chunk.get("section_h2", ""),
            "section_h3": chunk.get("section_h3", ""),
            "section_path": chunk["section_path"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "content_type": chunk["content_type"],
            "content": chunk["content"],       # 전체 내용 (잘리지 않음)
            "content_length": chunk["content_length"],
        }

        if include_adjacent:
            adj = _db.get_adjacent_chunks(chunk["id"])
            if adj["prev"]:
                entry["prev_chunk"] = {
                    "chunk_id": adj["prev"]["id"],
                    "chunk_index": adj["prev"]["chunk_index"],
                    "content": adj["prev"]["content"],
                }
            if adj["next"]:
                entry["next_chunk"] = {
                    "chunk_id": adj["next"]["id"],
                    "chunk_index": adj["next"]["chunk_index"],
                    "content": adj["next"]["content"],
                }

        output.append(entry)

    print(f"[GetFullContent] ◀ {len(output)} chunks returned", flush=True)
    return json.dumps(output, ensure_ascii=False, indent=2)
