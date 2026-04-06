"""
structured_query.py - section_path 계층 구조를 활용하는 정밀 검색 Tool

감사보고서 section_path 구조 예시:
  "(첨부)재무제표 > 주석 > 9. 종속기업, 관계기업및공동기업 투자: > 가. 변동 내역"

에이전트가 "2022~2024 이익"처럼 요청하면:
  1. years="2022,2023,2024" 로 연도 필터
  2. section_path_contains="재무제표" 으로 경로 좁힘
  3. chunk_type="Table_Row" 로 수치 데이터 집중
  4. tags="#이익" 로 태그 기반 추가 필터 (DB 태그는 반드시 '#' 프리픽스 포함)

⚠️ 태그 주의사항:
  DB에 저장된 태그는 '#' 프리픽스 형태 (예: #이익, #수익, #비유동자산)
  tags 파라미터에 '#' 없이 입력하면 자동으로 추가됨.
"""

import json
from langchain_core.tools import tool

_db = None


def init_structured_query(db):
    global _db
    _db = db


@tool
def structured_query(
    years: str = "",
    section_path_contains: str = "",
    chunk_type: str = "all",
    is_consolidated: int = -1,
    tags: str = "",
    keyword: str = "",
    limit: int = 30,
) -> str:
    """section_path 계층 구조와 태그를 활용하여 정확한 문서를 검색합니다.

    감사보고서 section_path 구조:
      최상위: "(첨부)재무제표" | "독립된 감사인의 감사보고서"
              | "내부회계관리제도 감사 또는 검토의견" | "외부감사 실시내용"
      주석:   "(첨부)재무제표 > 주석 > N. 주석제목"

    chunk_type:
      - Narrative: 일반 서술 텍스트
      - Note:      주석 설명 텍스트
      - Table_Row: 테이블 행 (재무 수치 데이터)

    Args:
        years:                 연도 (쉼표 구분, 예: "2022,2023,2024" 또는 빈 문자열=전체)
        section_path_contains: section_path 부분 문자열 (예: "주석", "9. 종속기업", "재무제표")
        chunk_type:            Narrative | Note | Table_Row | all
        is_consolidated:       1=연결재무제표, 0=별도재무제표, -1=전체
        tags:                  쉼표 구분 태그. DB 태그는 '#' 프리픽스 포함.
                               (예: "#이익,#수익,#비유동자산"). '#' 없이 입력해도 자동 보정.
                               사용 가능한 태그는 DB 컨텍스트의 '사용 가능한 태그' 섹션 참조
        keyword:               본문 키워드 (부분 매칭, 쉼표로 다중 OR 검색)
        limit:                 최대 결과 수

    Returns:
        매칭된 청크 리스트 JSON (chunk_uid, section_path, chunk_type, fiscal_year 포함)
    """
    if _db is None:
        return json.dumps({"error": "Database not initialized"}, ensure_ascii=False)

    print(
        f"[StructuredQuery] ▶ years='{years}', section_path='{section_path_contains}', "
        f"chunk_type={chunk_type}, is_consolidated={is_consolidated}, "
        f"tags='{tags}', keyword='{keyword}', limit={limit}",
        flush=True,
    )

    # 연도 파싱
    year_list = None
    if years:
        try:
            year_list = [int(y.strip()) for y in years.split(",") if y.strip()]
        except ValueError:
            return json.dumps({"error": f"Invalid years format: {years}"}, ensure_ascii=False)

    # is_consolidated 파싱
    is_consol = None
    if is_consolidated in (0, 1):
        is_consol = is_consolidated

    # tags 파싱 + 정규화 (DB 태그는 '#' 프리픽스 형태로 저장됨)
    tag_list = None
    if tags:
        raw_tags = [t.strip() for t in tags.split(",") if t.strip()]
        # '#' 없이 입력된 태그는 자동으로 '#' 추가
        tag_list = [t if t.startswith("#") else f"#{t}" for t in raw_tags]

    results = _db.structured_search(
        years=year_list,
        section_path_contains=section_path_contains if section_path_contains else None,
        chunk_type=chunk_type if chunk_type != "all" else None,
        is_consolidated=is_consol,
        keyword=keyword if keyword else None,
        tags=tag_list,
        limit=limit,
    )

    print(f"[StructuredQuery] ◀ {len(results)} results", flush=True)
    for r in results[:3]:
        yr = r.get("fiscal_year", "?")
        path = str(r.get("section_path") or "?")[:60]
        ctype = r.get("chunk_type", "?")
        print(f"  year={yr}, type={ctype}, path={path}", flush=True)

    PREVIEW_LIMIT = 2000

    output = []
    for r in results:
        full_content = r["content"]
        content_length = r["content_length"]
        is_truncated = content_length > PREVIEW_LIMIT

        entry = {
            "chunk_uid": r["chunk_uid"],
            "fiscal_year": r["fiscal_year"],
            "period": r.get("period", ""),
            "section_path": r["section_path"],
            "chunk_type": r["chunk_type"],
            "is_consolidated": bool(r["is_consolidated"]),
            "content": full_content[:PREVIEW_LIMIT] + ("...[이하 잘림 — get_full_content 사용]" if is_truncated else ""),
            "content_length": content_length,
            "is_truncated": is_truncated,
            "table_ref": r.get("table_ref", ""),
            "table_unit": r.get("table_unit", ""),
        }
        output.append(entry)

    return json.dumps(output, ensure_ascii=False, indent=2)


@tool
def list_available_sections(year: int = 0, level: int = 3) -> str:
    """DB에 저장된 섹션 목록을 조회합니다. 어떤 섹션이 있는지 파악할 때 사용하세요.

    section_path는 ' > ' 구분자로 계층화된 경로입니다.

    Args:
        year:  연도 (0이면 전체)
        level: 경로 레벨
               1 = 최상위 (예: "(첨부)재무제표", "독립된 감사인의 감사보고서")
               2 = 두 번째 레벨 (예: "주석")
               3 = 세 번째 레벨 — 주석 번호 (예: "9. 종속기업, 관계기업및공동기업 투자:")

    Returns:
        섹션 목록 JSON [{section, year, chunk_count}]
    """
    if _db is None:
        return json.dumps({"error": "Database not initialized"}, ensure_ascii=False)

    results = _db.list_sections(
        year=year if year else None,
        level=level,
    )
    return json.dumps(results, ensure_ascii=False, indent=2)
