"""
structured_query.py - DB 스키마(h1~h6)를 활용하는 구조 기반 검색 Tool

에이전트가 "2022~2024 영업이익"처럼 요청하면:
  1. years=[2022,2023,2024] 로 연도 필터
  2. section_h2="재무제표" 로 섹션 좁힘
  3. keyword="영업이익" 으로 content 검색
  4. CSV 태그가 있으면 csv_refs 포함하여 반환

에이전트에게 DB 구조 정보도 제공하여 정확한 쿼리를 유도.
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
    section_h2: str = "",
    section_h3: str = "",
    section_h4: str = "",
    section_h5: str = "",
    keyword: str = "",
    content_type: str = "all",
    limit: int = 30,
) -> str:
    """DB의 h1~h6 구조를 활용하여 정확한 문서를 검색합니다.

    감사보고서 구조:
      h2: "독립된 감사인의 감사보고서" | "재무제표" | "주석" | "내부회계관리제도 감사보고서" | "외부감사 실시내용"
      h3: 주석 번호 (예: "1. 일반적 사항", "30. 특수관계자 거래")
      h4: 세부 항목 (예: "2.9 유형자산", "2.15 수익인식")
      h5: 세세부 (예: "가. 보고기간종료일 현재 범주별 금융상품 내역")

    Args:
        years: 연도 (쉼표 구분, 예: "2022,2023,2024" 또는 빈 문자열=전체)
        section_h2: h2 섹션명 (정확 매칭)
        section_h3: h3 섹션명 (부분 매칭)
        section_h4: h4 섹션명 (부분 매칭)
        section_h5: h5 섹션명 (부분 매칭)
        keyword: 본문 키워드 (부분 매칭)
        content_type: text | table | table_csv | mixed | all
        limit: 최대 결과 수

    Returns:
        매칭된 청크 리스트 JSON (section 구조 메타데이터 포함)
    """
    if _db is None:
        return json.dumps({"error": "Database not initialized"}, ensure_ascii=False)

    print(
        f"[StructuredQuery] ▶ years='{years}', h2='{section_h2}', h3='{section_h3}', "
        f"h4='{section_h4}', keyword='{keyword}', content_type={content_type}, limit={limit}",
        flush=True,
    )

    # 연도 파싱
    year_list = None
    if years:
        try:
            year_list = [int(y.strip()) for y in years.split(",") if y.strip()]
        except ValueError:
            return json.dumps({"error": f"Invalid years format: {years}"}, ensure_ascii=False)

    results = _db.structured_search(
        years=year_list,
        section_h2=section_h2 if section_h2 else None,
        section_h3=section_h3 if section_h3 else None,
        section_h4=section_h4 if section_h4 else None,
        section_h5=section_h5 if section_h5 else None,
        keyword=keyword if keyword else None,
        content_type=content_type if content_type != "all" else None,
        limit=limit,
    )

    # CSV 참조 추출
    import re
    csv_re = re.compile(r"\[TABLE_CSV\]\s+(.+\.csv)")

    print(f"[StructuredQuery] ◀ {len(results)} results", flush=True)
    for r in results[:3]:
        print(f"  year={r.get('year')}, h2={r.get('section_h2','?')}, h3={r.get('section_h3','?')[:40]}", flush=True)

    output = []
    for r in results:
        entry = {
            "chunk_id": r["id"],
            "year": r["year"],
            "section_h2": r.get("section_h2", ""),
            "section_h3": r.get("section_h3", ""),
            "section_h4": r.get("section_h4", ""),
            "section_h5": r.get("section_h5", ""),
            "section_path": r["section_path"],
            "content_type": r["content_type"],
            "content": r["content"][:1000],  # 미리보기
            "content_length": r["content_length"],
            "csv_refs": csv_re.findall(r["content"]),
        }
        output.append(entry)

    return json.dumps(output, ensure_ascii=False, indent=2)


@tool
def list_available_sections(year: int = 0, level: int = 3) -> str:
    """DB에 저장된 섹션 목록을 조회합니다. 어떤 섹션이 있는지 파악할 때 사용하세요.

    Args:
        year: 연도 (0이면 전체)
        level: 헤딩 레벨 (2=대분류, 3=주석번호, 4=세부, 5=세세부)

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
