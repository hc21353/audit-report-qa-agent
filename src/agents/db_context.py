"""
db_context.py - DB 실제 섹션 구조를 동적으로 조회하여 에이전트 컨텍스트 생성

그래프 초기화 시 한 번만 실행(캐시).
Retriever의 검색 플래닝 프롬프트에 실제 DB 구조를 주입해
LLM이 정확한 section_h2/h3/h4 파라미터를 생성하도록 유도.

포맷 예시:
  ## DB 섹션 구조 (2024년 기준)
  h2: 독립된 감사인의 감사보고서 (4 chunks)
    └ keyword 검색 권장
  h2: 재무제표 (1 chunks)
    └ [TABLE_CSV] 태그로 수치 데이터 포함
  h2: 주석 (136 chunks)
    h3: 1. 일반적 사항 (1) | 2. 중요한 회계처리방침 (25) | ...
  ...
"""

from __future__ import annotations
from typing import Optional


def build_db_context(db, years: Optional[list[int]] = None) -> str:
    """
    DB의 실제 섹션 구조를 조회하여 컨텍스트 문자열 생성.

    Args:
        db:    AuditDB 인스턴스
        years: 조회할 연도 리스트 (None이면 전체)

    Returns:
        플래닝 프롬프트에 삽입할 구조 문자열
    """
    if db is None:
        return ""

    try:
        # 연도 목록
        stats = db.stats()
        available_years = sorted(stats.get("documents_by_year", {}).keys())
        if not available_years:
            return ""

        target_years = years if years else available_years

        lines = ["## DB 실제 섹션 구조"]
        lines.append(f"적재된 연도: {', '.join(str(y) for y in available_years)}")
        lines.append("")

        # 연도별 h2 → h3 구조 구성
        for year in target_years:
            h2_sections = db.list_sections(year=year, level=2)
            if not h2_sections:
                continue

            lines.append(f"### {year}년")

            # h2별로 h3 섹션 그룹화
            h3_by_h2 = _group_h3_by_h2(db, year)

            for h2_row in h2_sections:
                h2_name = h2_row["section"]
                h2_count = h2_row["chunk_count"]
                lines.append(f'  h2: "{h2_name}" ({h2_count} chunks)')

                h3_list = h3_by_h2.get(h2_name, [])
                if h3_list:
                    # h3가 많으면 압축 표시
                    if len(h3_list) <= 8:
                        for h3_name, h3_count in h3_list:
                            lines.append(f'      h3: "{h3_name}" ({h3_count})')
                    else:
                        # 상위 6개 + 나머지 개수
                        for h3_name, h3_count in h3_list[:6]:
                            lines.append(f'      h3: "{h3_name}" ({h3_count})')
                        lines.append(f'      ... 외 {len(h3_list) - 6}개 h3 섹션')
                else:
                    lines.append(f'      (h3 없음 — keyword 검색 권장)')

            lines.append("")

        lines.append("## 검색 파라미터 가이드")
        lines.append("- section_h2: 위 h2 이름을 정확히 입력 (공백/오타 주의)")
        lines.append('- section_h3: 위 h3 이름 또는 부분 문자열 (예: "특수관계자")')
        lines.append("- section_h4/h5: 매우 세부적. 확실하지 않으면 빈 문자열로 남길 것")
        lines.append('- keyword: h2+h3로 범위를 좁힌 후 추가 필터링에 사용')

        return "\n".join(lines)

    except Exception as e:
        print(f"[DBContext] Failed to build context: {e}")
        return ""


def _group_h3_by_h2(db, year: int) -> dict[str, list[tuple[str, int]]]:
    """
    h3 섹션을 h2별로 그룹화.

    Returns:
        {h2_name: [(h3_name, chunk_count), ...]}
    """
    h3_sections = db.list_sections(year=year, level=3)

    # h3 섹션이 어느 h2에 속하는지 확인하려면
    # documents 테이블에서 (section_h2, section_h3, count) 조회
    sql = """
        SELECT section_h2, section_h3, COUNT(*) as cnt
        FROM documents
        WHERE year = ? AND section_h3 IS NOT NULL
        GROUP BY section_h2, section_h3
        ORDER BY section_h2, section_h3
    """
    try:
        rows = db.conn.execute(sql, (year,)).fetchall()
    except Exception:
        return {}

    result: dict[str, list[tuple[str, int]]] = {}
    for row in rows:
        h2 = row["section_h2"] or ""
        h3 = row["section_h3"] or ""
        cnt = row["cnt"]
        if h2 not in result:
            result[h2] = []
        result[h2].append((h3, cnt))

    return result
