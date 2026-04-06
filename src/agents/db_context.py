"""
db_context.py - DB 실제 섹션 구조를 동적으로 조회하여 에이전트 컨텍스트 생성

그래프 초기화 시 한 번만 실행(캐시).
Retriever의 검색 플래닝 프롬프트에 실제 DB 구조를 주입해
LLM이 정확한 section_path_contains 파라미터를 생성하도록 유도.

section_path 구조:
  최상위 > 두번째레벨 > 주석번호 > 세부항목 > ...

포맷 예시:
  ## DB 섹션 구조 (2024년 기준)
  적재된 연도: 2014, 2015, ..., 2024

  ### 2024년
    "(첨부)재무제표" (1138 chunks)
      └ "주석" (sub-level)
          └ 주석목록: "1. 일반적 사항:" (12) | "2. 중요한 회계처리방침" (35) | ...
    "독립된 감사인의 감사보고서" (28 chunks)
    "내부회계관리제도 감사 또는 검토의견" (15 chunks)
    "외부감사 실시내용" (8 chunks)
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
        stats = db.stats()
        available_years = sorted(stats.get("documents_by_year", {}).keys())
        if not available_years:
            return ""

        target_years = years if years else available_years

        lines = ["## DB 실제 섹션 구조"]
        lines.append(f"적재된 연도: {', '.join(str(y) for y in available_years)}")
        lines.append("")

        for year in target_years:
            # 레벨1: 최상위 섹션
            top_sections = db.list_sections(year=year, level=1)
            if not top_sections:
                continue

            lines.append(f"### {year}년")

            for top_row in top_sections:
                top_name = top_row["section"]
                top_count = top_row["chunk_count"]
                lines.append(f'  "{top_name}" ({top_count} chunks)')

                # "(첨부)재무제표" 하위의 주석 번호 목록 제공
                if "(첨부)재무제표" in top_name:
                    note_sections = _get_note_sections(db, year)
                    if note_sections:
                        note_preview = " | ".join(
                            f'"{n}" ({c})' for n, c in note_sections[:20]
                        )
                        lines.append(f'    주석 목록: {note_preview}')
                        if len(note_sections) > 20:
                            lines.append(f'    ... 외 {len(note_sections)-20}개 주석')

            lines.append("")

        # 태그 목록 추가 (major 카테고리 중심으로 LLM에게 제공)
        tag_lines = _build_tag_context(db)
        if tag_lines:
            lines.append(tag_lines)
            lines.append("")

        lines.append("## 검색 파라미터 가이드")
        lines.append("- section_path_contains: section_path 경로의 부분 문자열을 입력")
        lines.append('  예: "주석" → 모든 주석, "9. 종속기업" → 9번 주석, "재무제표" → 재무제표 섹션')
        lines.append("- chunk_type: Narrative(서술) | Note(주석설명) | Table_Row(테이블행/수치)")
        lines.append("  수치 데이터가 필요하면 chunk_type=Table_Row 권장")
        lines.append("- is_consolidated: 현재 DB는 별도재무제표(0)만 적재됨. 항상 -1(전체) 사용 권장")
        lines.append("- tags: 반드시 위 '사용 가능한 태그' 목록의 정확한 값(#포함)만 사용")
        lines.append('  예: "#이익", "#수익", "#비유동자산" — 임의로 만들지 말 것')

        return "\n".join(lines)

    except Exception as e:
        print(f"[DBContext] Failed to build context: {e}")
        return ""


def _build_tag_context(db) -> str:
    """
    DB에서 실제 태그 목록을 조회하여 컨텍스트 문자열 생성.

    major / theme / segment 카테고리를 나눠 제공.
    태그는 DB에 '#' 프리픽스 형태로 저장되어 있음.
    """
    try:
        rows = db.conn.execute(
            """
            SELECT tag, tag_category, COUNT(*) as cnt
            FROM tags
            GROUP BY tag, tag_category
            ORDER BY tag_category, cnt DESC
            """
        ).fetchall()

        if not rows:
            return ""

        by_cat: dict[str, list[str]] = {}
        for row in rows:
            cat = row["tag_category"] or "기타"
            tag = row["tag"]
            # 일부 태그에 공백 포함된 오염 데이터 필터링
            if not tag or " " in tag.strip("#"):
                continue
            by_cat.setdefault(cat, []).append(tag)

        lines = ["## 사용 가능한 태그 (structured_query tags 파라미터에 사용)"]
        lines.append("⚠️ 태그는 반드시 '#' 프리픽스를 포함한 정확한 값을 사용하세요.")
        lines.append("")

        cat_labels = {"major": "주요 계정/항목", "theme": "회계 주제", "segment": "사업 부문"}
        for cat in ("major", "theme", "segment"):
            tags_in_cat = by_cat.get(cat, [])
            if tags_in_cat:
                label = cat_labels.get(cat, cat)
                lines.append(f"[{label}] {', '.join(tags_in_cat)}")

        return "\n".join(lines)

    except Exception as e:
        print(f"[DBContext] Failed to build tag context: {e}")
        return ""


def _get_note_sections(db, year: int) -> list[tuple[str, int]]:
    """
    주석 번호 목록 조회 ((첨부)재무제표 > 주석 하위의 level=3 세그먼트).

    Returns:
        [(note_name, chunk_count), ...]
    """
    try:
        # level=3: section_path의 세 번째 세그먼트 (주석 번호)
        sections = db.list_sections(year=year, level=3)
        # "(첨부)재무제표 > 주석" 하위 항목만 확인
        # list_sections는 전체 level=3 세그먼트를 반환하므로,
        # 실제 주석 번호 형태("N. ...")만 필터링
        note_sections = [
            (s["section"], s["chunk_count"])
            for s in sections
            if s["year"] == year and _is_note_number(s["section"])
        ]
        return sorted(note_sections, key=lambda x: _note_sort_key(x[0]))
    except Exception:
        return []


def _is_note_number(section: str) -> bool:
    """'1. 일반적 사항:' 같은 주석 번호 형태인지 확인"""
    import re
    return bool(re.match(r"^\d+\.", section.strip()))


def _note_sort_key(section: str) -> int:
    """주석 번호 정렬용 키"""
    import re
    m = re.match(r"^(\d+)", section.strip())
    return int(m.group(1)) if m else 999
