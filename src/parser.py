"""
parser.py - 마크다운 감사보고서 구조 파싱

마크다운 헤딩(#)을 기반으로 문서를 섹션으로 분리하고,
각 섹션에 h1~h6 계층 정보를 부여한다.

이것이 청킹의 Layer 1에 해당한다.
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentMeta:
    """YAML frontmatter에서 추출한 메타데이터"""
    title: str = ""
    company: str = ""
    year: int = 0
    period: str = ""
    source: str = ""
    doc_type: str = ""


@dataclass
class Section:
    """파싱된 하나의 섹션 (최하위 헤딩 단위)"""
    year: int = 0
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    section_h4: Optional[str] = None
    section_h5: Optional[str] = None
    section_h6: Optional[str] = None
    section_path: str = ""
    content: str = ""
    content_type: str = "text"    # text | table | table_csv | mixed
    source_file: str = ""
    csv_refs: list = field(default_factory=list)  # [TABLE_CSV] 경로 리스트

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "section_h1": self.section_h1,
            "section_h2": self.section_h2,
            "section_h3": self.section_h3,
            "section_h4": self.section_h4,
            "section_h5": self.section_h5,
            "section_h6": self.section_h6,
            "section_path": self.section_path,
            "content": self.content,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "csv_refs": self.csv_refs,
        }


# ─── 헤딩 레벨 → 인덱스 매핑 ────────────────────────────────

HEADING_KEYS = ["section_h1", "section_h2", "section_h3",
                "section_h4", "section_h5", "section_h6"]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_RE = re.compile(r"^\|.*\|$")
TABLE_CSV_RE = re.compile(r"^\[TABLE_CSV\]\s+(.+\.csv)\s*$")


def _detect_content_type(text: str, csv_refs: list = None) -> str:
    """텍스트에 마크다운 테이블 또는 CSV 참조가 포함되어 있는지 감지"""
    if csv_refs:
        # CSV 참조가 있는 섹션
        lines = text.strip().split("\n")
        non_empty = [l for l in lines if l.strip() and not TABLE_CSV_RE.match(l.strip())]
        if not non_empty:
            return "table_csv"
        return "mixed"

    lines = text.strip().split("\n")
    table_lines = sum(1 for line in lines if TABLE_RE.match(line.strip()))
    text_lines = len(lines) - table_lines

    if table_lines == 0:
        return "text"
    if text_lines <= 2:  # 테이블 외 텍스트가 거의 없으면
        return "table"
    return "mixed"


def _build_section_path(headings: dict) -> str:
    """현재 헤딩 계층에서 section_path 생성"""
    parts = []
    for key in HEADING_KEYS:
        val = headings.get(key)
        if val:
            parts.append(val)
    return " > ".join(parts)


def _clean_heading(text: str) -> str:
    """헤딩 텍스트 정리 (불필요한 공백 등)"""
    # 한글 사이 공백 제거: "재 무 제 표" → "재무제표"
    # 단, 의미있는 공백은 유지해야 하므로 보수적으로 처리
    return text.strip()


# ─── 메인 파서 ───────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[DocumentMeta, str]:
    """YAML frontmatter를 파싱하고 본문을 반환"""
    meta = DocumentMeta()

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                data = yaml.safe_load(parts[1])
                if isinstance(data, dict):
                    meta.title = data.get("title", "")
                    meta.company = data.get("company", "")
                    meta.year = data.get("year", 0)
                    meta.period = data.get("period", "")
                    meta.source = data.get("source", "")
                    meta.doc_type = data.get("type", "")
            except yaml.YAMLError:
                pass
            body = parts[2]
        else:
            body = text
    else:
        body = text

    return meta, body


def parse_markdown(filepath: str) -> tuple[DocumentMeta, list[Section]]:
    """
    마크다운 파일을 파싱하여 섹션 리스트를 반환한다.

    Layer 1 청킹: 최하위 헤딩 단위로 분리.
    각 섹션은 모든 상위 헤딩 정보를 DB 컬럼용으로 보유.

    Args:
        filepath: 마크다운 파일 경로

    Returns:
        (DocumentMeta, list[Section])
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    source_file = path.name

    # 1. frontmatter 파싱
    meta, body = parse_frontmatter(text)

    # 2. 라인 단위로 순회하며 헤딩 기반 분할
    lines = body.split("\n")
    sections: list[Section] = []

    # 현재 활성화된 헤딩 계층
    current_headings = {key: None for key in HEADING_KEYS}
    current_content_lines: list[str] = []
    current_csv_refs: list[str] = []

    def _flush_section():
        """현재까지 쌓인 content를 Section으로 저장"""
        content = "\n".join(current_content_lines).strip()
        if not content and not current_csv_refs:
            return

        section = Section(
            year=meta.year,
            section_h1=current_headings["section_h1"],
            section_h2=current_headings["section_h2"],
            section_h3=current_headings["section_h3"],
            section_h4=current_headings["section_h4"],
            section_h5=current_headings["section_h5"],
            section_h6=current_headings["section_h6"],
            section_path=_build_section_path(current_headings),
            content=content,
            content_type=_detect_content_type(content, current_csv_refs),
            source_file=source_file,
            csv_refs=list(current_csv_refs),
        )
        sections.append(section)

    for line in lines:
        heading_match = HEADING_RE.match(line.strip())
        csv_match = TABLE_CSV_RE.match(line.strip())

        if heading_match:
            # 이전 섹션 저장
            _flush_section()
            current_content_lines = []
            current_csv_refs = []

            # 헤딩 레벨 파악 (# = 1, ## = 2, ...)
            level = len(heading_match.group(1))
            heading_text = _clean_heading(heading_match.group(2))
            heading_key = f"section_h{level}"

            # 해당 레벨 업데이트
            if heading_key in current_headings:
                current_headings[heading_key] = heading_text

                # 하위 레벨 초기화 (새 상위 헤딩이 나오면 하위는 리셋)
                for i in range(level + 1, 7):
                    current_headings[f"section_h{i}"] = None
        elif csv_match:
            # [TABLE_CSV] 태그 → CSV 참조로 수집
            csv_path = csv_match.group(1).strip()
            current_csv_refs.append(csv_path)
            current_content_lines.append(line)  # 원본 태그도 content에 보존
        else:
            current_content_lines.append(line)

    # 마지막 섹션 저장
    _flush_section()

    print(f"[Parser] {source_file}: {len(sections)} sections parsed")
    return meta, sections


def parse_directory(dir_path: str) -> list[tuple[DocumentMeta, list[Section]]]:
    """디렉토리 내 모든 .md 파일을 파싱"""
    results = []
    md_dir = Path(dir_path)
    if not md_dir.exists():
        print(f"[Parser] Directory not found: {dir_path}")
        return results

    for md_file in sorted(md_dir.glob("*.md")):
        meta, sections = parse_markdown(str(md_file))
        results.append((meta, sections))

    return results


# ─── 디버그 출력 ─────────────────────────────────────────────

def print_sections_summary(sections: list[Section]):
    """섹션 요약 출력"""
    for i, sec in enumerate(sections):
        content_preview = sec.content[:80].replace("\n", " ")
        print(f"  [{i:3d}] {sec.section_path}")
        print(f"        type={sec.content_type}, len={len(sec.content)}, "
              f"preview: {content_preview}...")
        print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parser.py <markdown_file>")
        sys.exit(1)

    meta, sections = parse_markdown(sys.argv[1])
    print(f"\nMeta: year={meta.year}, company={meta.company}")
    print(f"Total sections: {len(sections)}\n")
    print_sections_summary(sections[:20])
