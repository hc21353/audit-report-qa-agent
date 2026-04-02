"""
chunker.py - 2레이어 청킹 시스템

Layer 1 (parser.py): 마크다운 헤딩 기반 섹션 분리 → 이미 완료
Layer 2 (이 파일):   최하위 섹션 content가 max_chunk_size 초과 시 추가 분할

전략:
  - section_based: 구조 보존 + 긴 content만 분할 (메인)
  - fixed_size:    구조 무시, 글자수 분할 (비교 베이스라인)
  - recursive:     구분자 기반 분할 (비교 베이스라인)
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.parser import Section, TABLE_RE, TABLE_CSV_RE


@dataclass
class Chunk:
    """최종 DB 삽입 단위"""
    year: int = 0
    section_h1: Optional[str] = None
    section_h2: Optional[str] = None
    section_h3: Optional[str] = None
    section_h4: Optional[str] = None
    section_h5: Optional[str] = None
    section_h6: Optional[str] = None
    section_path: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    content_type: str = "text"
    content: str = ""
    source_file: str = ""
    chunking_strategy: str = ""

    def to_db_row(self) -> dict:
        return {
            "year": self.year,
            "section_h1": self.section_h1,
            "section_h2": self.section_h2,
            "section_h3": self.section_h3,
            "section_h4": self.section_h4,
            "section_h5": self.section_h5,
            "section_h6": self.section_h6,
            "section_path": self.section_path,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "content_type": self.content_type,
            "content": self.content,
            "source_file": self.source_file,
            "chunking_strategy": self.chunking_strategy,
        }


# ─── 유틸리티 ────────────────────────────────────────────────

def _is_table_block(text: str) -> bool:
    """텍스트가 마크다운 테이블인지 판별"""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if TABLE_RE.match(l.strip()))
    return table_lines / len(lines) > 0.5


def _split_text_and_tables(text: str) -> list[tuple[str, str]]:
    """
    텍스트를 (content, type) 블록으로 분리.
    연속된 테이블 라인은 하나의 'table' 블록으로, 나머지는 'text' 블록으로.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_type = "text"

    for line in lines:
        is_table_line = TABLE_RE.match(line.strip()) if line.strip() else False

        if is_table_line and current_type == "text" and current_lines:
            # 텍스트 블록 종료, 테이블 시작
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append((content, "text"))
            current_lines = [line]
            current_type = "table"
        elif not is_table_line and current_type == "table" and line.strip():
            # 테이블 블록 종료, 텍스트 시작
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append((content, "table"))
            current_lines = [line]
            current_type = "text"
        else:
            current_lines.append(line)

    # 마지막 블록
    content = "\n".join(current_lines).strip()
    if content:
        blocks.append((content, current_type))

    return blocks


def _recursive_split(text: str, max_size: int, overlap: int,
                     separators: list[str]) -> list[str]:
    """재귀적 텍스트 분할"""
    if len(text) <= max_size:
        return [text]

    chunks = []
    for sep in separators:
        parts = text.split(sep)
        if len(parts) <= 1:
            continue

        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) > max_size and current:
                chunks.append(current.strip())
                # 오버랩: 이전 청크 끝부분 포함
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + sep + part
                else:
                    current = part
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return chunks

    # 어떤 구분자로도 안 쪼개지면 강제 분할
    chunks = []
    for i in range(0, len(text), max_size - overlap):
        chunk = text[i:i + max_size]
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


# ─── 재무제표 특수 처리 ──────────────────────────────────────

# 재무제표 내 개별 표 이름 매핑 (공백 포함 표기 → 정규 표기)
_FS_NAME_PATTERNS = [
    (re.compile(r"재\s*무\s*상\s*태\s*표"), "재무상태표"),
    (re.compile(r"포\s*괄\s*손\s*익\s*계\s*산\s*서"), "포괄손익계산서"),  # 손익계산서보다 먼저 매칭
    (re.compile(r"손\s*익\s*계\s*산\s*서"), "손익계산서"),
    (re.compile(r"자\s*본\s*변\s*동\s*표"), "자본변동표"),
    (re.compile(r"현\s*금\s*흐\s*름\s*표"), "현금흐름표"),
    (re.compile(r"이\s*익\s*잉\s*여\s*금\s*처\s*분\s*계\s*산\s*서"), "이익잉여금처분계산서"),
]


def _split_financial_statements(section: Section) -> list[Section]:
    """
    재무제표(h2) 섹션을 개별 재무제표(재무상태표, 손익계산서 등)로 분할.

    재무제표 섹션은 h3 헤딩 없이 "> 재 무 상 태 표" 같은 blockquote 패턴 +
    [TABLE_CSV] 태그로 구성되어 있으므로, [TABLE_CSV] 경계에서 분할하고
    앞의 텍스트에서 재무제표 이름을 추출하여 synthetic h3로 설정.
    """
    content = section.content
    if not content or "[TABLE_CSV]" not in content:
        return [section]

    # [TABLE_CSV] 위치를 기준으로 블록 분할
    lines = content.split("\n")
    blocks = []       # [(text_lines, csv_line)]
    current_text = []

    for line in lines:
        if TABLE_CSV_RE.match(line.strip()):
            blocks.append((list(current_text), line))
            current_text = []
        else:
            current_text.append(line)

    # 남은 텍스트가 있으면 (표지 부분 등) 별도 블록으로
    trailing_text = "\n".join(current_text).strip()

    sub_sections = []

    for text_lines, csv_line in blocks:
        text_block = "\n".join(text_lines).strip()

        # 재무제표 이름 추출
        h3_name = None
        combined = text_block + " " + csv_line
        for pattern, name in _FS_NAME_PATTERNS:
            if pattern.search(combined):
                h3_name = name
                break

        if not h3_name:
            h3_name = "기타재무제표"

        sub_content = (text_block + "\n\n" + csv_line).strip()
        csv_refs = TABLE_CSV_RE.findall(csv_line.strip())

        sub = Section(
            year=section.year,
            section_h1=section.section_h1,
            section_h2=section.section_h2,
            section_h3=h3_name,
            section_h4=section.section_h4,
            section_h5=section.section_h5,
            section_h6=section.section_h6,
            section_path=f"{section.section_path} > {h3_name}" if section.section_path else h3_name,
            content=sub_content,
            content_type="table_csv" if csv_refs else "text",
            source_file=section.source_file,
            csv_refs=csv_refs,
        )
        sub_sections.append(sub)

    # 표지 텍스트 (첫 [TABLE_CSV] 이전 텍스트가 모든 블록 앞에 있는 경우)
    # blocks의 첫 번째 text_block이 표지면 별도 섹션으로
    if trailing_text:
        trail = Section(
            year=section.year,
            section_h1=section.section_h1,
            section_h2=section.section_h2,
            section_h3=section.section_h3,
            section_h4=section.section_h4,
            section_h5=section.section_h5,
            section_h6=section.section_h6,
            section_path=section.section_path,
            content=trailing_text,
            content_type="text",
            source_file=section.source_file,
        )
        sub_sections.append(trail)

    return sub_sections if sub_sections else [section]


# ─── 메인 청킹 전략 ─────────────────────────────────────────

def chunk_section_based(
    sections: list[Section],
    max_chunk_size: int = 1500,
    min_chunk_size: int = 100,
    sub_method: str = "recursive",
    chunk_overlap: int = 200,
    separators: list[str] = None,
    table_handling: str = "keep_whole",
    table_max_size: int = 3000,
) -> list[Chunk]:
    """
    메인 전략: section_based
    - 짧은 섹션: 그대로 1청크
    - 긴 섹션: sub_method로 추가 분할
    - 테이블: table_handling에 따라 처리
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    all_chunks: list[Chunk] = []

    # 재무제표 섹션 특수 처리: h3 없이 여러 재무제표가 합쳐진 경우 분할
    expanded_sections = []
    for section in sections:
        if (section.section_h2 == "재무제표"
                and section.section_h3 is None
                and section.csv_refs):
            expanded_sections.extend(_split_financial_statements(section))
        else:
            expanded_sections.append(section)

    for section in expanded_sections:
        content = section.content.strip()
        if not content or len(content) < min_chunk_size:
            # 너무 짧은 섹션은 그대로 저장 (빈 건 스킵)
            if content:
                chunk = _section_to_chunk(section, 0, 1, "section_based")
                all_chunks.append(chunk)
            continue

        # 테이블과 텍스트 분리
        if section.content_type in ("table", "mixed", "table_csv"):
            blocks = _split_text_and_tables(content)
        else:
            blocks = [(content, "text")]

        # 각 블록 처리
        sub_chunks: list[tuple[str, str]] = []  # (content, type)
        for block_content, block_type in blocks:
            if block_type == "table":
                if table_handling == "keep_whole" or len(block_content) <= table_max_size:
                    sub_chunks.append((block_content, "table"))
                else:
                    # 큰 테이블은 행 단위로 분할 가능 (현재는 그대로 유지)
                    sub_chunks.append((block_content, "table"))
            else:
                if len(block_content) <= max_chunk_size:
                    sub_chunks.append((block_content, "text"))
                else:
                    # Layer 2: 추가 분할
                    split_texts = _recursive_split(
                        block_content, max_chunk_size, chunk_overlap, separators
                    )
                    for st in split_texts:
                        sub_chunks.append((st, "text"))

        # Chunk 객체 생성
        total = len(sub_chunks)
        for idx, (chunk_content, chunk_type) in enumerate(sub_chunks):
            chunk = _section_to_chunk(section, idx, total, "section_based")
            chunk.content = chunk_content
            # [TABLE_CSV] 태그가 있으면 content_type을 table_csv/mixed로 보정
            if "[TABLE_CSV]" in chunk_content:
                non_csv_lines = [l for l in chunk_content.split("\n")
                                 if l.strip() and not TABLE_CSV_RE.match(l.strip())]
                chunk.content_type = "mixed" if non_csv_lines else "table_csv"
            else:
                chunk.content_type = chunk_type
            all_chunks.append(chunk)

    return all_chunks


def chunk_fixed_size(
    sections: list[Section],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    비교 베이스라인: 모든 섹션을 이어붙인 후 고정 크기로 분할.
    DB 메타데이터는 원본 위치 기준으로 매핑.
    """
    all_chunks: list[Chunk] = []

    for section in sections:
        content = section.content.strip()
        if not content:
            continue

        if len(content) <= chunk_size:
            chunk = _section_to_chunk(section, 0, 1, "fixed_size")
            all_chunks.append(chunk)
        else:
            parts = []
            for i in range(0, len(content), chunk_size - chunk_overlap):
                part = content[i:i + chunk_size]
                if part.strip():
                    parts.append(part.strip())

            for idx, part in enumerate(parts):
                chunk = _section_to_chunk(section, idx, len(parts), "fixed_size")
                chunk.content = part
                all_chunks.append(chunk)

    return all_chunks


def chunk_recursive(
    sections: list[Section],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] = None,
) -> list[Chunk]:
    """
    비교 베이스라인: 구분자 기반 재귀 분할.
    DB 메타데이터는 원본 위치 기준으로 매핑.
    """
    if separators is None:
        separators = ["\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ", "\n\n", "\n"]

    all_chunks: list[Chunk] = []

    for section in sections:
        content = section.content.strip()
        if not content:
            continue

        if len(content) <= chunk_size:
            chunk = _section_to_chunk(section, 0, 1, "recursive")
            all_chunks.append(chunk)
        else:
            parts = _recursive_split(content, chunk_size, chunk_overlap, separators)
            for idx, part in enumerate(parts):
                chunk = _section_to_chunk(section, idx, len(parts), "recursive")
                chunk.content = part
                all_chunks.append(chunk)

    return all_chunks


# ─── 헬퍼 ───────────────────────────────────────────────────

def _section_to_chunk(section: Section, chunk_index: int,
                      total_chunks: int, strategy: str) -> Chunk:
    """Section → Chunk 변환"""
    return Chunk(
        year=section.year,
        section_h1=section.section_h1,
        section_h2=section.section_h2,
        section_h3=section.section_h3,
        section_h4=section.section_h4,
        section_h5=section.section_h5,
        section_h6=section.section_h6,
        section_path=section.section_path,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        content_type=section.content_type,
        content=section.content,
        source_file=section.source_file,
        chunking_strategy=strategy,
    )


# ─── 디스패처 ───────────────────────────────────────────────

def chunk_sections(sections: list[Section], config: dict) -> list[Chunk]:
    """
    config에 따라 적절한 청킹 전략을 선택하여 실행.

    Args:
        sections: parser.py에서 나온 Section 리스트
        config:   app.yaml의 chunking 섹션
    """
    strategy = config.get("active_strategy", "section_based")
    strategy_config = config.get("strategies", {}).get(strategy, {})

    if strategy == "section_based":
        layer2 = strategy_config.get("layer2", {})
        sub_method = layer2.get("sub_chunking_method", "recursive")
        sub_opts = layer2.get("sub_chunking_options", {}).get(sub_method, {})

        return chunk_section_based(
            sections,
            max_chunk_size=layer2.get("max_chunk_size", 1500),
            min_chunk_size=layer2.get("min_chunk_size", 100),
            sub_method=sub_method,
            chunk_overlap=sub_opts.get("chunk_overlap", 200),
            separators=sub_opts.get("separators"),
            table_handling=layer2.get("table_handling", "keep_whole"),
            table_max_size=layer2.get("table_max_size", 3000),
        )

    elif strategy == "fixed_size":
        return chunk_fixed_size(
            sections,
            chunk_size=strategy_config.get("chunk_size", 512),
            chunk_overlap=strategy_config.get("chunk_overlap", 64),
        )

    elif strategy == "recursive":
        return chunk_recursive(
            sections,
            chunk_size=strategy_config.get("chunk_size", 1000),
            chunk_overlap=strategy_config.get("chunk_overlap", 200),
            separators=strategy_config.get("separators"),
        )

    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
