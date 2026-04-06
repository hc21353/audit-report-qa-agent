"""
csv_reader.py - [TABLE_CSV] 태그의 CSV 파일을 동적으로 읽는 Tool

에이전트가 검색 결과에서 [TABLE_CSV] 태그를 발견하면 이 tool을 호출하여
실제 표 데이터를 가져온다.

사용 흐름:
  1. 에이전트가 벡터/구조 검색으로 청크 획득
  2. 청크 content에 [TABLE_CSV] path/to/file.csv 발견
  3. csv_reader tool 호출 → CSV 내용을 마크다운 테이블 또는 dict로 반환

이 모듈은 LangGraph ToolNode로 등록되거나, 직접 함수 호출도 가능.
"""

import csv
import re
from pathlib import Path
from typing import Optional


TABLE_CSV_RE = re.compile(r"\[TABLE_CSV\]\s+(.+\.csv)\s*")


def read_csv(csv_path: str, output_format: str = "markdown") -> dict:
    """
    CSV 파일을 읽어 지정된 형식으로 반환.

    Args:
        csv_path:      CSV 파일 경로
        output_format: "markdown" | "dict" | "raw"

    Returns:
        {
            "success": bool,
            "path": str,
            "format": str,
            "content": str | list[dict] | list[list[str]],
            "rows": int,
            "columns": list[str],
            "error": str (실패 시)
        }
    """
    path = Path(csv_path)
    if not path.exists():
        return {
            "success": False,
            "path": csv_path,
            "format": output_format,
            "content": "",
            "rows": 0,
            "columns": [],
            "error": f"File not found: {csv_path}",
        }

    rows = _read_csv_raw(path)
    if not rows:
        return {
            "success": False,
            "path": csv_path,
            "format": output_format,
            "content": "",
            "rows": 0,
            "columns": [],
            "error": "Empty CSV file",
        }

    headers = rows[0]
    data_rows = rows[1:]

    if output_format == "markdown":
        content = _to_markdown(headers, data_rows)
    elif output_format == "dict":
        content = _to_dict_list(headers, data_rows)
    else:  # raw
        content = rows

    return {
        "success": True,
        "path": csv_path,
        "format": output_format,
        "content": content,
        "rows": len(data_rows),
        "columns": headers,
    }


def extract_csv_refs(text: str) -> list[str]:
    """텍스트에서 [TABLE_CSV] 경로를 모두 추출"""
    return TABLE_CSV_RE.findall(text)


def read_all_csv_in_chunk(
    chunk_content: str,
    base_dir: str,
    output_format: str = "markdown",
) -> list[dict]:
    """
    청크 content에 포함된 모든 [TABLE_CSV] 태그의 CSV를 읽어서 반환.
    에이전트가 청크를 받은 후 표 데이터가 필요할 때 호출.

    Args:
        chunk_content: 청크의 content 텍스트
        base_dir:      CSV 파일의 기준 디렉토리
        output_format: "markdown" | "dict" | "raw"

    Returns:
        각 CSV에 대한 read_csv 결과 리스트
    """
    csv_refs = extract_csv_refs(chunk_content)
    if not csv_refs:
        return []

    results = []
    base = Path(base_dir)
    for ref in csv_refs:
        full_path = base / ref
        result = read_csv(str(full_path), output_format)
        results.append(result)

    return results


def resolve_chunk_with_tables(
    chunk_content: str,
    base_dir: str,
) -> str:
    """
    청크 content의 [TABLE_CSV] 태그를 실제 마크다운 테이블로 치환하여 반환.
    에이전트가 LLM에 전달할 최종 컨텍스트를 만들 때 사용.

    Args:
        chunk_content: [TABLE_CSV] 태그가 포함된 content
        base_dir:      CSV 파일의 기준 디렉토리

    Returns:
        [TABLE_CSV] 태그가 실제 테이블로 치환된 텍스트
    """
    base = Path(base_dir)

    def _replace(match):
        csv_rel_path = match.group(1).strip()
        full_path = base / csv_rel_path
        result = read_csv(str(full_path), "markdown")
        if result["success"]:
            return result["content"]
        return f"[TABLE_CSV_ERROR] {result['error']}"

    return TABLE_CSV_RE.sub(_replace, chunk_content)


# ─── 내부 유틸리티 ───────────────────────────────────────────

def _read_csv_raw(csv_path: Path) -> list[list[str]]:
    """CSV → 2D 리스트. UTF-8 실패 시 CP949 시도."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.reader(f))
    except UnicodeDecodeError:
        with open(csv_path, "r", encoding="cp949") as f:
            return list(csv.reader(f))


def _to_markdown(headers: list[str], data_rows: list[list[str]]) -> str:
    """헤더 + 데이터 → 마크다운 테이블 문자열"""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in data_rows:
        while len(row) < len(headers):
            row.append("")
        lines.append("| " + " | ".join(row[:len(headers)]) + " |")
    return "\n".join(lines)


def _to_dict_list(headers: list[str], data_rows: list[list[str]]) -> list[dict]:
    """헤더 + 데이터 → dict 리스트"""
    result = []
    for row in data_rows:
        d = {}
        for i, h in enumerate(headers):
            d[h] = row[i] if i < len(row) else ""
        result.append(d)
    return result


# ─── LangGraph Tool 정의용 ──────────────────────────────────

# LangGraph에서 @tool 데코레이터로 등록할 때 사용할 함수 시그니처

def csv_reader_tool(csv_path: str, base_dir: str = "./data/parsed_md",
                    output_format: str = "markdown") -> str:
    """
    LangGraph Tool: CSV 파일을 읽어 마크다운 테이블로 반환.

    Args:
        csv_path:      [TABLE_CSV] 태그에 있는 상대 경로
        base_dir:      CSV 파일의 기준 디렉토리
        output_format: "markdown" | "dict"

    Returns:
        마크다운 테이블 문자열 (또는 에러 메시지)
    """
    full_path = str(Path(base_dir) / csv_path)
    result = read_csv(full_path, output_format)

    if result["success"]:
        if output_format == "markdown":
            return result["content"]
        else:
            import json
            return json.dumps(result["content"], ensure_ascii=False, indent=2)
    else:
        return f"[ERROR] {result['error']}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_reader.py <csv_file>")
        sys.exit(1)

    result = read_csv(sys.argv[1])
    if result["success"]:
        print(f"Columns: {result['columns']}")
        print(f"Rows: {result['rows']}")
        print()
        print(result["content"][:500])
    else:
        print(f"Error: {result['error']}")
