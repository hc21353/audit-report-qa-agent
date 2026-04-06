"""
csv_reader_tool.py - [TABLE_CSV] 태그의 CSV를 동적으로 읽는 Tool

기존 src/csv_reader.py의 함수를 langchain @tool로 래핑.
"""

import json
from langchain_core.tools import tool
from src.csv_reader import read_csv, extract_csv_refs, resolve_chunk_with_tables
from pathlib import Path

_base_dir = "./parsed_data"


def init_csv_reader(base_dir: str):
    global _base_dir
    _base_dir = base_dir


@tool
def csv_reader_tool(csv_path: str, output_format: str = "markdown") -> str:
    """[TABLE_CSV] 태그에 있는 CSV 파일을 읽어 테이블 데이터를 반환합니다.

    검색된 청크에 [TABLE_CSV] tables/xxx.csv 형태의 태그가 있을 때 호출하세요.

    Args:
        csv_path: CSV 파일의 상대 경로 (예: tables/감사보고서_2024_table_002_....csv)
        output_format: "markdown" (마크다운 테이블) 또는 "dict" (JSON 객체)

    Returns:
        CSV 내용 (마크다운 테이블 문자열 또는 JSON)
    """
    full_path = str(Path(_base_dir) / csv_path)
    result = read_csv(full_path, output_format)

    if result["success"]:
        if output_format == "markdown":
            return result["content"]
        return json.dumps(result["content"], ensure_ascii=False, indent=2)
    else:
        return f"[ERROR] {result['error']}"
