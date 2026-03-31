"""
loader.py - 메인 적재 파이프라인

파이프라인:
  1. 마크다운 파싱 (parser.py → Section 리스트)
  2. 메타데이터 → metadata 테이블
  3. 섹션 청킹 (chunker.py → Chunk 리스트)
  4. 청크 → documents 테이블

[TABLE_CSV] 태그는 그대로 DB에 저장.
에이전트가 조회 시 csv_reader tool로 동적 로드.

사용법:
  python -m src.loader                          # 전체 적재
  python -m src.loader --file 감사보고서_2024.md  # 단일 파일
  python -m src.loader --year 2024              # 특정 연도 재적재
"""

import argparse
from pathlib import Path

from src.config import load_config
from src.db import AuditDB
from src.parser import parse_markdown, DocumentMeta
from src.chunker import chunk_sections


def load_single_file(
    filepath: str,
    db: AuditDB,
    chunking_config: dict,
    clear_existing: bool = True,
) -> dict:
    """단일 마크다운 파일을 파싱하여 DB에 적재."""
    stats = {"file": filepath, "metadata": 0, "documents": 0}

    # 1. 파싱
    meta, sections = parse_markdown(filepath)
    if not meta.year:
        print(f"[Loader] WARNING: No year in frontmatter for {filepath}")
        return stats

    year = meta.year
    print(f"\n[Loader] Processing year={year}, file={filepath}")

    # 2. 기존 데이터 삭제 (재적재)
    if clear_existing:
        db.clear_year(year)

    # 3. 메타데이터 저장
    _load_metadata(db, meta, filepath)
    stats["metadata"] = 1

    # 4. 청킹 및 문서 저장
    chunks = chunk_sections(sections, chunking_config)
    if chunks:
        rows = [c.to_db_row() for c in chunks]
        db.insert_documents_batch(rows)
        stats["documents"] = len(chunks)

    return stats


def _load_metadata(db: AuditDB, meta: DocumentMeta, filepath: str):
    """메타데이터 테이블에 저장"""
    period_start = ""
    period_end = ""
    if meta.period and "~" in meta.period:
        parts = meta.period.split("~")
        period_start = parts[0].strip()
        period_end = parts[1].strip()

    db.insert_metadata(
        year=meta.year,
        period_start=period_start,
        period_end=period_end,
        auditor="",  # TODO: 본문에서 추출
        opinion="",  # TODO: 본문에서 추출
        source_file=Path(filepath).name,
    )


def load_all(
    input_dir: str,
    db: AuditDB,
    chunking_config: dict,
) -> list[dict]:
    """디렉토리 내 모든 마크다운 파일 적재"""
    md_dir = Path(input_dir)
    if not md_dir.exists():
        print(f"[Loader] Directory not found: {input_dir}")
        return []

    all_stats = []
    md_files = sorted(md_dir.glob("*.md"))
    print(f"[Loader] Found {len(md_files)} markdown files in {input_dir}")

    for md_file in md_files:
        stats = load_single_file(str(md_file), db, chunking_config)
        all_stats.append(stats)

    return all_stats


def print_stats(all_stats: list[dict]):
    """적재 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("적재 결과 요약")
    print("=" * 60)
    total_docs = 0

    for stats in all_stats:
        print(f"  {stats['file']}")
        print(f"    metadata:  {stats['metadata']}")
        print(f"    documents: {stats['documents']} chunks")
        total_docs += stats["documents"]

    print(f"\n  Total: {total_docs} chunks")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="감사보고서 DB 적재")
    parser.add_argument("--file", type=str, help="단일 마크다운 파일 경로")
    parser.add_argument("--dir", type=str, help="마크다운 디렉토리")
    parser.add_argument("--year", type=int, help="특정 연도만 재적재")
    parser.add_argument("--db", type=str, help="DB 경로")
    args = parser.parse_args()

    cfg = load_config()
    db_path = args.db or cfg.db_path
    input_dir = args.dir or cfg.parsing_dir
    chunking_config = cfg.app.get("chunking", {})

    db = AuditDB(db_path)
    db.init_tables()

    if args.file:
        stats = load_single_file(args.file, db, chunking_config)
        print_stats([stats])
    elif args.year:
        md_dir = Path(input_dir)
        matches = list(md_dir.glob(f"*{args.year}*.md"))
        if not matches:
            print(f"[Loader] No file found for year {args.year} in {input_dir}")
        else:
            all_stats = [load_single_file(str(f), db, chunking_config) for f in matches]
            print_stats(all_stats)
    else:
        all_stats = load_all(input_dir, db, chunking_config)
        print_stats(all_stats)

    print("\n[DB 현황]")
    for k, v in db.stats().items():
        print(f"  {k}: {v}")

    db.close()


if __name__ == "__main__":
    main()
