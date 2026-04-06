"""
일반 디비 구축 스크립트 (SQLite)
──────────────────
semantic_chunks_tagged.json (JSONL) → SQLite audit_reports.db

테이블:
  - chunks      : 모든 청크 (텍스트 + 메타데이터)
  - tags        : 청크-태그 매핑 (다대다 정규화)
  - metadata    : 연도별 감사보고서 메타 (auditor, opinion 등)
  - fts_chunks  : FTS5 전문 검색 인덱스
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # src/ → 프로젝트 루트
CHUNKS_PATH = ROOT / "parsed_data" / "chunks" / "semantic_chunks_tagged.jsonl"
DB_PATH = ROOT / "db" / "audit_reports.db"

# ── 스키마 ────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
-- 청크 메인 테이블
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid       TEXT UNIQUE,              -- "{doc_id}_{row_idx}" 형태 고유 ID
    doc_id          TEXT NOT NULL,            -- e.g. SAM_2016_ANNUAL
    fiscal_year     INTEGER NOT NULL,
    period          TEXT,                     -- e.g. 제 47 기
    section_path    TEXT,
    chunk_type      TEXT,                     -- Narrative / Note / Table_Row
    is_consolidated INTEGER DEFAULT 0,        -- BOOL (0/1)
    content         TEXT NOT NULL,            -- 검색/LLM에 넣을 텍스트
    content_length  INTEGER,
    -- Table_Row 전용 구조 필드 (NULL = 해당 없음)
    table_ref       TEXT,
    table_unit      TEXT,
    row_json        TEXT,                     -- JSON string of row_data
    -- FTS 용 note_refs (쉼표 구분)
    note_refs_csv   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- 태그 정규화 테이블 (1청크 : N태그)
CREATE TABLE IF NOT EXISTS tags (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_uid   TEXT NOT NULL,
    tag         TEXT NOT NULL,
    tag_category TEXT,                        -- major / segment / theme
    FOREIGN KEY (chunk_uid) REFERENCES chunks(chunk_uid)
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_chunk ON tags(chunk_uid);

-- 연도별 감사보고서 메타
CREATE TABLE IF NOT EXISTS report_metadata (
    fiscal_year     INTEGER PRIMARY KEY,
    doc_id          TEXT,
    period          TEXT,
    auditor         TEXT,
    opinion         TEXT,                     -- 적정/한정 등
    period_start    TEXT,
    period_end      TEXT,
    source_file     TEXT,
    num_chunks      INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- FTS5 전문 검색 인덱스
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_uid,
    content,
    section_path,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
);
"""

TRIGGER_SQL = """
-- FTS 자동 동기화 트리거
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, chunk_uid, content, section_path)
    VALUES (new.id, new.chunk_uid, new.content, new.section_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, chunk_uid, content, section_path)
    VALUES ('delete', old.id, old.chunk_uid, old.content, old.section_path);
END;
"""


# ── 청크 로더 ─────────────────────────────────────────────────────────────────
def load_chunks(path: Path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  청크 로드 완료: {len(chunks):,}개")
    return chunks


# ── 단일 청크 파싱 ────────────────────────────────────────────────────────────
def parse_chunk(row_idx: int, chunk: dict) -> tuple[dict, list[dict]]:
    """
    Returns:
        chunk_row  : dict for INSERT INTO chunks
        tag_rows   : list of dicts for INSERT INTO tags
    """
    meta = chunk["metadata"]
    text = chunk["text"]
    struct = text.get("struct", {})

    chunk_uid = f"{meta['doc_id']}_{row_idx:06d}"
    content = text.get("original", "")

    # Table_Row 전용 필드
    table_ref = struct.get("table_ref")
    table_unit = struct.get("unit")
    row_data = struct.get("row_data")
    row_json = json.dumps(row_data, ensure_ascii=False) if row_data else None

    note_refs = meta.get("note_refs", [])
    note_refs_csv = ",".join(str(n) for n in note_refs) if note_refs else None

    chunk_row = {
        "chunk_uid": chunk_uid,
        "doc_id": meta.get("doc_id"),
        "fiscal_year": meta.get("fiscal_year"),
        "period": meta.get("period"),
        "section_path": meta.get("section_path"),
        "chunk_type": meta.get("chunk_type"),
        "is_consolidated": int(meta.get("is_consolidated", False)),
        "content": content,
        "content_length": len(content),
        "table_ref": table_ref,
        "table_unit": table_unit,
        "row_json": row_json,
        "note_refs_csv": note_refs_csv,
    }

    # 태그 rows
    tag_rows = []
    for cat, key in [("major", "tags_major"), ("segment", "tags_segment"), ("theme", "tags_theme")]:
        for tag in meta.get(key, []):
            tag_rows.append({"chunk_uid": chunk_uid, "tag": tag, "tag_category": cat})

    return chunk_row, tag_rows


# ── 연도별 메타 집계 ──────────────────────────────────────────────────────────
def build_report_metadata(chunks: list[dict]) -> list[dict]:
    from collections import defaultdict
    year_info: dict[int, dict] = defaultdict(lambda: {"count": 0, "doc_id": None, "period": None})
    for c in chunks:
        m = c["metadata"]
        y = m["fiscal_year"]
        year_info[y]["count"] += 1
        year_info[y]["doc_id"] = m.get("doc_id")
        year_info[y]["period"] = m.get("period")

    rows = []
    for year, info in sorted(year_info.items()):
        rows.append({
            "fiscal_year": year,
            "doc_id": info["doc_id"],
            "period": info["period"],
            "auditor": None,    # 파싱 모듈에서 채울 것
            "opinion": None,
            "period_start": None,
            "period_end": None,
            "source_file": f"감사보고서_{year}.md",
            "num_chunks": info["count"],
        })
    return rows


# ── 메인 ─────────────────────────────────────────────────────────────────────
def build_db(chunks_path: Path, db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # DB가 이미 존재하면 백업
    if db_path.exists():
        backup = db_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
        db_path.rename(backup)
        print(f"  기존 DB 백업: {backup.name}")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")

    print("스키마 생성...")
    con.executescript(SCHEMA_SQL)
    con.executescript(TRIGGER_SQL)
    con.commit()

    print("청크 로드...")
    chunks = load_chunks(chunks_path)

    print("chunks 테이블 삽입...")
    chunk_rows, all_tag_rows = [], []
    for idx, chunk in enumerate(chunks):
        cr, tr = parse_chunk(idx, chunk)
        chunk_rows.append(cr)
        all_tag_rows.extend(tr)

    con.executemany(
        """INSERT OR IGNORE INTO chunks
           (chunk_uid, doc_id, fiscal_year, period, section_path, chunk_type,
            is_consolidated, content, content_length, table_ref, table_unit,
            row_json, note_refs_csv)
           VALUES (:chunk_uid, :doc_id, :fiscal_year, :period, :section_path,
                   :chunk_type, :is_consolidated, :content, :content_length,
                   :table_ref, :table_unit, :row_json, :note_refs_csv)""",
        chunk_rows
    )
    print(f"  → {len(chunk_rows):,}개 삽입")

    print("tags 테이블 삽입...")
    con.executemany(
        "INSERT INTO tags (chunk_uid, tag, tag_category) VALUES (:chunk_uid, :tag, :tag_category)",
        all_tag_rows
    )
    print(f"  → {len(all_tag_rows):,}개 태그 삽입")

    print("report_metadata 삽입...")
    meta_rows = build_report_metadata(chunks)
    con.executemany(
        """INSERT OR REPLACE INTO report_metadata
           (fiscal_year, doc_id, period, auditor, opinion,
            period_start, period_end, source_file, num_chunks)
           VALUES (:fiscal_year, :doc_id, :period, :auditor, :opinion,
                   :period_start, :period_end, :source_file, :num_chunks)""",
        meta_rows
    )
    for r in meta_rows:
        print(f"    {r['fiscal_year']}년: {r['num_chunks']:,}개 청크")

    con.commit()
    con.close()

    size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"\n✅ SQLite DB 완성: {db_path}  ({size_mb:.1f} MB)")
    print(f"   총 청크: {len(chunk_rows):,}  태그: {len(all_tag_rows):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    print(f"=== SQLite DB 구축 ===")
    print(f"  청크 파일: {args.chunks}")
    print(f"  DB 경로  : {args.db}")
    build_db(args.chunks, args.db)
