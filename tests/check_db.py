"""
──────────────────────────────────────────────────────────────────
SQLite + ChromaDB 검증 스크립트

  1. DB 스키마 출력 (테이블 컬럼/타입/제약조건)
  2. 데이터 검증 (NULL, 중복, 길이 이상값)
  3. 누락/이상값 탐지 로직
  4. 이상 데이터 처리 전략 (격리 / 자동 수정 / 로그 기록)

실행: 
  --db db/audit_reports.db --vector db/vectorstore/chroma
  --fix          # 자동 수정 모드
  -export-bad   # 이상 데이터를 bad_rows.csv로 내보내기
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 경로 기본값 ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "db" / "audit_reports.db"
VECTOR_DIR = ROOT / "db" / "vectorstore" / "chroma"
LOG_PATH   = ROOT / "db" / "validation_log.jsonl"

SEP   = "=" * 60
SEP2  = "-" * 40

# ── 검증 규칙 상수 ────────────────────────────────────────────────────────────
VALID_YEARS      = range(2014, 2030)        # 허용 연도 범위
VALID_CHUNK_TYPES = {
    "Paragraph", "Table_Row", "Title", "List_Item", "Caption", "Header", "Note", "Narrative"
}
MIN_CONTENT_LEN  = 5                        # 내용 최소 길이
MAX_CONTENT_LEN  = 50_000                   # 내용 최대 길이
MIN_TAG_LEN      = 1
MAX_TAG_LEN      = 100


# ══════════════════════════════════════════════════════════════════
# 1. 스키마 출력
# ══════════════════════════════════════════════════════════════════
def print_schema(con: sqlite3.Connection):
    print(f"\n{SEP}")
    print("[ DB 스키마 ]")
    print(SEP)

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    for table in tables:
        cols = con.execute(f"PRAGMA table_info({table})").fetchall()
        fks  = con.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        idxs = con.execute(f"PRAGMA index_list({table})").fetchall()

        print(f"\n  ┌─ [{table}]")
        for col in cols:
            cid, name, dtype, notnull, dflt, pk = col
            flags = []
            if pk:       flags.append("PK")
            if notnull:  flags.append("NOT NULL")
            if dflt is not None: flags.append(f"DEFAULT={dflt}")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"  │  {name:<25} {dtype:<15}{flag_str}")

        if fks:
            print(f"  │  --- 외래키 ---")
            for fk in fks:
                print(f"  │  {fk[3]} → {fk[2]}.{fk[4]}")

        if idxs:
            print(f"  │  --- 인덱스 ---")
            for idx in idxs:
                idx_cols = [r[2] for r in con.execute(f"PRAGMA index_info({idx[1]})").fetchall()]
                unique = "UNIQUE " if idx[2] else ""
                print(f"  │  {unique}({', '.join(idx_cols)})")

        # FTS5 가상 테이블 구분
        is_virtual = con.execute(
            "SELECT sql FROM sqlite_master WHERE name=? AND type='table'", (table,)
        ).fetchone()
        if is_virtual and is_virtual[0] and "fts5" in is_virtual[0].lower():
            print(f"  │  ※ FTS5 가상 테이블")

        print(f"  └{'─'*40}")

    # 뷰 목록
    views = con.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()
    if views:
        print(f"\n  뷰: {[v[0] for v in views]}")


# ══════════════════════════════════════════════════════════════════
# 2. 데이터 검증 (누락/이상값 탐지)
# ══════════════════════════════════════════════════════════════════
class ValidationResult:
    """검증 결과 수집기"""
    def __init__(self):
        self.issues: list[dict] = []   # 발견된 문제 목록
        self.fixed:  list[dict] = []   # 수정된 항목 목록

    def add(self, table: str, rowid: int | None, field: str,
            issue_type: str, detail: str, severity: str = "WARNING"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "table": table, "rowid": rowid, "field": field,
            "issue_type": issue_type, "detail": detail, "severity": severity,
        }
        self.issues.append(entry)
        sev_icon = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}.get(severity, "?")
        print(f"    {sev_icon} [{issue_type}] row={rowid} / {field}: {detail}")

    def summary(self) -> dict:
        from collections import Counter
        by_sev = Counter(i["severity"] for i in self.issues)
        by_type = Counter(i["issue_type"] for i in self.issues)
        return {"total": len(self.issues), "by_severity": dict(by_sev),
                "by_type": dict(by_type), "fixed": len(self.fixed)}


def validate_chunks(con: sqlite3.Connection, vr: ValidationResult):
    """chunks 테이블 전체 검증"""
    print(f"\n{SEP2}")
    print("  [chunks 검증]")

    rows = con.execute(
        "SELECT id, chunk_uid, fiscal_year, chunk_type, section_path, content "
        "FROM chunks"
    ).fetchall()

    seen_uids = {}  # 중복 uid 탐지

    for row in rows:
        rid, uid, year, ctype, path, content = row

        # ── NULL / 빈 값 탐지
        for field, val in [("chunk_uid", uid), ("fiscal_year", year),
                            ("chunk_type", ctype), ("content", content)]:
            if val is None or (isinstance(val, str) and val.strip() == ""):
                vr.add("chunks", rid, field, "NULL_OR_EMPTY",
                       f"필수 필드가 비어 있음", severity="ERROR")

        # ── 중복 UID
        if uid:
            if uid in seen_uids:
                vr.add("chunks", rid, "chunk_uid", "DUPLICATE_UID",
                       f"'{uid}'가 row {seen_uids[uid]}에서 중복됨", severity="ERROR")
            else:
                seen_uids[uid] = rid

        # ── 연도 범위 이상
        if year is not None and year not in VALID_YEARS:
            vr.add("chunks", rid, "fiscal_year", "INVALID_YEAR",
                   f"fiscal_year={year} (허용 범위: {VALID_YEARS.start}~{VALID_YEARS.stop-1})",
                   severity="ERROR")

        # ── chunk_type 허용 목록 외
        if ctype and ctype not in VALID_CHUNK_TYPES:
            vr.add("chunks", rid, "chunk_type", "UNKNOWN_CHUNK_TYPE",
                   f"'{ctype}' — 허용 목록: {VALID_CHUNK_TYPES}", severity="WARNING")

        # ── content 길이 이상
        if content:
            if len(content) < MIN_CONTENT_LEN:
                vr.add("chunks", rid, "content", "TOO_SHORT",
                       f"길이={len(content)} (최소 {MIN_CONTENT_LEN}자)", severity="WARNING")
            if len(content) > MAX_CONTENT_LEN:
                vr.add("chunks", rid, "content", "TOO_LONG",
                       f"길이={len(content):,} (최대 {MAX_CONTENT_LEN:,}자)", severity="WARNING")

        # ── section_path 형식 (None 허용이지만 빈 문자열은 의심)
        if path is not None and path.strip() == "":
            vr.add("chunks", rid, "section_path", "EMPTY_PATH",
                   "section_path가 빈 문자열", severity="INFO")

    total = len(rows)
    bad   = len([i for i in vr.issues if i["table"] == "chunks"])
    print(f"    → 검사 완료: {total:,}행 / 이상 {bad}건")


def validate_tags(con: sqlite3.Connection, vr: ValidationResult):
    """tags 테이블 검증"""
    print(f"\n{SEP2}")
    print("  [tags 검증]")

    rows = con.execute("SELECT id, chunk_uid, tag FROM tags").fetchall()

    # 고아 태그(chunk_uid가 chunks 테이블에 없는 경우)
    valid_uids = {r[0] for r in con.execute("SELECT chunk_uid FROM chunks").fetchall()}

    seen_pairs = set()  # (chunk_uid, tag) 중복 탐지
    for row in rows:
        rid, uid, tag = row

        if not uid or not tag:
            vr.add("tags", rid, "tag/chunk_uid", "NULL_OR_EMPTY",
                   "필수 필드 누락", severity="ERROR")
            continue

        if uid not in valid_uids:
            vr.add("tags", rid, "chunk_uid", "ORPHAN_TAG",
                   f"chunk_uid='{uid}'가 chunks 테이블에 없음", severity="ERROR")

        if len(tag) < MIN_TAG_LEN or len(tag) > MAX_TAG_LEN:
            vr.add("tags", rid, "tag", "INVALID_TAG_LENGTH",
                   f"태그 길이={len(tag)} (허용: {MIN_TAG_LEN}~{MAX_TAG_LEN})", severity="WARNING")

        pair = (uid, tag)
        if pair in seen_pairs:
            vr.add("tags", rid, "tag", "DUPLICATE_TAG",
                   f"(chunk_uid, tag)=({uid}, {tag}) 중복", severity="WARNING")
        else:
            seen_pairs.add(pair)

    total = len(rows)
    bad   = len([i for i in vr.issues if i["table"] == "tags"])
    print(f"    → 검사 완료: {total:,}행 / 이상 {bad}건")


def validate_metadata(con: sqlite3.Connection, vr: ValidationResult):
    """report_metadata 검증"""
    print(f"\n{SEP2}")
    print("  [report_metadata 검증]")

    rows = con.execute(
        "SELECT fiscal_year, doc_id, period, num_chunks FROM report_metadata"
    ).fetchall()

    for row in rows:
        year, doc_id, period, num_chunks = row

        if year not in VALID_YEARS:
            vr.add("report_metadata", None, "fiscal_year", "INVALID_YEAR",
                   f"fiscal_year={year}", severity="ERROR")

        if not doc_id or not doc_id.strip():
            vr.add("report_metadata", None, "doc_id", "NULL_OR_EMPTY",
                   "doc_id 누락", severity="ERROR")

        if num_chunks is None or num_chunks <= 0:
            vr.add("report_metadata", None, "num_chunks", "INVALID_COUNT",
                   f"num_chunks={num_chunks}", severity="WARNING")

        # metadata의 num_chunks와 실제 chunks 수 불일치 탐지
        actual = con.execute(
            "SELECT COUNT(*) FROM chunks WHERE fiscal_year=?", (year,)
        ).fetchone()[0]
        if num_chunks and actual != num_chunks:
            vr.add("report_metadata", None, "num_chunks", "COUNT_MISMATCH",
                   f"metadata={num_chunks} vs 실제 chunks={actual}",
                   severity="WARNING")

    print(f"    → 검사 완료: {len(rows)}행")


def validate_referential_integrity(con: sqlite3.Connection, vr: ValidationResult):
    """참조 무결성 검증 (FTS 인덱스 ↔ chunks)"""
    print(f"\n{SEP2}")
    print("  [참조 무결성 검증]")

    try:
        # FTS rowid가 chunks.id와 1:1 대응하는지 확인
        fts_count = con.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if fts_count != chunk_count:
            vr.add("fts_chunks", None, "rowid", "FTS_COUNT_MISMATCH",
                   f"FTS={fts_count:,} vs chunks={chunk_count:,} — FTS 재빌드 필요",
                   severity="ERROR")
        else:
            print(f"    ✅ FTS ↔ chunks 수량 일치 ({fts_count:,}개)")
    except Exception as e:
        vr.add("fts_chunks", None, "-", "FTS_ACCESS_ERROR",
               str(e), severity="WARNING")


# ══════════════════════════════════════════════════════════════════
# 3. 처리 전략 — 이상 데이터 수정/격리
# ══════════════════════════════════════════════════════════════════
def apply_fixes(con: sqlite3.Connection, vr: ValidationResult):
    """
    처리 전략:
      ERROR   → 수정 가능하면 자동 수정, 불가능하면 quarantine 테이블로 격리
      WARNING → 로그 기록 후 통과
      INFO    → 로그만
    """
    print(f"\n{SEP}")
    print("[ 이상 데이터 처리 전략 적용 ]")
    print(SEP)

    # quarantine 테이블 생성 (없으면)
    con.execute("""
        CREATE TABLE IF NOT EXISTS quarantine (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT,
            source_rowid INTEGER,
            issue_type   TEXT,
            detail       TEXT,
            captured_at  TEXT
        )
    """)

    errors = [i for i in vr.issues if i["severity"] == "ERROR"]
    if not errors:
        print("  ✅ 처리할 ERROR 없음")
        return

    for issue in errors:
        table    = issue["table"]
        rowid    = issue["rowid"]
        itype    = issue["issue_type"]
        detail   = issue["detail"]
        ts       = issue["timestamp"]

        # ── 전략 1: 빈 content → 플레이스홀더로 자동 수정
        if itype == "NULL_OR_EMPTY" and issue["field"] == "content" and rowid:
            placeholder = "[내용 없음 — 자동 수정됨]"
            con.execute("UPDATE chunks SET content=? WHERE id=?", (placeholder, rowid))
            vr.fixed.append(issue)
            print(f"  🔧 [자동 수정] chunks.id={rowid} content → 플레이스홀더")

        # ── 전략 2: 중복 UID → 나중 행을 quarantine으로 격리
        elif itype == "DUPLICATE_UID" and rowid:
            con.execute("""
                INSERT INTO quarantine(source_table, source_rowid, issue_type, detail, captured_at)
                VALUES (?,?,?,?,?)
            """, (table, rowid, itype, detail, ts))
            con.execute("DELETE FROM chunks WHERE id=?", (rowid,))
            vr.fixed.append(issue)
            print(f"  🔒 [격리] chunks.id={rowid} → quarantine (중복 UID)")

        # ── 전략 3: 고아 태그 → quarantine 후 삭제
        elif itype == "ORPHAN_TAG" and rowid:
            con.execute("""
                INSERT INTO quarantine(source_table, source_rowid, issue_type, detail, captured_at)
                VALUES (?,?,?,?,?)
            """, (table, rowid, itype, detail, ts))
            con.execute("DELETE FROM tags WHERE id=?", (rowid,))
            vr.fixed.append(issue)
            print(f"  🔒 [격리] tags.id={rowid} → quarantine (고아 태그)")

        # ── 전략 4: FTS 불일치 → 수동 재빌드 안내
        elif itype == "FTS_COUNT_MISMATCH":
            print(f"  ⚠️  FTS 재빌드 필요 — 수동 실행 권장:")
            print(f"      INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild');")

        # ── 나머지 ERROR → quarantine 기록만
        else:
            if rowid:
                con.execute("""
                    INSERT INTO quarantine(source_table, source_rowid, issue_type, detail, captured_at)
                    VALUES (?,?,?,?,?)
                """, (table, rowid, itype, detail, ts))
                print(f"  📋 [기록] {table}.id={rowid} → quarantine ({itype})")

    con.commit()
    print(f"\n  처리 완료: {len(vr.fixed)}건 수정/격리")


# ══════════════════════════════════════════════════════════════════
# 4. 이상 데이터 CSV 내보내기
# ══════════════════════════════════════════════════════════════════
def export_bad_rows(vr: ValidationResult, out_path: Path):
    if not vr.issues:
        print("  내보낼 이상 데이터 없음")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=vr.issues[0].keys())
        w.writeheader()
        w.writerows(vr.issues)
    print(f"  CSV 저장: {out_path} ({len(vr.issues)}건)")


# ══════════════════════════════════════════════════════════════════
# 5. 검증 로그 저장 (JSONL)
# ══════════════════════════════════════════════════════════════════
def save_log(vr: ValidationResult, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        run_entry = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "summary": vr.summary(),
            "issues": vr.issues,
        }
        f.write(json.dumps(run_entry, ensure_ascii=False) + "\n")
    print(f"  로그 저장: {log_path}")


# ══════════════════════════════════════════════════════════════════
# 6. 기존 SQLite 요약 (원본 유지)
# ══════════════════════════════════════════════════════════════════
def check_sqlite_summary(con: sqlite3.Connection):
    print(f"\n{SEP}")
    print("[ SQLite 기본 요약 ]")
    print(SEP)

    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"  chunks 총 {total:,}개")

    print("  ▶ 연도별 청크 수:")
    for row in con.execute(
        "SELECT fiscal_year, COUNT(*) FROM chunks GROUP BY fiscal_year ORDER BY fiscal_year"
    ).fetchall():
        print(f"    {row[0]}년: {row[1]:,}개")

    print("  ▶ chunk_type 분포:")
    for row in con.execute(
        "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY 2 DESC"
    ).fetchall():
        print(f"    {row[0]}: {row[1]:,}개")

    tag_total = con.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    print(f"  tags 총 {tag_total:,}개")


# ══════════════════════════════════════════════════════════════════
# 7. ChromaDB 검증 (기존 + 메타데이터 일관성 추가)
# ══════════════════════════════════════════════════════════════════
def check_chroma(vector_dir: Path, sqlite_con: sqlite3.Connection, vr: ValidationResult):
    print(f"\n{SEP}")
    print("[ ChromaDB 검증 ]")
    print(SEP)

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("  chromadb 미설치 — pip install chromadb")
        return

    summary_path = vector_dir / "build_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        print(f"  빌드 시각 : {s.get('built_at')}")
        print(f"  임베딩 모델: {s.get('model')}")
        print(f"  임베딩 차원: {s.get('embedding_dim')}")
        print(f"  총 벡터 수 : {s.get('total_vectors', s.get('total', 0)):,}")

    client = chromadb.PersistentClient(
        path=str(vector_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_collection("samsung_audit")
    chroma_count = col.count()
    sqlite_count = sqlite_con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    print(f"\n  ChromaDB 벡터 수: {chroma_count:,}")
    print(f"  SQLite  청크 수 : {sqlite_count:,}")

    if chroma_count != sqlite_count:
        diff = abs(chroma_count - sqlite_count)
        vr.add("chroma", None, "vector_count", "CHROMA_SQLITE_MISMATCH",
               f"차이 {diff:,}개 (chroma={chroma_count}, sqlite={sqlite_count})",
               severity="ERROR")
    else:
        print("  ✅ SQLite ↔ ChromaDB 수량 일치")

    # 샘플 메타데이터 연도 검증
    print("\n  ▶ 샘플 메타데이터 연도 검증 (50개):")
    sample = col.get(limit=50, include=["metadatas"])
    bad_meta = 0
    for uid, meta in zip(sample["ids"], sample["metadatas"]):
        year = meta.get("fiscal_year")
        if year is None or int(year) not in VALID_YEARS:
            bad_meta += 1
            vr.add("chroma", None, "fiscal_year", "INVALID_YEAR_IN_VECTOR",
                   f"uid={uid}, fiscal_year={year}", severity="WARNING")
    if bad_meta == 0:
        print("    ✅ 샘플 50개 연도 정상")
    else:
        print(f"    ⚠️  이상 {bad_meta}건")


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB 검증 v2")
    parser.add_argument("--db",         type=Path, default=DB_PATH)
    parser.add_argument("--vector",     type=Path, default=VECTOR_DIR)
    parser.add_argument("--fix",        action="store_true", help="이상 데이터 자동 수정/격리")
    parser.add_argument("--export-bad", action="store_true", help="이상 데이터 CSV 내보내기")
    parser.add_argument("--log",        type=Path, default=LOG_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"❌ SQLite DB 없음: {args.db}")
        sys.exit(1)

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    vr = ValidationResult()

    # 1. 스키마 출력
    print_schema(con)

    # 2. 기본 요약
    check_sqlite_summary(con)

    # 3. 데이터 검증
    print(f"\n{SEP}")
    print("[ 데이터 검증 — 누락/이상값 탐지 ]")
    print(SEP)
    validate_chunks(con, vr)
    validate_tags(con, vr)
    validate_metadata(con, vr)
    validate_referential_integrity(con, vr)

    # 4. ChromaDB 검증
    if args.vector.exists():
        check_chroma(args.vector, con, vr)
    else:
        print(f"\n⚠️  ChromaDB 디렉토리 없음: {args.vector}")

    # 5. 결과 요약
    print(f"\n{SEP}")
    print("[ 검증 결과 요약 ]")
    print(SEP)
    summ = vr.summary()
    print(f"  총 이상 건수 : {summ['total']}")
    print(f"  심각도별     : {summ['by_severity']}")
    print(f"  유형별       : {summ['by_type']}")

    # 6. 처리 전략 적용 (--fix 옵션)
    if args.fix:
        apply_fixes(con, vr)
    else:
        errors = [i for i in vr.issues if i["severity"] == "ERROR"]
        if errors:
            print(f"\n  ⚠️  ERROR {len(errors)}건 발견 — '--fix' 옵션으로 자동 수정 가능")

    # 7. CSV 내보내기 (--export-bad 옵션)
    if args.export_bad:
        out = args.db.parent / "bad_rows.csv"
        print(f"\n{SEP}")
        print("[ 이상 데이터 CSV 내보내기 ]")
        print(SEP)
        export_bad_rows(vr, out)

    # 8. 로그 저장
    save_log(vr, args.log)

    con.close()

    print(f"\n{SEP}")
    print("검증 완료")
    print(SEP)

    # ERROR 있으면 비정상 종료 코드
    if any(i["severity"] == "ERROR" for i in vr.issues):
        sys.exit(2)