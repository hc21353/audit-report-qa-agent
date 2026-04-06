"""
db.py - SQLite 데이터베이스 관리 (새 스키마)

테이블:
  - chunks:          파싱된 감사보고서 청크 (chunk_uid, fiscal_year, section_path, chunk_type ...)
  - tags:            청크-태그 매핑 (chunk_uid, tag, tag_category)
  - report_metadata: 연도별 메타 정보 (fiscal_year, auditor, opinion ...)
  - fts_chunks:      FTS5 전문 검색 인덱스
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional


class AuditDB:
    def __init__(self, db_path: str = "./data/audit_reports.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    # ─── BM25 키워드 검색 (FTS5) ────────────────────────────

    @staticmethod
    def _strip_korean_particles(word: str) -> str:
        """한국어 조사/어미 제거 (FTS5 토큰 매칭률 개선)"""
        suffixes = [
            "에서의", "으로의", "에서는", "으로는", "에서도",
            "에서", "으로", "에는", "에도", "와의", "과의",
            "이란", "란", "라는",
            "의", "는", "은", "이", "가", "을", "를",
            "에", "와", "과", "로", "도", "만", "부터",
            "까지", "처럼", "같은", "에게", "한테",
        ]
        for suffix in suffixes:
            if len(word) > len(suffix) + 1 and word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    def bm25_search(
        self,
        query: str,
        top_k: int = 20,
        year: Optional[int] = None,
        section_path_contains: Optional[str] = None,
        chunk_type: Optional[str] = None,
        is_consolidated: Optional[int] = None,
    ) -> list[dict]:
        """
        FTS5 BM25 기반 키워드 검색.

        Args:
            query:                검색 키워드
            top_k:                최대 결과 수
            year:                 연도 필터 (fiscal_year)
            section_path_contains: section_path 부분 문자열 필터
            chunk_type:           Narrative | Note | Table_Row
            is_consolidated:      0=별도, 1=연결

        Returns:
            [{"id", "chunk_uid", "fiscal_year", "section_path", "chunk_type",
              "content", "bm25_score", ...}]
        """
        words = query.strip().split()
        if not words:
            return []

        all_terms: set[str] = set()
        for w in words:
            all_terms.add(w)
            stripped = self._strip_korean_particles(w)
            if stripped != w and len(stripped) > 1:
                all_terms.add(stripped)

        fts_query = " OR ".join(f'"{t}"' for t in all_terms)

        sql = """
            SELECT c.id, c.chunk_uid, c.doc_id, c.fiscal_year, c.period,
                   c.section_path, c.chunk_type, c.is_consolidated,
                   c.content, c.content_length, c.table_ref, c.table_unit,
                   rank AS bm25_score
            FROM chunks c
            JOIN fts_chunks fts ON c.id = fts.rowid
            WHERE fts_chunks MATCH ?
        """
        params: list = [fts_query]

        if year:
            sql += " AND c.fiscal_year = ?"
            params.append(year)
        if section_path_contains:
            sql += " AND c.section_path LIKE ?"
            params.append(f"%{section_path_contains}%")
        if chunk_type and chunk_type != "all":
            sql += " AND c.chunk_type = ?"
            params.append(chunk_type)
        if is_consolidated is not None:
            sql += " AND c.is_consolidated = ?"
            params.append(is_consolidated)

        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[DB] BM25 search error: {e}")
            return []

    # ─── 구조 기반 검색 ──────────────────────────────────────

    def structured_search(
        self,
        years: Optional[list[int]] = None,
        section_path_contains: Optional[str] = None,
        chunk_type: Optional[str] = None,
        is_consolidated: Optional[int] = None,
        keyword: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        section_path / chunk_type / tags / keyword를 조합한 구조 기반 검색.

        예시:
          - years=[2022,2023,2024], section_path_contains="주석"
            → 3개 연도 주석 섹션 전체
          - section_path_contains="30.", chunk_type="Table_Row"
            → 30번 주석의 테이블 행들
          - tags=["영업이익", "손익"]
            → 해당 태그가 붙은 청크
        """
        select_cols = """
            SELECT c.id, c.chunk_uid, c.doc_id, c.fiscal_year, c.period,
                   c.section_path, c.chunk_type, c.is_consolidated,
                   c.content, c.content_length,
                   c.table_ref, c.table_unit, c.note_refs_csv
        """

        if tags:
            tag_placeholders = ",".join("?" * len(tags))
            sql = f"""
                {select_cols}
                FROM chunks c
                JOIN tags t ON c.chunk_uid = t.chunk_uid
                WHERE t.tag IN ({tag_placeholders})
            """
            params: list = list(tags)
        else:
            sql = f"""
                {select_cols}
                FROM chunks c WHERE 1=1
            """
            params = []

        if years:
            placeholders = ",".join("?" * len(years))
            sql += f" AND c.fiscal_year IN ({placeholders})"
            params.extend(years)
        if section_path_contains:
            sql += " AND c.section_path LIKE ?"
            params.append(f"%{section_path_contains}%")
        if chunk_type and chunk_type != "all":
            sql += " AND c.chunk_type = ?"
            params.append(chunk_type)
        if is_consolidated is not None:
            sql += " AND c.is_consolidated = ?"
            params.append(is_consolidated)
        if keyword:
            kw_parts = [k.strip() for k in re.split(r"[,;]", keyword) if k.strip()]
            if kw_parts:
                content_clauses = " OR ".join(["c.content LIKE ?"] * len(kw_parts))
                path_clauses = " OR ".join(["c.section_path LIKE ?"] * len(kw_parts))
                sql += f" AND (({content_clauses}) OR ({path_clauses}))"
                params.extend([f"%{k}%" for k in kw_parts])
                params.extend([f"%{k}%" for k in kw_parts])

        if tags:
            sql += " GROUP BY c.chunk_uid"  # 태그 JOIN 중복 제거

        # 연도 필터가 있으면 연도 내 id 순(문서 순서), 없으면 최신 연도 우선 반환
        if years:
            sql += " ORDER BY c.fiscal_year, c.id LIMIT ?"
        else:
            sql += " ORDER BY c.fiscal_year DESC, c.id LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ─── 섹션 목록 조회 ──────────────────────────────────────

    def list_sections(self, year: Optional[int] = None, level: int = 1) -> list[dict]:
        """
        section_path에서 특정 레벨의 섹션 목록을 반환.
        (Python에서 ' > ' 구분자로 파싱)

        level=1: 최상위 (예: "(첨부)재무제표", "독립된 감사인의 감사보고서")
        level=2: 두 번째 세그먼트 (예: "주석")
        level=3: 세 번째 세그먼트 — 주석 번호 (예: "9. 종속기업, 관계기업및공동기업 투자:")

        Returns:
            [{"section": str, "year": int, "chunk_count": int}]
        """
        sql = "SELECT section_path, fiscal_year FROM chunks WHERE section_path IS NOT NULL"
        params: list = []
        if year:
            sql += " AND fiscal_year = ?"
            params.append(year)

        rows = self.conn.execute(sql, params).fetchall()

        counter: dict[tuple, int] = {}
        for row in rows:
            path = row["section_path"]
            yr = row["fiscal_year"]
            parts = [p.strip() for p in path.split(" > ")]
            if len(parts) >= level:
                section = parts[level - 1]
                key = (yr, section)
                counter[key] = counter.get(key, 0) + 1

        return [
            {"section": sec, "year": yr, "chunk_count": cnt}
            for (yr, sec), cnt in sorted(counter.items())
        ]

    # ─── 청크 ID 기반 전체 내용 조회 ─────────────────────────

    def get_chunks_by_ids(self, ids) -> list[dict]:
        """
        chunk_uid (str) 또는 integer id 목록의 전체 내용 반환.

        Args:
            ids: chunk_uid 문자열 리스트 또는 integer id 리스트

        Returns:
            청크 전체 내용 딕셔너리 리스트
        """
        if not ids:
            return []

        if ids and isinstance(ids[0], str):
            col = "chunk_uid"
        else:
            col = "id"

        placeholders = ",".join("?" * len(ids))
        sql = f"""
            SELECT id, chunk_uid, doc_id, fiscal_year, period, section_path,
                   chunk_type, is_consolidated, content, content_length,
                   table_ref, table_unit, row_json, note_refs_csv
            FROM chunks WHERE {col} IN ({placeholders})
            ORDER BY fiscal_year, id
        """
        rows = self.conn.execute(sql, ids).fetchall()
        return [dict(row) for row in rows]

    def get_adjacent_chunks(self, chunk_id: int) -> dict:
        """
        특정 청크(integer id)의 이전/다음 청크 반환.
        같은 section_path + fiscal_year 내에서 id 기준 전후 청크를 찾음.

        Returns:
            {"current": {...}, "prev": {...} or None, "next": {...} or None}
        """
        current_rows = self.conn.execute(
            """
            SELECT id, chunk_uid, fiscal_year, section_path,
                   chunk_type, content, content_length
            FROM chunks WHERE id = ?
            """,
            (chunk_id,)
        ).fetchall()

        if not current_rows:
            return {"current": None, "prev": None, "next": None}

        current = dict(current_rows[0])
        section_path = current["section_path"]
        fiscal_year = current["fiscal_year"]

        prev_rows = self.conn.execute(
            """SELECT id, chunk_uid, fiscal_year, section_path, content, content_length
               FROM chunks
               WHERE section_path = ? AND fiscal_year = ? AND id < ?
               ORDER BY id DESC LIMIT 1""",
            (section_path, fiscal_year, chunk_id)
        ).fetchall()

        next_rows = self.conn.execute(
            """SELECT id, chunk_uid, fiscal_year, section_path, content, content_length
               FROM chunks
               WHERE section_path = ? AND fiscal_year = ? AND id > ?
               ORDER BY id ASC LIMIT 1""",
            (section_path, fiscal_year, chunk_id)
        ).fetchall()

        return {
            "current": current,
            "prev": dict(prev_rows[0]) if prev_rows else None,
            "next": dict(next_rows[0]) if next_rows else None,
        }

    # ─── 태그 검색 ────────────────────────────────────────────

    def search_by_tags(
        self,
        tags: list[str],
        years: Optional[list[int]] = None,
        limit: int = 50,
    ) -> list[dict]:
        """태그로 청크 검색"""
        tag_placeholders = ",".join("?" * len(tags))
        sql = f"""
            SELECT DISTINCT c.id, c.chunk_uid, c.fiscal_year, c.section_path,
                   c.chunk_type, c.content, c.content_length
            FROM chunks c
            JOIN tags t ON c.chunk_uid = t.chunk_uid
            WHERE t.tag IN ({tag_placeholders})
        """
        params = list(tags)
        if years:
            year_placeholders = ",".join("?" * len(years))
            sql += f" AND c.fiscal_year IN ({year_placeholders})"
            params.extend(years)
        sql += " ORDER BY c.fiscal_year, c.id LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ─── 메타데이터 조회 ─────────────────────────────────────

    def get_metadata(self, year: Optional[int] = None) -> list[dict]:
        if year:
            rows = self.conn.execute(
                "SELECT * FROM report_metadata WHERE fiscal_year = ?", (year,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM report_metadata ORDER BY fiscal_year"
            ).fetchall()
        return [dict(row) for row in rows]

    # ─── 통계 ────────────────────────────────────────────────

    def stats(self) -> dict:
        result = {}
        result["chunks"] = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        result["tags"] = self.conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        result["report_metadata"] = self.conn.execute(
            "SELECT COUNT(*) FROM report_metadata"
        ).fetchone()[0]
        rows = self.conn.execute(
            "SELECT fiscal_year, COUNT(*) as cnt FROM chunks GROUP BY fiscal_year ORDER BY fiscal_year"
        ).fetchall()
        result["documents_by_year"] = {row["fiscal_year"]: row["cnt"] for row in rows}
        return result

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    db = AuditDB("./data/audit_reports.db")
    print(db.stats())
    db.close()
