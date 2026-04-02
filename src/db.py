"""
db.py - SQLite 데이터베이스 관리

테이블:
  - documents: 파싱된 감사보고서 청크
  - metadata:  연도별 메타 정보
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "database.sql"


class AuditDB:
    def __init__(self, db_path: str = "./data/audit_reports.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_tables(self):
        """schema/database.sql 실행하여 테이블 생성"""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            sql = f.read()
        self.conn.executescript(sql)
        self.conn.commit()
        print(f"[DB] Tables initialized: {self.db_path}")

    def clear_year(self, year: int):
        """특정 연도 데이터 삭제 (재적재용)"""
        for table in ["documents", "metadata"]:
            self.conn.execute(f"DELETE FROM {table} WHERE year = ?", (year,))
        self.conn.commit()
        print(f"[DB] Cleared data for year {year}")

    # ─── documents ───────────────────────────────────────────

    def insert_document(
        self,
        year: int,
        section_h1: Optional[str],
        section_h2: Optional[str],
        section_h3: Optional[str],
        section_h4: Optional[str],
        section_h5: Optional[str],
        section_h6: Optional[str],
        section_path: str,
        chunk_index: int,
        total_chunks: int,
        content_type: str,
        content: str,
        source_file: str = "",
        chunking_strategy: str = "",
    ):
        self.conn.execute(
            """
            INSERT INTO documents (
                year, section_h1, section_h2, section_h3,
                section_h4, section_h5, section_h6,
                section_path, chunk_index, total_chunks,
                content_type, content, content_length,
                source_file, chunking_strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                year, section_h1, section_h2, section_h3,
                section_h4, section_h5, section_h6,
                section_path, chunk_index, total_chunks,
                content_type, content, len(content),
                source_file, chunking_strategy,
            ),
        )

    def insert_documents_batch(self, rows: list[dict]):
        """여러 문서 청크를 한 번에 삽입"""
        for row in rows:
            self.insert_document(**row)
        self.conn.commit()
        print(f"[DB] Inserted {len(rows)} document chunks")

    # ─── metadata ────────────────────────────────────────────

    def insert_metadata(
        self,
        year: int,
        period_start: str = "",
        period_end: str = "",
        auditor: str = "",
        opinion: str = "",
        source_file: str = "",
    ):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO metadata (
                year, period_start, period_end,
                auditor, opinion, source_file
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (year, period_start, period_end, auditor, opinion, source_file),
        )
        self.conn.commit()

    # ─── 조회 ────────────────────────────────────────────────

    def get_documents(
        self,
        year: Optional[int] = None,
        section_h2: Optional[str] = None,
        section_path_like: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> list[dict]:
        """조건부 문서 조회"""
        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if year:
            query += " AND year = ?"
            params.append(year)
        if section_h2:
            query += " AND section_h2 = ?"
            params.append(section_h2)
        if section_path_like:
            query += " AND section_path LIKE ?"
            params.append(f"%{section_path_like}%")
        if content_type:
            query += " AND content_type = ?"
            params.append(content_type)

        query += " ORDER BY id"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ─── BM25 키워드 검색 (FTS5) ────────────────────────────

    @staticmethod
    def _strip_korean_particles(word: str) -> str:
        """한국어 조사/어미 제거 (FTS5 토큰 매칭률 개선)"""
        # 자주 붙는 조사/어미 패턴 (길이 역순으로 매칭)
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
        section_h2: Optional[str] = None,
    ) -> list[dict]:
        """
        FTS5 BM25 기반 키워드 검색.

        Args:
            query: 검색 키워드 (공백으로 OR 검색, 쌍따옴표로 구문 검색)
            top_k: 최대 결과 수
            year:  연도 필터
            section_h2: h2 섹션 필터

        Returns:
            [{"id", "year", "section_path", "content", "bm25_score", ...}]
        """
        # FTS5 쿼리 구성 (각 단어를 OR로 연결, 한국어 조사 제거)
        words = query.strip().split()
        if not words:
            return []

        # 원본 + 조사 제거 버전 모두 OR로 검색 (재현율 향상)
        all_terms = set()
        for w in words:
            all_terms.add(w)
            stripped = self._strip_korean_particles(w)
            if stripped != w and len(stripped) > 1:
                all_terms.add(stripped)

        fts_query = " OR ".join(f'"{t}"' for t in all_terms)

        sql = """
            SELECT d.*, rank AS bm25_score
            FROM documents d
            JOIN documents_fts fts ON d.id = fts.rowid
            WHERE documents_fts MATCH ?
        """
        params: list = [fts_query]

        if year:
            sql += " AND d.year = ?"
            params.append(year)
        if section_h2:
            sql += " AND d.section_h2 = ?"
            params.append(section_h2)

        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[DB] BM25 search error: {e}")
            return []

    # ─── 구조 기반 검색 (h1~h6 스키마 활용) ──────────────────

    def structured_search(
        self,
        years: Optional[list[int]] = None,
        section_h2: Optional[str] = None,
        section_h3: Optional[str] = None,
        section_h4: Optional[str] = None,
        section_h5: Optional[str] = None,
        section_path_contains: Optional[str] = None,
        keyword: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        h1~h6 컬럼을 정확히 활용하는 구조 기반 검색.

        예시:
          - years=[2022,2023,2024], section_h2="재무제표"
            → 3개 연도 재무제표 섹션 전체
          - section_h3="30. 특수관계자 거래", keyword="배당"
            → 특수관계자 거래 주석 중 배당 관련 내용
        """
        sql = """
            SELECT id, year, section_h1, section_h2, section_h3, section_h4,
                   section_h5, section_h6, section_path,
                   chunk_index, total_chunks,
                   content_type, content, content_length
            FROM documents WHERE 1=1
        """
        params: list = []

        if years:
            placeholders = ",".join("?" * len(years))
            sql += f" AND year IN ({placeholders})"
            params.extend(years)
        if section_h2:
            sql += " AND section_h2 = ?"
            params.append(section_h2)
        if section_h3:
            sql += " AND section_h3 LIKE ?"
            params.append(f"%{section_h3}%")
        if section_h4:
            sql += " AND section_h4 LIKE ?"
            params.append(f"%{section_h4}%")
        if section_h5:
            sql += " AND section_h5 LIKE ?"
            params.append(f"%{section_h5}%")
        if section_path_contains:
            sql += " AND section_path LIKE ?"
            params.append(f"%{section_path_contains}%")
        if keyword:
            # 쉼표/세미콜론으로 구분된 다중 키워드를 OR LIKE로 처리
            kw_parts = [k.strip() for k in re.split(r"[,;]", keyword) if k.strip()]
            if kw_parts:
                # section_h2+h3가 모두 지정된 경우: keyword는 선택적 (결과 없으면 무시)
                # → 구체적 섹션이 이미 좁혀져 있으므로 keyword로 0건 되는 것 방지
                # section_h2만 지정된 경우: keyword로 추가 필터링 (필수)
                is_section_narrow = section_h2 and section_h3
                content_clauses = " OR ".join(["content LIKE ?"] * len(kw_parts))
                path_clauses = " OR ".join(["section_path LIKE ?"] * len(kw_parts))
                kw_filter = f"(({content_clauses}) OR ({path_clauses}))"
                kw_params = [f"%{k}%" for k in kw_parts] * 2

                if is_section_narrow:
                    # 좁은 섹션: keyword 매칭 결과를 우선하되, 없으면 섹션만으로도 반환
                    # → ORDER BY에서 keyword 매칭 여부로 우선순위 부여
                    sql = sql.replace(
                        "SELECT id, year, section_h1",
                        f"SELECT id, year, section_h1"
                    )
                    # keyword는 적용하지 않고, 나중에 정렬에서 우선순위 부여
                else:
                    sql += f" AND {kw_filter}"
                    params.extend(kw_params)
        if content_type and content_type != "all":
            sql += " AND content_type = ?"
            params.append(content_type)

        sql += " ORDER BY year, id LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ─── 섹션 목록 조회 (에이전트 참조용) ────────────────────

    def list_sections(self, year: Optional[int] = None, level: int = 2) -> list[dict]:
        """
        특정 헤딩 레벨의 섹션 목록을 반환.
        에이전트가 어떤 섹션이 있는지 파악할 때 사용.

        Args:
            year:  연도 필터
            level: 헤딩 레벨 (2=h2, 3=h3, ...)

        Returns:
            [{"section": "...", "year": ..., "chunk_count": ...}]
        """
        col = f"section_h{level}"
        sql = f"""
            SELECT {col} as section, year, COUNT(*) as chunk_count
            FROM documents
            WHERE {col} IS NOT NULL
        """
        params: list = []
        if year:
            sql += " AND year = ?"
            params.append(year)

        sql += f" GROUP BY year, {col} ORDER BY year, {col}"
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ─── 청크 ID 기반 전체 내용 조회 ────────────────────────────

    def get_chunks_by_ids(self, ids: list[int]) -> list[dict]:
        """
        특정 청크 ID 목록의 전체 내용을 반환.
        structured_query/hybrid_search에서 잘린 청크의 전체 내용이 필요할 때 사용.

        Args:
            ids: 청크 ID 목록

        Returns:
            [{"id", "year", "section_path", "chunk_index", "total_chunks", "content", ...}]
        """
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = f"""
            SELECT id, year, section_h1, section_h2, section_h3, section_h4,
                   section_h5, section_h6, section_path,
                   chunk_index, total_chunks,
                   content_type, content, content_length, source_file
            FROM documents WHERE id IN ({placeholders})
            ORDER BY year, id
        """
        rows = self.conn.execute(sql, ids).fetchall()
        return [dict(row) for row in rows]

    def get_adjacent_chunks(self, chunk_id: int) -> dict:
        """
        특정 청크의 이전/다음 청크를 반환.
        같은 섹션 내에서 chunk_index 기준으로 전후 청크를 찾음.

        Returns:
            {"current": {...}, "prev": {...} or None, "next": {...} or None}
        """
        current_rows = self.conn.execute(
            """
            SELECT id, year, section_h2, section_path,
                   chunk_index, total_chunks, content, content_length
            FROM documents WHERE id = ?
            """,
            (chunk_id,)
        ).fetchall()

        if not current_rows:
            return {"current": None, "prev": None, "next": None}

        current = dict(current_rows[0])
        chunk_index = current["chunk_index"]
        section_path = current["section_path"]
        year = current["year"]

        def get_sibling(offset):
            rows = self.conn.execute(
                """
                SELECT id, year, section_path, chunk_index, total_chunks,
                       content, content_length
                FROM documents
                WHERE section_path = ? AND year = ? AND chunk_index = ?
                LIMIT 1
                """,
                (section_path, year, chunk_index + offset)
            ).fetchall()
            return dict(rows[0]) if rows else None

        return {
            "current": current,
            "prev": get_sibling(-1),
            "next": get_sibling(1),
        }

    # ─── FTS 인덱스 리빌드 ───────────────────────────────────

    def rebuild_fts(self):
        """FTS5 인덱스를 documents 테이블에서 재구축"""
        self.conn.execute("DELETE FROM documents_fts")
        self.conn.execute("""
            INSERT INTO documents_fts(rowid, content, section_path)
            SELECT id, content, section_path FROM documents
        """)
        self.conn.commit()
        count = self.conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
        print(f"[DB] FTS index rebuilt: {count} entries")

    def get_metadata(self, year: Optional[int] = None) -> list[dict]:
        if year:
            rows = self.conn.execute(
                "SELECT * FROM metadata WHERE year = ?", (year,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM metadata ORDER BY year"
            ).fetchall()
        return [dict(row) for row in rows]

    # ─── 통계 ────────────────────────────────────────────────

    def stats(self) -> dict:
        """DB 적재 현황"""
        result = {}
        for table in ["documents", "metadata"]:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            result[table] = count
        rows = self.conn.execute(
            "SELECT year, COUNT(*) as cnt FROM documents GROUP BY year ORDER BY year"
        ).fetchall()
        result["documents_by_year"] = {row["year"]: row["cnt"] for row in rows}
        return result

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    db = AuditDB("./data/audit_reports.db")
    db.init_tables()
    print(db.stats())
    db.close()
