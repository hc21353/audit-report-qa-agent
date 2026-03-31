-- ============================================================
-- database.sql - 삼성전자 감사보고서 RAG 시스템 스키마
-- ============================================================

-- 파싱된 감사보고서 청크
-- 상위 헤딩 → 컬럼, 최하위 헤딩 내 content → 청킹 후 저장
CREATE TABLE IF NOT EXISTS documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    year             INTEGER NOT NULL,

    -- 마크다운 헤딩 기반 섹션 구조 (각 # 레벨별 컬럼)
    section_h1       TEXT,
    section_h2       TEXT,
    section_h3       TEXT,
    section_h4       TEXT,
    section_h5       TEXT,
    section_h6       TEXT,
    section_path     TEXT NOT NULL,     -- 전체 경로 (예: '재무제표 > 주석 > 4. 범주별 금융상품')

    -- 청킹 정보
    chunk_index      INTEGER NOT NULL DEFAULT 0,   -- 같은 섹션 내 청크 순서
    total_chunks     INTEGER NOT NULL DEFAULT 1,   -- 해당 섹션 총 청크 수

    -- 본문
    content_type     TEXT NOT NULL,     -- text | table | mixed
    content          TEXT NOT NULL,
    content_length   INTEGER,

    -- 메타
    source_file      TEXT,
    chunking_strategy TEXT,
    parsed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
CREATE INDEX IF NOT EXISTS idx_documents_section_path ON documents(section_path);
CREATE INDEX IF NOT EXISTS idx_documents_h2 ON documents(section_h2);
CREATE INDEX IF NOT EXISTS idx_documents_h3 ON documents(section_h3);
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);


-- 연도별 감사보고서 메타데이터
CREATE TABLE IF NOT EXISTS metadata (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    year             INTEGER NOT NULL UNIQUE,
    period_start     TEXT,
    period_end       TEXT,
    auditor          TEXT,
    opinion          TEXT,              -- 적정 | 한정 | 부적정 | 의견거절
    source_file      TEXT,
    parsed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- FTS5 가상 테이블 (BM25 키워드 검색용)
-- documents 테이블과 rowid로 연결
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    content,
    section_path,
    content='documents',
    content_rowid='id',
    tokenize='unicode61'
);

-- FTS 인덱스 동기화 트리거
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, content, section_path)
    VALUES (new.id, new.content, new.section_path);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content, section_path)
    VALUES ('delete', old.id, old.content, old.section_path);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, content, section_path)
    VALUES ('delete', old.id, old.content, old.section_path);
    INSERT INTO documents_fts(rowid, content, section_path)
    VALUES (new.id, new.content, new.section_path);
END;
