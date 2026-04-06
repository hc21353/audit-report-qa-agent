# 데이터베이스 가이드

## 스키마 개요

SQLite(`data/audit_reports.db`)는 4개 테이블로 구성된다.

| 테이블 | 역할 |
|--------|------|
| `chunks` | 모든 청크 (텍스트 + 메타데이터) |
| `tags` | 청크-태그 매핑 (다대다 정규화) |
| `report_metadata` | 연도별 감사보고서 메타 (auditor, opinion 등) |
| `fts_chunks` | FTS5 전문검색 인덱스 (BM25) |

---

## 데이터 흐름

```
JSONL (semantic_chunks_tagged.jsonl)
    │
    ├─[build_db.py]──────────────────────────────────────────────────────┐
    │                                                                     ↓
    │   ┌──────────────────────────────────────────────────────────────┐ │
    │   │ chunks 테이블                                                  │ │
    │   │  chunk_uid, fiscal_year, section_path, chunk_type,           │ │
    │   │  content, char_count, is_consolidated, csv_refs, ...         │ │
    │   ├──────────────────────────────────────────────────────────────┤ │
    │   │ tags 테이블                                                    │ │
    │   │  chunk_uid, tag, tag_category                                 │ │
    │   ├──────────────────────────────────────────────────────────────┤ │
    │   │ report_metadata 테이블                                         │ │
    │   │  fiscal_year, auditor, opinion, report_date, ...             │ │
    │   ├──────────────────────────────────────────────────────────────┤ │
    │   │ fts_chunks (FTS5 인덱스)                                      │ │
    │   │  content 전문검색용 역인덱스                                   │ │
    │   └──────────────────────────────────────────────────────────────┘ │
    │                                                                     │
    └─[build_vectordb.py]─────────────────────────────────────────────────┘
         chunks.content → KoE5 임베딩 → ChromaDB (samsung_audit 컬렉션)
         메타데이터: chunk_uid, fiscal_year, section_path
```

---

## 주요 컬럼 설명 (`chunks`)

| 컬럼 | 설명 |
|------|------|
| `chunk_uid` | 고유 식별자 (`{year}_{section_hash}_{idx}`) |
| `fiscal_year` | 연도 (2014~2024) |
| `section_path` | `"(첨부)재무제표 > 주석 > 9. 종속기업"` 형태의 계층 경로 |
| `chunk_type` | `Narrative` / `Note` / `Table_Row` |
| `content` | 실제 텍스트 내용 |
| `is_consolidated` | 1=연결재무제표, 0=별도재무제표 |
| `csv_refs` | 연결된 CSV 파일 경로 (JSON 배열) |

---

## 자주 쓰는 쿼리 패턴

### 특정 연도·섹션 조회
```sql
SELECT chunk_uid, section_path, content
FROM chunks
WHERE fiscal_year = 2024
  AND section_path LIKE '%주석%'
  AND section_path LIKE '%특수관계자%';
```

### 연도별 같은 섹션 비교
```sql
SELECT fiscal_year, content
FROM chunks
WHERE section_path LIKE '%(첨부)재무제표%'
  AND chunk_type = 'Table_Row'
ORDER BY fiscal_year;
```

### 감사의견 전체 이력 조회
```sql
SELECT fiscal_year, opinion, auditor
FROM report_metadata
ORDER BY fiscal_year;
```

### FTS5 BM25 전문검색
```sql
SELECT c.chunk_uid, c.fiscal_year, c.section_path, c.content,
       bm25(fts_chunks) AS score
FROM fts_chunks
JOIN chunks c USING (chunk_uid)
WHERE fts_chunks MATCH '총자산 유동자산'
  AND c.fiscal_year = 2024
ORDER BY score
LIMIT 10;
```

### 태그 기반 조회
```sql
SELECT c.chunk_uid, c.content
FROM chunks c
JOIN tags t USING (chunk_uid)
WHERE t.tag = '재무위험'
  AND c.fiscal_year = 2023;
```

---

## ChromaDB 연동

벡터 검색 결과의 `id`는 `chunks.chunk_uid`와 동일하다.
벡터 검색 후 SQLite에서 전체 메타데이터를 조회하는 방식:

```python
from src.db import AuditDB
import chromadb

# 벡터 검색
results = collection.query(
    query_embeddings=[query_vec],
    n_results=10,
    include=["metadatas", "distances"],
)
chunk_uids = results["ids"][0]

# SQLite에서 전체 내용 조회
db = AuditDB("./data/audit_reports.db")
chunks = db.get_chunks_by_ids(chunk_uids)
```

---

## AuditDB 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `bm25_search(query, top_k, year, section_path_contains)` | FTS5 BM25 검색 |
| `list_sections(year, level)` | 섹션 목록 조회 (level=1: 최상위) |
| `get_chunks_by_ids(ids)` | chunk_uid 배열로 전체 내용 조회 |
| `get_adjacent_chunks(chunk_id)` | 인접 청크 조회 (맥락 확장) |
| `get_metadata(year)` | 연도별 감사 메타 조회 |
