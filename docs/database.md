# 데이터베이스 가이드

> 스키마 상세: [`../schema/database.md`](../schema/database.md)
> DDL: [`../schema/database.sql`](../schema/database.sql)

## 데이터 흐름

```
HTML 감사보고서
  ↓ (파싱)
마크다운 (감사보고서_2024.md)
  ↓ (구조 분할 + 청킹)
┌──────────────────────────────┐
│ documents 테이블              │  비정형 텍스트 청크
│  - 헤딩 구조 → 컬럼          │
│  - 최하위 content → 청크      │
├──────────────────────────────┤
│ financial_data 테이블         │  정형 재무 수치
│  - 재무상태표 숫자 추출       │
│  - 손익계산서 숫자 추출       │
├──────────────────────────────┤
│ metadata 테이블               │  연도별 메타 정보
│  - 감사법인, 감사의견 등      │
└──────────────────────────────┘
  ↓ (임베딩)
벡터 DB (FAISS / Chroma)
  - documents.content 임베딩
  - documents.id 로 연결
```

## 자주 쓰는 쿼리 패턴

### 특정 연도/섹션 조회
```sql
SELECT content FROM documents
WHERE year = 2024 AND section_h2 = '주석' AND section_h3 LIKE '%특수관계자%';
```

### 연도별 같은 계정 비교
```sql
SELECT year, current_amount FROM financial_data
WHERE account_name = '자산총계'
ORDER BY year;
```

### 감사의견 변경 확인
```sql
SELECT year, opinion FROM metadata ORDER BY year;
```

## 벡터 DB 연동

벡터 DB의 각 벡터는 `documents.id`를 메타데이터로 갖는다. 벡터 검색 후 해당 id로 SQLite에서 전체 메타데이터를 조회하는 방식.

```python
# 벡터 검색 → id 획득 → SQLite 조회
results = vector_store.search(query, top_k=10)
doc_ids = [r.metadata["doc_id"] for r in results]
chunks = db.execute(
    "SELECT * FROM documents WHERE id IN ({})".format(",".join("?"*len(doc_ids))),
    doc_ids
)
```
