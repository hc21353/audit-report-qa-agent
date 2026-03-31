# 데이터베이스 스키마

## 개요

SQLite 기반 3개 테이블 구조. 비정형 텍스트(documents)와 정형 수치(financial_data)를 분리하여 하이브리드 검색을 지원한다.

## 테이블

### documents

파싱된 감사보고서 텍스트를 청크 단위로 저장한다.

**핵심 설계 원칙**: 마크다운 헤딩 계층(`#` ~ `######`)이 곧 DB 컬럼이다.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| id | INTEGER PK | 자동 증가 |
| year | INTEGER | 감사보고서 연도 |
| section_h1 ~ h6 | TEXT | 마크다운 헤딩 레벨별 섹션명 |
| section_path | TEXT | 전체 경로 (예: `재무제표 > 주석 > 4. 범주별 금융상품`) |
| chunk_index | INTEGER | 같은 섹션 내 청크 순서 (0부터) |
| total_chunks | INTEGER | 해당 섹션의 총 청크 수 (1이면 분할 안 됨) |
| content_type | TEXT | `text` / `table` / `mixed` |
| content | TEXT | 실제 텍스트 내용 |
| content_length | INTEGER | 글자 수 |
| source_file | TEXT | 원본 파일명 |
| chunking_strategy | TEXT | 사용된 청킹 전략 |
| parsed_at | TIMESTAMP | 파싱 시각 |

**청킹 흐름 예시:**

```
## 주석 (→ section_h2)
  ### 30. 특수관계자 거래 (→ section_h3)
    ##### 다. 채권채무 (→ section_h5, content 3000자 + 표)
      → chunk_index=0, total_chunks=2: 텍스트 부분
      → chunk_index=1, total_chunks=2: 표 부분
```

### metadata

연도별 감사보고서 메타 정보.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| year | INTEGER | 연도 (UNIQUE) |
| period_start | TEXT | 회계기간 시작 |
| period_end | TEXT | 회계기간 종료 |
| auditor | TEXT | 감사법인명 |
| opinion | TEXT | 감사의견 (적정/한정/부적정/의견거절) |
