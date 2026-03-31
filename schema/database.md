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

### financial_data

재무제표 숫자를 정형 데이터로 저장. SQL로 직접 연도 비교, 비율 계산 가능.

| 컬럼 | 타입 | 설명 |
|---|---|---|
| year | INTEGER | 연도 |
| statement_type | TEXT | `balance_sheet` / `income_statement` / `cash_flow` 등 |
| account_name | TEXT | 계정명 (예: 유동자산) |
| account_level | INTEGER | 깊이 (1=대분류, 2=중분류, 3=소분류) |
| parent_account | TEXT | 상위 계정명 |
| current_amount | REAL | 당기 금액 (백만원) |
| previous_amount | REAL | 전기 금액 (백만원) |

**활용 예시:**

```sql
-- 2018 vs 2022 부채총계 비교
SELECT year, account_name, current_amount
FROM financial_data
WHERE account_name = '부채총계' AND year IN (2018, 2022);

-- 최근 5년 유동비율 추이
SELECT f1.year,
       f1.current_amount AS 유동자산,
       f2.current_amount AS 유동부채,
       ROUND(f1.current_amount * 100.0 / f2.current_amount, 2) AS 유동비율
FROM financial_data f1
JOIN financial_data f2 ON f1.year = f2.year
WHERE f1.account_name = '유동자산'
  AND f2.account_name = '유동부채'
  AND f1.statement_type = 'balance_sheet'
  AND f2.statement_type = 'balance_sheet'
  AND f1.year >= 2020
ORDER BY f1.year;
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
