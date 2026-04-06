# Tool 가이드

## 개요

에이전트가 사용하는 Tool은 **Retriever용**과 **Analyst용**으로 나뉜다.

| Tool | 사용 에이전트 | 카테고리 |
|------|-------------|---------|
| `hybrid_search` | Retriever | 검색 |
| `structured_query` | Retriever | 검색 |
| `list_available_sections` | Retriever | 검색 보조 |
| `csv_reader` | Retriever, Analyst | 데이터 로드 |
| `db_fetch` | Retriever | 검색 보조 |
| `web_search` | Retriever | 외부 검색 |
| `calculator` | Analyst | 분석 |
| `chart_generator` | Analyst | 시각화 |

---

## 검색 도구 (Retriever)

### hybrid_search

KoE5 벡터 유사도 + SQLite FTS5 BM25를 RRF(Reciprocal Rank Fusion)로 결합한 하이브리드 검색.

**언제 사용**: 개념적 질문, 서술형 설명, 정책·방침 내용
```
예: "건설중인자산의 감가상각 개시시점 정책은?"
    "핵심감사사항에서 언급된 위험 요소는?"
```

**주요 파라미터**
- `query`: 검색 텍스트
- `years`: 연도 필터 (없으면 전체)
- `section_path_contains`: 섹션 경로 부분 문자열 필터
- `top_k`: 최대 반환 수 (기본 10)

---

### structured_query

`section_path` 기반 SQL 정밀 조회. 특정 섹션의 청크를 직접 가져올 때 사용.

**언제 사용**: 특정 섹션 전체 조회, 구조적으로 명확한 위치의 내용
```
예: "독립된 감사인의 감사보고서 전체 내용"
    "주석 30번 특수관계자 거래 내용"
```

**주요 파라미터**
- `section_path_contains`: 섹션 경로 키워드 (예: `"특수관계자"`)
- `year`: 연도 필터
- `chunk_type`: `Narrative` / `Note` / `Table_Row`
- `is_consolidated`: 1=연결, 0=별도

---

### list_available_sections

실제 DB에서 현재 존재하는 섹션 목록을 동적으로 조회. Retriever가 검색 계획을 세울 때 가장 먼저 호출.

**역할**: Retriever LLM이 존재하지 않는 섹션을 검색하지 않도록 실제 구조 확인
```
예 반환값:
  - "(첨부)재무제표"
  - "독립된 감사인의 감사보고서"
  - "내부회계관리제도 감사 또는 검토의견"
  - "외부감사 실시내용"
```

---

### db_fetch

`chunk_uid`로 청크 전체 내용을 직접 조회. 인접 청크 맥락 확장에도 사용.

**언제 사용**: 하이브리드 검색 결과에서 chunk_uid를 얻은 후 전체 원문이 필요할 때

---

### csv_reader

`[TABLE_CSV:파일명]` 태그가 감지된 청크에서 해당 CSV 파일을 로드.

**언제 사용**: 재무제표 표 데이터, 수치 비교가 필요한 Table_Row 청크

**동작 흐름**
```
1. Retriever가 [TABLE_CSV:tables/2024_재무상태표.csv] 태그 감지
2. csv_reader 호출 → CSV 로드 → state["csv_data"]에 저장
3. Analyst가 csv_data를 컨텍스트로 활용하여 답변 생성
```

---

### web_search

DuckDuckGo를 통한 웹 검색. 감사보고서에 없는 최신 정보 보충용.

**언제 사용**: 최신 뉴스, 시장 동향, K-IFRS 개정 내용 등
```
예: "2025년 K-IFRS 17호 개정 내용"
    "삼성전자 최근 반도체 부문 동향"
```

---

## 분석 도구 (Analyst)

### calculator

재무비율 및 수식 계산. 프리셋과 커스텀 수식 모두 지원.

**프리셋 목록**
| 프리셋 | 수식 |
|--------|------|
| 유동비율 | 유동자산 / 유동부채 × 100 |
| 부채비율 | 부채총계 / 자본총계 × 100 |
| ROE | 당기순이익 / 자본총계 × 100 |
| ROA | 당기순이익 / 자산총계 × 100 |
| 영업이익률 | 영업이익 / 매출액 × 100 |

**Analyst 사용 방식**

Analyst가 답변에 `CALC:` 패턴을 포함하면 자동으로 calculator가 호출된다:
```
CALC: 324966127 / 143143547 * 100
→ calculator 호출 → 결과: 227.0%
```

---

### chart_generator

Plotly 기반 차트 생성. 생성된 차트는 Streamlit UI에서 렌더링된다.

**지원 차트 타입**
- `bar`: 막대 차트 (연도별 비교)
- `line`: 선 차트 (추이 분석)
- `stacked_bar`: 누적 막대 (구성 비율)
- `pie`: 파이 차트 (비중)

**사용 흐름**
```
예: "최근 5년 총자산 추이 차트로 보여줘"
    → Retriever: 5년치 총자산 수치 검색
    → Analyst: chart_generator(type=line, years=[2020..2024], values=[...])
    → Streamlit: 차트 렌더링
```
