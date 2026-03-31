# 시스템 아키텍처

## 개요

삼성전자 감사보고서(2014~2024) 10년치를 기반으로 한 멀티 에이전트 Agentic RAG 시스템.

## 전체 파이프라인

```
HTML 감사보고서 → 파싱 → 마크다운 → 구조 기반 청킹 → SQLite + 벡터DB
                                                          ↓
                              사용자 질문 → Orchestrator → Query Rewriter
                                                          ↓
                                               Retriever (벡터 + SQL)
                                                          ↓
                                               Analyst (LLM + Tools)
                                                          ↓
                                                        답변
```

## 계층 구조

### 1. 데이터 레이어

| 저장소 | 용도 | 검색 방식 |
|---|---|---|
| SQLite `documents` | 비정형 텍스트 청크 | section_path 필터링 |
| SQLite `financial_data` | 정형 재무 수치 | SQL 쿼리 |
| SQLite `metadata` | 연도별 메타 | SQL 쿼리 |
| FAISS / Chroma | 임베딩 벡터 | 의미 유사도 검색 |

### 2. 에이전트 레이어 (LangGraph)

```
┌─────────────┐
│ Orchestrator │  질문 의도 분류, 서브 질문 분해
└──────┬──────┘
       ↓
┌──────────────┐
│Query Rewriter│  검색 최적화 쿼리 생성 (최대 3개)
└──────┬───────┘
       ↓
┌──────────┐
│ Retriever│  벡터 검색 + SQL 조회 + 섹션 필터
└──────┬───┘
       ↓
┌─────────┐
│ Analyst │  답변 생성, 계산, 차트, 웹 검색
└────┬────┘
     ↓ (추가 검색 필요 시 Retriever로 루프백)
   답변
```

### 3. Tool 레이어

- **Internal**: vector_search, sql_query, section_filter, calculator, chart_generator
- **MCP**: web_search, dart_api (예정)

### 4. 인터페이스 레이어

- Streamlit 기반 채팅 UI
- 연도 선택, 섹션 필터, 소스 표시, 차트 렌더링

## 핵심 설계 결정

### 왜 SQLite + 벡터DB 이중 구조인가?

- "2024년 총자산" → SQL이 정확 (벡터 검색은 유사 문서를 가져올 뿐 숫자 정확도 보장 불가)
- "핵심감사사항 설명" → 벡터 검색이 적합 (의미 기반)
- 에이전트가 질문 유형에 따라 적절한 도구를 선택

### 왜 마크다운 헤딩 = DB 컬럼인가?

- 감사보고서는 법적으로 구조가 정해져 있어 헤딩 구조가 안정적
- "주석 30번의 특수관계자 거래" 같은 정밀 필터링 가능
- 연도별 같은 섹션 비교가 section_path 기준으로 간단해짐
