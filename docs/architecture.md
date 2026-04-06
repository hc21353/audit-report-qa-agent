# 시스템 아키텍처

## 개요

삼성전자 감사보고서(2014~2024) **11년치**를 기반으로 한 멀티 에이전트 Agentic RAG 시스템.

---

## 전체 파이프라인

```
JSONL (semantic_chunks_tagged.jsonl)
  │
  ├─[build_db.py]──────→ SQLite (chunks / tags / metadata / fts_chunks)
  │
  └─[build_vectordb.py]─→ ChromaDB (KoE5, 1024dim)
                                        │
                  사용자 질문 ────────────┘
                      ↓
                Orchestrator  (의도 분류 + 서브 질문 분해)
                      ↓
               Query Rewriter  (검색 쿼리 최적화, 최대 3개)
                      ↓
                  Retriever  (LLM 기반 자율 검색)
                    ├─ hybrid_search   (KoE5 벡터 + BM25 RRF)
                    ├─ structured_query (SQL 정밀 검색)
                    └─ csv_reader       ([TABLE_CSV] 자동 로드)
                      ↓
                   Analyst  (LLM + Tools)
                    ├─ calculator
                    └─ chart_generator
                      ↓ (NEED_MORE_SEARCH → Retriever 루프백, 최대 5회)
                    최종 답변
```

---

## 계층 구조

### 1. 데이터 레이어

| 저장소 | 테이블 / 컬렉션 | 용도 | 검색 방식 |
|--------|----------------|------|----------|
| SQLite | `chunks` | 비정형 텍스트 청크 (section_path, content, chunk_type) | FTS5 BM25, SQL 필터 |
| SQLite | `tags` | 청크-태그 매핑 (다대다) | JOIN |
| SQLite | `report_metadata` | 연도별 메타 (auditor, opinion 등) | SQL |
| SQLite | `fts_chunks` | FTS5 전문검색 인덱스 | BM25 (MATCH) |
| ChromaDB | `samsung_audit` | KoE5 임베딩 벡터 (1024dim, cosine) | 벡터 유사도 |

### 2. 에이전트 레이어 (LangGraph)

```
┌──────────────────────────────────────────────┐
│              LangGraph StateGraph             │
│                                               │
│  ┌─────────────┐    simple_lookup             │
│  │ Orchestrator│──────────────────────┐       │
│  └──────┬──────┘                      │       │
│         │ (그 외)                      ↓       │
│  ┌──────────────┐            ┌──────────────┐ │
│  │Query Rewriter│──────────→ │   Retriever  │ │
│  └──────────────┘            └──────┬───────┘ │
│                                     ↓         │
│                             ┌──────────────┐  │
│                             │   Analyst    │  │
│                             └──────┬───────┘  │
│                      need_more ────┘   done → END
└──────────────────────────────────────────────┘
```

**Orchestrator**
- 규칙 기반 사전 분석 (confidence=high이면 LLM 스킵, 지연 최소화)
- 의도: `simple_lookup` / `comparison` / `trend` / `calculation` / `general`
- 연도·섹션·키워드 추출, 복합 질문 서브 질문 분해

**Query Rewriter**
- simple_lookup은 건너뜀 (orchestrator에서 직접 retriever로)
- 시맨틱/구조 검색용 쿼리 최대 3개 생성

**Retriever**
- `list_available_sections`로 실제 DB 섹션 구조 확인 후 검색 계획 수립
- `hybrid_search`: KoE5 벡터 + SQLite FTS5 BM25 결합 (RRF)
- `structured_query`: section_path 기반 SQL 정밀 조회
- `[TABLE_CSV]` 태그 감지 → `csv_reader`로 자동 로드

**Analyst**
- 검색 결과 컨텍스트화 → LLM 답변 생성
- `CALC:` 패턴 감지 → calculator 호출
- `NEED_MORE_SEARCH:` 패턴 감지 → Retriever 루프백 (최대 5회)

### 3. Tool 레이어

| Tool | 에이전트 | 용도 |
|------|---------|------|
| `hybrid_search` | Retriever | KoE5 벡터 + BM25 하이브리드 검색 |
| `structured_query` | Retriever | section_path SQL 정밀 검색 |
| `list_available_sections` | Retriever | 실제 DB 섹션 목록 동적 조회 |
| `csv_reader` | Retriever, Analyst | [TABLE_CSV] 파일 로드 |
| `db_fetch` | Retriever | chunk_uid로 전체 내용 조회 |
| `web_search` | Retriever | DuckDuckGo 웹 검색 |
| `calculator` | Analyst | 재무비율 계산 + 커스텀 수식 |
| `chart_generator` | Analyst | Plotly 차트 생성 |

### 4. 인터페이스 레이어

- Streamlit 기반 채팅 UI (`src/app.py`)
- **사이드바**: 연도/섹션 필터, 에이전트별 LLM 모델 선택기, 벡터 DB 상태
- **멀티턴**: 최근 3턴 대화 이력을 쿼리 컨텍스트로 자동 주입 (`app.py` 단독 구현)
- 진행 상황: 각 에이전트 노드 완료 시 상세 정보 실시간 표시

---

## 핵심 설계 결정

### SQLite + ChromaDB 이중 구조

- **수치 질문** ("2024년 총자산"): SQL이 정확 — 벡터 검색은 유사 문서만 반환
- **서술 질문** ("핵심감사사항 설명"): KoE5 벡터 검색이 적합 — 의미 기반
- Retriever가 질문 유형에 따라 적절한 tool 조합을 자율 결정

### section_path = 계층 구조 키

- 감사보고서는 법적으로 헤딩 구조가 고정 → `section_path`로 안정적 필터링
- `"(첨부)재무제표 > 주석 > 30. 특수관계자 거래"` 같은 정밀 조회 가능
- 연도 비교 시 `section_path` 기준으로 동일 섹션 대조

### 생성 모델 vs 채점 모델 분리

- **생성**: `runs.yaml`의 `model`/`models` 필드 또는 UI 모델 선택기로 변경 가능
- **채점**: `eval/judge.py`의 `JUDGE_MODEL` 상수로 고정 → 여러 모델 비교 시 공정한 기준 유지

### 규칙 기반 사전 분석 (Orchestrator 최적화)

- 단순 질문 (confidence=high)은 LLM 호출 없이 규칙으로 처리 → 지연 감소
- 복합/불명확 질문만 LLM 호출

---

## LLM 백엔드 우선순위

```
agents.yaml [에이전트별 backend 직접 지정]
    ↓ (null이면)
runtime.yaml [agent_backends 섹션]
    ↓ (null이면)
runtime.yaml [llm.default_backend]
```

에이전트별 `model`도 같은 우선순위로 결정됨.
