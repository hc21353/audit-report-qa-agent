# Samsung Audit Report QA Agent

삼성전자 감사보고서(2014~2024) **11년치**를 기반으로 한 **멀티 에이전트 Agentic RAG** 시스템.

## 주요 기능

- **시계열 비교 QA**: "2018 vs 2022 부채 구조 차이" 같은 연도 비교 질문 지원
- **하이브리드 검색**: KoE5 벡터 유사도(ChromaDB) + SQLite FTS5 BM25 검색 (RRF 결합)
- **LLM 기반 자율 검색**: Retriever가 DB 섹션 구조를 직접 조회하여 최적 검색 파라미터 결정
- **동적 CSV 로드**: `[TABLE_CSV]` 태그로 표 데이터를 필요할 때만 로드
- **재무비율 계산**: 유동비율, 부채비율, ROE, ROA 등 자동 계산
- **멀티 백엔드**: 에이전트별로 다른 머신의 Ollama 서버 연결 가능
- **멀티턴 대화**: 이전 대화 컨텍스트를 자동으로 다음 질문에 반영
- **UI 모델 선택기**: Streamlit 사이드바에서 에이전트별 LLM 모델 실시간 변경
- **LLM 비교 평가**: 생성 모델을 바꿔가며 동일 채점 모델로 품질 비교

## 프로젝트 구조

```
project/
├── config/
│   ├── app.yaml          파싱·청킹·임베딩·벡터스토어·UI 설정
│   ├── agents.yaml       에이전트별 모델·백엔드 URL·시스템 프롬프트
│   ├── runtime.yaml      기본 LLM 백엔드 + 에이전트별 오버라이드
│   └── tools.yaml        Tool 스펙 정의
│
├── src/
│   ├── config.py         YAML 설정 로더 (Config 데이터클래스)
│   ├── parser.py         마크다운 파싱 + [TABLE_CSV] 인식
│   ├── chunker.py        2레이어 청킹 (헤딩 분리 → 긴 content 재귀 분할)
│   ├── csv_reader.py     CSV 읽기 유틸리티
│   ├── db.py             SQLite 관리 (chunks, tags, metadata, FTS5)
│   ├── loader.py         데이터 적재 파이프라인
│   ├── build_db.py       파싱 결과(JSONL) → SQLite 구축
│   ├── build_vectordb.py SQLite chunks → KoE5 임베딩 → ChromaDB 구축
│   ├── app.py            Streamlit 채팅 UI (모델 선택기, 멀티턴)
│   │
│   ├── tools/
│   │   ├── hybrid_search.py      KoE5 벡터 + BM25 하이브리드 검색 (RRF)
│   │   ├── structured_query.py   SQL 기반 정밀 섹션 검색
│   │   ├── db_fetch.py           chunk_uid로 전체 내용 조회
│   │   ├── csv_reader_tool.py    [TABLE_CSV] 동적 로드
│   │   ├── calculator.py         재무비율 계산 + 커스텀 수식
│   │   ├── chart_generator.py    Plotly 차트 생성
│   │   └── web_search.py         DuckDuckGo 웹 검색
│   │
│   └── agents/
│       ├── state.py          GraphState 스키마 + initial_state()
│       ├── llm.py            에이전트별 백엔드 LLM 팩토리
│       ├── orchestrator.py   질문 의도 분류 + 서브 질문 분해
│       ├── query_rewriter.py 검색 쿼리 최적화 (최대 3개)
│       ├── retriever.py      LLM 기반 자율 검색 + CSV 자동 로드
│       ├── analyst.py        답변 생성 + 추가 검색 판단
│       ├── graph.py          LangGraph 조립 + run_query / run_query_stream
│       ├── db_context.py     DB 섹션 구조 컨텍스트 빌드
│       └── tracer.py         에이전트 실행 추적 + 로그 저장
│
├── eval/
│   ├── questions.yaml    평가 질문 세트 (ground_truth 포함)
│   ├── runs.yaml         실험 설정 (모델·벡터스토어 조합)
│   ├── run_eval.py       평가 파이프라인 (생성/채점 모델 분리)
│   └── judge.py          LLM-as-a-Judge (채점 모델 고정)
│
├── docs/                 아키텍처·DB·Tool·평가 가이드
├── experiments/          실험 결과 및 비교 리포트
├── logs/                 에이전트 실행 trace 로그
└── data/
    ├── audit_reports.db       SQLite (chunks, tags, metadata, fts_chunks)
    ├── vectorstore/chroma/    ChromaDB (KoE5, 1024dim)
    └── legacy/
        └── parsed_md/         파싱된 마크다운 + tables/ (CSV)
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 멀티 에이전트 | LangGraph |
| LLM | Ollama (`fredrezones55/qwen3.5-opus:9b-tooling` 기본) |
| 임베딩 | KoE5 (`nlpai-lab/KoE5`, 한국어 특화, 1024dim) |
| 벡터 DB | ChromaDB (cosine 유사도) |
| 정형 DB | SQLite + FTS5 (BM25 전문검색) |
| UI | Streamlit |

## 에이전트 흐름

```
사용자 질문
    ↓
Orchestrator  (의도 분류: simple_lookup / comparison / trend / calculation / general)
    ├─ simple_lookup ──────────────────────────────────┐
    └─ 그 외 → Query Rewriter (검색 쿼리 최대 3개 생성) ┘
                                                       ↓
                                    Retriever  (list_available_sections → 최적 tool 조합 결정)
                                      ├─ hybrid_search   (KoE5 벡터 + BM25 RRF)
                                      ├─ structured_query (SQL 정밀 검색)
                                      └─ csv_reader       ([TABLE_CSV] 자동 로드)
                                                       ↓
                                    Analyst   (답변 생성 + calculator / chart_generator)
                                      └─ NEED_MORE_SEARCH → Retriever 루프백 (최대 5회)
                                                       ↓
                                                   최종 답변
```

## 시작하기

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Ollama 모델 다운로드
ollama pull fredrezones55/qwen3.5-opus:9b-tooling

# 3. SQLite DB 구축 (JSONL → chunks/tags/metadata/FTS5)
python -m src.build_db

# 4. ChromaDB 벡터 인덱스 구축 (KoE5 임베딩)
python -m src.build_vectordb

# 5. UI 실행
streamlit run src/app.py
```

## CLI 단일 질문 테스트

```bash
python -u -m src.agents.graph "2024년 삼성전자 총자산은?"
```

## 평가 파이프라인

```bash
# 전체 실험 실행 (runs.yaml 기준)
python eval/run_eval.py

# 단일 실험
python eval/run_eval.py --run koe5_default

# 생성 모델 일괄 변경 (채점 모델은 judge.py에 고정)
python eval/run_eval.py --model qwen2.5:14b

# 특정 질문만
python eval/run_eval.py --questions q01,q03,q05

# 기존 결과 재채점
python eval/run_eval.py --judge-only experiments/koe5_default
```

## 멀티 머신 Ollama 설정

에이전트별로 다른 머신의 Ollama 서버를 사용하려면 `runtime.yaml`을 수정:

```yaml
# runtime.yaml
agent_backends:
  orchestrator:   http://192.168.1.10:11434   # GPU 서버 A
  query_rewriter: http://192.168.1.20:11434   # 경량 서버 B
  retriever:      http://192.168.1.10:11434
  analyst:        http://192.168.1.10:11434   # 가장 큰 모델 권장
```

또는 `agents.yaml`의 각 에이전트 `backend` 필드에 직접 URL 입력 (agents.yaml > runtime.yaml 우선).

## UI 모델 선택기

Streamlit 사이드바 → **에이전트별 모델 변경** expander에서 에이전트별 LLM 모델을 실시간으로 변경할 수 있다. **모델 적용** 버튼을 누르면 그래프가 재초기화된다.

선택 가능한 모델 목록은 `runtime.yaml`의 `llm.available_models`에 항목을 추가하면 자동 반영된다.
