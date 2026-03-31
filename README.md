# Samsung Audit Report RAG System

삼성전자 감사보고서(2014~2024) 10년치를 기반으로 한 **멀티 에이전트 Agentic RAG** 시스템.

## 주요 기능

- **시계열 비교 QA**: "2018 vs 2022 부채 구조 차이" 같은 연도 비교 질문 지원
- **하이브리드 검색**: 벡터 유사도 + SQLite 구조 기반 검색
- **동적 CSV 로드**: 에이전트가 표 데이터를 필요할 때만 CSV에서 로드
- **재무비율 계산**: 유동비율, 부채비율, ROE 등 자동 계산
- **멀티 백엔드**: 에이전트별로 다른 머신의 Ollama 서버 연결 가능
- **실험 프레임워크**: 임베딩/청킹/LLM 조합별 성능 비교

## 프로젝트 구조

```
project/
├─ config/
│  ├─ app.yaml          파싱·청킹·임베딩·벡터스토어·UI
│  ├─ agents.yaml       에이전트별 모델·백엔드 URL·시스템 프롬프트
│  ├─ tools.yaml        Tool 스펙 정의
│  └─ runtime.yaml      기본 백엔드 + 에이전트별 오버라이드
│
├─ src/
│  ├─ config.py         YAML config 로더
│  ├─ parser.py         마크다운 파싱 + [TABLE_CSV] 인식
│  ├─ chunker.py        2레이어 청킹 (section_based + 비교 베이스라인)
│  ├─ csv_reader.py     CSV 읽기 유틸리티
│  ├─ db.py             SQLite (documents + metadata)
│  ├─ loader.py         적재 파이프라인
│  ├─ build_index.py    임베딩 + FAISS 인덱스 빌드
│  ├─ app.py            Streamlit 채팅 UI
│  │
│  ├─ tools/            에이전트 Tool (6개)
│  │  ├─ vector_search.py    벡터 유사도 검색
│  │  ├─ sql_query.py        구조 기반 검색
│  │  ├─ section_filter.py   섹션 필터링
│  │  ├─ csv_reader_tool.py  CSV 동적 로드
│  │  ├─ calculator.py       수식 계산 + 재무비율 프리셋
│  │  └─ chart_generator.py  Plotly 차트
│  │
│  └─ agents/           LangGraph 멀티 에이전트 (4노드)
│     ├─ state.py       GraphState 스키마
│     ├─ llm.py         에이전트별 백엔드 LLM 팩토리
│     ├─ orchestrator.py 질문 분석 + 라우팅
│     ├─ query_rewriter.py 검색 쿼리 최적화
│     ├─ retriever.py   하이브리드 검색 + CSV 자동 로드
│     ├─ analyst.py     답변 생성 + 추가 검색 판단
│     └─ graph.py       그래프 조립 + run_query()
│
├─ schema/              DDL + 설명
├─ eval/                평가 질문 + 실험 조합
├─ docs/                아키텍처·DB·Tool·평가 가이드
└─ data/
   ├─ raw_html/         원본 HTML
   ├─ parsed_md/        파싱된 마크다운 + tables/ (CSV)
   ├─ audit_reports.db  SQLite
   └─ vector_index/     FAISS 인덱스
```

## 기술 스택

| 구분 | 기술 |
|---|---|
| 멀티 에이전트 | LangGraph |
| LLM | Ollama (Qwen2.5, LLaMA3.1, Gemma3) |
| 임베딩 | multilingual-e5-large, ko-sroberta, bge-m3 |
| 벡터 DB | FAISS |
| 정형 DB | SQLite |
| UI | Streamlit |

## 아키텍처

```
사용자 질문
  ↓
Orchestrator (의도 분류 + 서브 질문 분해)
  ↓
Query Rewriter (검색 쿼리 최적화, 회계 용어 변환)
  ↓
Retriever (벡터 검색 + 구조 검색 + CSV 로드)
  ↓
Analyst (답변 생성 + 계산 + 차트)
  ↓ (추가 검색 필요 시 Retriever로 루프백)
최종 답변
```

에이전트별 다른 머신의 Ollama 연결 가능:

```yaml
# runtime.yaml
agent_backends:
  orchestrator: http://192.168.1.10:11434   # GPU 서버 A
  query_rewriter: http://192.168.1.20:11434 # 경량 서버 B
  analyst: http://192.168.1.10:11434        # GPU 서버 A
```

## 시작하기

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. Ollama 모델 다운로드
ollama pull qwen2.5:14b
ollama pull qwen2.5:7b

# 3. DB 초기화 + 데이터 적재
python -m src.loader

# 4. 벡터 인덱스 빌드
python -m src.build_index

# 5. 앱 실행
streamlit run src/app.py
```

## 단일 파일 적재

```bash
python -m src.loader --file data/parsed_md/감사보고서_2024.md
```

## 임베딩 모델 변경

```bash
# config/app.yaml의 active_model 변경
python -m src.build_index --embedding ko-sroberta --rebuild
```

## CLI 테스트

```bash
python -m src.agents.graph "2024년 삼성전자 총자산은?"
```
