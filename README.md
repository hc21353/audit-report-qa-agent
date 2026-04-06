# Samsung Audit Report QA Agent

삼성전자 감사보고서(2014~2024) **11년치**를 기반으로 한 **멀티 에이전트 Agentic RAG** 시스템.

3조 김병국, 김서영, 김유현, 김재희, 배민규, 조혜주

---

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

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 멀티 에이전트 | LangGraph 0.6.11 |
| LLM | Ollama (fredrezones55/qwen3.5-opus:9b-tooling 기본) |
| 임베딩 | KoE5 (nlpai-lab/KoE5, 한국어 특화, 1024dim) |
| 벡터 DB | ChromaDB 1.5.5 (cosine 유사도) |
| 정형 DB | SQLite 3 + FTS5 (BM25 전문검색) |
| UI | Streamlit 1.50.0 |
| 테스트 | pytest 8.4.2, pytest-cov |
| Python | 3.11 |

---

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
│   ├── parse_final.py    최종 파싱 및 청킹
│   ├── semantic_chunking.py 의미론적 청킹 유틸리티
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
├── tests/                pytest & 검증 스크립트 포함
│   ├── pytest/           pytest 테스트만
│   │   ├── test_parse.py 파싱 및 DB 연결 테스트
│   │   └── __init__.py
│   │
│   ├── validation/       검증/진단 스크립트
│   │   ├── check_chunks.py   청크 통계 계산
│   │   ├── check_db.py       DB 무결성 검증
│   │   ├── check_parse.py    파싱 함수 검증
│   │   ├── test_app.py       Streamlit 앱 테스트
│   │   ├── test_paths.py     경로 검증
│   │   ├── test_setup.py     설정 진단
│   │   ├── test_verify_paths.py 이동 후 경로 재검증
│   │   └── README.md         스크립트 사용 가이드
│   │
│   └── README.md         테스트 실행 가이드
│
├── docs/                 아키텍처·DB·Tool·평가 가이드
├── experiments/          실험 결과 및 비교 리포트
├── logs/                 에이전트 실행 trace 로그
├── parsed_data/          파싱된 JSONL 데이터 + 청크 파일
│
├── db/                   데이터베이스
│   ├── audit_reports.db       SQLite (13,154 chunks, 16.75MB)
│   ├── vectorstore/chroma/    ChromaDB (KoE5 1024dim, 13,154 벡터)
│   └── validation_log.jsonl   검증 로그
│
├── Dockerfile           Docker 이미지 정의 (Python 3.11)
├── docker-compose.yml   멀티 서비스 오케스트레이션
├── requirements.txt     Python 의존성
├── setup.sh            자동 초기화 스크립트
└── README.md           이 파일
```

---

## 설치 및 실행

### 시스템 요구사항

- **Python**: 3.11 이상
- **Ollama**: LLM 서버 (로컬 또는 원격) - **반드시 필수**
- **메모리**: 최소 16GB RAM (권장 32GB)
- **디스크**: 벡터스토어를 위해 ~10GB 여유

### ⚠️ Ollama 모델 설치 (필수)

이 프로젝트는 **Ollama의 다음 모델이 필요**합니다:

```bash
# 기본 모델 (현재 설정)
ollama pull fredrezones55/qwen3.5-opus:9b-tooling

# 또는 대체 모델들
ollama pull qwen2.5:14b    # 원래 권장
ollama pull mistral:7b      # 빠른 응답
ollama pull neural-chat:7b  # 더 나은 답변
```

**Ollama 설치 및 실행**:
```bash
# 1. Ollama 설치 (https://ollama.ai)

# 2. Ollama 서버 시작
ollama serve   # 기본 포트: http://localhost:11434

# 3. 모델 다운로드 및 테스트
ollama pull qwen2.5:14b
ollama run qwen2.5:14b
```

### 빠른 설치 (권장)

```bash
# 프로젝트 디렉토리로 이동
cd /Users/hyeju/audit-report-qa-agent

# setup.sh 실행 (자동 설정)
bash setup.sh
```

### 수동 설치

#### Step 1: Python 가상환경 생성
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows
```

#### Step 2: 패키지 설치
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 3: 데이터베이스 구축 (처음 한 번만)
```bash
python src/build_db.py        # SQLite 데이터베이스
python src/build_vectordb.py  # ChromaDB + KoE5 임베딩 (~30분)
```

### Streamlit UI 실행

#### 방법 1: 로컬 Streamlit (권장)

```bash
# 1. 가상환경 활성화
source .venv/bin/activate

# 2. Streamlit 서버 시작
streamlit run src/app.py
```

**출력:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

브라우저에서 `http://localhost:8501`을 열면 UI가 표시됩니다.

#### 방법 2: Docker Compose

```bash
cd /Users/hyeju/audit-report-qa-agent
docker-compose up --build
```

**접속:** `http://localhost:8501`

### 필요한 서버

#### Ollama LLM 서버 (필수)

```bash
# 1. Ollama 설치 및 실행
# macOS: https://ollama.ai → 다운로드 후 실행

# 2. 모델 다운로드 (선택사항, 처음 한 번만)
ollama pull qwen2.5:14b
ollama pull deepseek-r1:7b
ollama pull gemma3:4b

# 3. Ollama 서버 실행
ollama serve

# 4. 다른 터미널에서 모델 확인
ollama list
```

**기본 설정:**
- **Backend URL:** `http://localhost:11434`
- **설정 파일:** `config/runtime.yaml`

---

## 워크플로우

### 에이전트 실행 흐름

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
```


## 테스트 및 검증

### pytest 테스트

모든 pytest 테스트 파일은 `tests/pytest/` 폴더에 포함됨.

```bash
# 모든 테스트 실행
cd /Users/hyeju/audit-report-qa-agent
pytest tests/pytest/ -v

# 특정 테스트만 실행
pytest tests/pytest/test_parse.py -v
pytest tests/pytest/test_parse.py::test_chunk_load -v

# 커버리지 포함
pytest tests/pytest/ -v --cov=src --cov-report=html
```

#### 테스트 종류

| 파일 | 목적 | 명령어 |
|------|------|--------|
| `tests/pytest/test_parse.py` | 파싱/청킹/DB 연결 테스트 | `pytest tests/pytest/test_parse.py -v` |

### 검증 스크립트

검증 및 진단 스크립트는 `tests/validation/` 폴더에 포함됨.

```bash
# 청크 통계 확인
python tests/validation/check_chunks.py

# DB 무결성 검증
python tests/validation/check_db.py

# 설정 및 import 검증
python tests/validation/test_app.py

# 경로 검증
python tests/validation/test_paths.py

# 파싱 테스트
python tests/validation/check_parse.py
```

#### 검증 스크립트 종류

| 파일 | 목적 | 명령어 |
|------|------|--------|
| `tests/validation/check_chunks.py` | 청크 통계 계산 | `python tests/validation/check_chunks.py` |
| `tests/validation/check_db.py` | DB 무결성 검증 | `python tests/validation/check_db.py` |
| `tests/validation/check_parse.py` | 파싱 함수 검증 | `python tests/validation/check_parse.py` |
| `tests/validation/test_app.py` | 설정 및 import 검증 | `python tests/validation/test_app.py` |
| `tests/validation/test_paths.py` | 경로 검증 | `python tests/validation/test_paths.py` |
| `tests/validation/test_setup.py` | 설정 진단 | `python tests/validation/test_setup.py` |
| `tests/validation/test_verify_paths.py` | 이동 후 경로 재검증 | `python tests/validation/test_verify_paths.py` |

---

## CI/CD 설정

### GitHub Actions 워크플로우

`.github/workflows/ci.yml` 설정:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -r requirements.txt
    
    - name: Run setup diagnostics
      run: python tests/test_setup.py
    
    - name: Run pytest
      run: pytest tests/ -v --tb=short
    
    - name: Verify imports
      run: python -c "from src.config import load_config; print('✓ All imports successful')"
    
    - name: Check chunks
      run: python scripts/check_chunks.py
```

**주요 변경:**
- Python 버전: 3.11 (Dockerfile과 동일)
- pytest 실행: `pytest tests/ -v`
- 경로 검증 포함

---

## 문제 해결

### Q: Ollama 서버가 응답하지 않습니다

```bash
# 1. Ollama 프로세스 확인
ps aux | grep ollama

# 2. Ollama 다시 시작
ollama serve

# 3. 모델 다시 다운로드 (필요 시)
ollama pull qwen2.5:14b
```

### Q: "chromadb: No module named 'chromadb'" 에러

```bash
# 가상환경 활성화 후 재설치
source .venv/bin/activate
pip install chromadb==1.5.5
pip install -r requirements.txt
```

### Q: 벡터스토어가 비어있습니다

```bash
# 벡터스토어 재구축 (~30분 소요)
python src/build_vectordb.py

# 진행 상황 확인
tail -f logs/*.jsonl
```

### Q: 경로 관련 에러 (FileNotFoundError)

```bash
# 경로 검증 테스트 실행
pytest tests/test_verify_paths.py -v

# 현재 경로 확인
python -c "from pathlib import Path; print(Path.cwd())"

# 설정 파일 확인
python -c "from src.config import load_config; c = load_config(); print(c.database.path)"
```

### Q: Docker에서 Ollama 서버에 연결할 수 없습니다

```bash
# docker-compose.yml에서 환경 변수 확인
grep OLLAMA_BASE_URL docker-compose.yml

# macOS/Linux에서 host.docker.internal 사용
# Windows에서는 host.docker.internal도 가능
# 원격 서버: 실제 IP 주소 사용
```

### Q: 메모리 부족 에러

```bash
# 메모리 확인
free -h  # Linux
vm_stat  # macOS

# 메모리 절약 팁:
# 1. 한 번에 하나의 에이전트만 실행
# 2. --max-workers 줄이기 (설정에서)
# 3. 벡터스토어 캐시 크기 제한
```


