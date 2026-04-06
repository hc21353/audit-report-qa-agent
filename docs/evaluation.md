# 실험 및 평가

## 개요

임베딩 모델, 청킹 전략, LLM 생성 모델 조합별 성능을 비교하여 최적 구성을 찾는다.

**핵심 원칙**: 생성 모델과 채점 모델은 분리되어 있다.
- **생성 모델**: `runs.yaml`의 `model`/`models` 필드 또는 `--model` CLI 인자로 변경
- **채점 모델**: `eval/judge.py`의 `JUDGE_MODEL` 상수로 고정 → 비교 실험 간 공정한 기준 보장

---

## 실행 방법

```bash
# 전체 실험 (runs.yaml의 모든 run)
python eval/run_eval.py

# 단일 실험
python eval/run_eval.py --run koe5_default

# 생성 모델 일괄 오버라이드 (채점 모델은 고정)
python eval/run_eval.py --model qwen2.5:14b
python eval/run_eval.py --model qwen2.5:7b

# 두 모델 비교 (runs.yaml에 두 run 정의 후)
python eval/run_eval.py --run koe5_qwen2.5_14b,koe5_qwen2.5_7b

# 특정 질문만 실행
python eval/run_eval.py --questions q01,q03,q05

# Judge 없이 답변만 수집
python eval/run_eval.py --skip-judge

# 기존 결과 재채점 (생성은 건너뛰고 채점만)
python eval/run_eval.py --judge-only experiments/koe5_default
```

---

## 평가 차원

### 1. LLM 생성 모델 비교

같은 검색 파이프라인에서 생성 모델만 변경:

```yaml
# runs.yaml 예시
- name: koe5_qwen3.5
  vector_store: chroma
  # model 미지정 → agents.yaml 기본값 사용

- name: koe5_qwen2.5_14b
  vector_store: chroma
  model: qwen2.5:14b           # 전체 에이전트 동일 모델

- name: koe5_mixed
  vector_store: chroma
  models:                       # 에이전트별 개별 지정
    orchestrator:   qwen2.5:7b
    query_rewriter: qwen2.5:7b
    retriever:      qwen2.5:14b  # 검색 계획은 큰 모델
    analyst:        qwen2.5:14b  # 답변 생성은 큰 모델
```

**핵심 비교 포인트**: 에이전트별 모델 혼합(koe5_mixed)이 전체 14b보다 비용 효율적인가?

### 2. 청킹 전략 비교

같은 임베딩/LLM에서 청킹만 변경:

| 전략 | 방식 | 기대 효과 |
|------|------|----------|
| `section_based` | 헤딩 기반 구조 분리 | 섹션 정밀도 높음, 맥락 보존 |
| `fixed_size` | 512자 고정 | 균일한 크기, 맥락 손실 가능 |
| `recursive` | 구분자 기반 재귀 분할 | 중간 수준 |

**핵심 비교 포인트**: "주석 30번 특수관계자 거래" 같은 구조적 질문에서 `section_based`의 우위

### 3. 임베딩 모델 비교

같은 청킹/LLM에서 임베딩만 변경:

| 모델 | 특성 |
|------|------|
| KoE5 (`nlpai-lab/KoE5`) | **현재 사용** — 한국어 특화 E5, Recall@1 최고 |
| KURE-v1 | Recall@3/5 우수, 후보 다수 활용 시 |
| `bge-m3` | 긴 문맥(8192 토큰) 지원, 1024dim |
| `ko-sroberta` | 경량, 768dim |

---

## 채점 기준 (LLM-as-a-Judge)

채점 모델: `eval/judge.py`의 `JUDGE_MODEL` 상수 (변경 시 파일 직접 수정)

각 항목 1~5점:

| 항목 | 기준 |
|------|------|
| **관련성** (relevance) | 질문의 핵심을 다루는가 |
| **정확성** (accuracy) | 수치·연도·사실관계가 맞는가 (ground_truth 대조) |
| **완성도** (completeness) | 필요한 정보를 충분히 포함하는가 |
| **구체성** (specificity) | 백만원 단위·연도·섹션 등 구체 인용하는가 |
| **논리성** (coherence) | 논리적으로 잘 구성되어 있는가 |

**verdict 기준**
- `excellent`: 평균 ≥ 4.5
- `good`: 평균 ≥ 3.5
- `fair`: 평균 ≥ 2.5
- `poor`: 평균 < 2.5

---

## 결과 리포트

`python eval/run_eval.py`를 실행하면 `experiments/comparison_<timestamp>.md`가 자동 생성된다.

**리포트 구성**
1. **실험 요약 표**: 실험명 / 생성 모델 / 총점 / 관련성 / 정확성 / 완성도 / 구체성 / 평균 응답시간 / Excellent+Good 비율
2. **질문 유형별 비교**: simple_lookup / section_lookup / calculation / trend 등 유형별 점수
3. **질문별 상세 비교**: 각 질문에 대한 실험별 답변 요약과 채점 결과 (collapsible)

---

## 평가 질문 세트 (`eval/questions.yaml`)

총 질문 수 및 유형:

| 유형 | 설명 | 예시 |
|------|------|------|
| `simple_lookup` | 특정 연도·단일 항목 조회 | "2024년 총자산은?" |
| `section_lookup` | 특정 섹션 내용 조회 | "2024년 핵심감사사항은?" |
| `calculation` | 재무비율 계산 | "2024년 유동비율을 계산해줘" |
| `year_comparison` | 연도 간 비교 | "2022 vs 2024 부채 구조 비교" |
| `trend_analysis` | 기간 추이 분석 | "최근 5년 총자산 변화" |
| `multi_hop` | 복합 질문 | "재고자산 금액 비교 + 평가기준 설명" |

각 질문은 `ground_truth` 필드를 포함하며, accuracy 채점 시 대조에 사용된다.
