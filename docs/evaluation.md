# 실험 및 평가

## 개요

임베딩 모델, 청킹 전략, LLM 모델 조합별 성능을 비교하여 최적 구성을 찾는다.

## 평가 차원

### 1. 청킹 전략 비교

같은 임베딩/LLM에서 청킹만 변경하여 비교:

| 실험 | 청킹 | 기대 |
|---|---|---|
| baseline_section | 구조 기반 | 섹션 정밀도 높음, 맥락 보존 |
| baseline_fixed | 512자 고정 | 균일한 크기, 맥락 손실 가능 |
| baseline_recursive | 구분자 기반 | 중간 수준 |

**핵심 비교 포인트**: "주석 30번 특수관계자 거래" 같은 구조적 질문에서 section_based가 얼마나 우세한가

### 2. 임베딩 모델 비교

같은 청킹(section_based)/LLM에서 임베딩만 변경:

| 모델 | 특성 |
|---|---|
| multilingual-e5-large | 다국어 범용, 1024차원 |
| ko-sroberta | 한국어 특화, 768차원 |
| bge-m3 | 긴 문맥(8192 토큰), 1024차원 |
| ollama-nomic | 로컬 경량, 768차원 |

### 3. LLM 모델 비교

같은 검색 파이프라인에서 LLM만 변경:
- qwen2.5:14b vs qwen2.5:7b (크기 차이)
- qwen2.5 vs llama3.1 vs gemma3 (계열 차이)

## 평가 지표

### 검색 품질 (Retrieval)
- **Hit Rate**: 정답 청크가 top-k에 포함되는 비율
- **MRR** (Mean Reciprocal Rank): 정답 청크의 순위 역수 평균
- **Precision@k**: top-k 중 관련 청크 비율

### 답변 품질 (Generation)
- **정확도**: ground_truth 대비 수치 일치 (수치 질문)
- **완성도**: 질문의 모든 측면을 답변했는가 (서술 질문)
- **Tool 선택 정확도**: expected_tool과 실제 사용 tool 일치 여부

## 실행 방법

```bash
# 단일 실험
python run_experiment.py --run baseline_section

# 전체 실험
python run_experiment.py --all

# 결과 비교
python compare_results.py --output ./experiments/comparison.md
```
