"""
judge.py - LLM-as-a-Judge 평가 모듈

채점 모델은 JUDGE_MODEL 상수로 고정.
생성 모델(runs.yaml)이 바뀌어도 채점 기준이 일정하게 유지된다.

채점 기준 (각 1~5점):
  - relevance:    질문에 적절히 답하는가
  - accuracy:     사실관계 및 수치가 정확한가
  - completeness: 필요한 정보를 모두 포함하는가
  - specificity:  구체적 수치·출처·연도를 인용하는가
  - coherence:    논리적으로 잘 정리되어 있는가
"""

# ──────────────────────────────────────────────────────────────
# 채점 모델 고정: 여러 생성 모델을 비교할 때 공정한 기준을 위해
# 이 값은 run_eval.py의 runs.yaml과 무관하게 항상 동일하게 유지.
# ──────────────────────────────────────────────────────────────
JUDGE_MODEL = "fredrezones55/qwen3.5-opus:9b-tooling"

import json
import re
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


JUDGE_SYSTEM_PROMPT = """당신은 삼성전자 감사보고서 QA 시스템의 평가자입니다.
주어진 질문과 답변을 분석하고 JSON 형식으로 점수를 매기세요.
JSON만 반환하고 다른 텍스트는 절대 쓰지 마세요.
/no_think"""

JUDGE_PROMPT_TEMPLATE = """## 평가 대상

**질문**: {question}

**답변**:
{answer}

**참고 정보** (ground_truth, 있을 경우):
{ground_truth}

## 평가 기준 및 지침

다음 5가지 기준으로 1~5점을 부여하세요:

1. **관련성 (relevance)**: 답변이 질문의 핵심을 다루는가?
   - 5: 질문 완전히 답변 | 3: 부분적으로 답변 | 1: 질문과 무관

2. **정확성 (accuracy)**: 수치, 연도, 사실관계가 맞는가?
   - 5: 모든 수치 정확 | 3: 일부 수치 오류 | 1: 대부분 오류 또는 정보 없음
   - ground_truth가 제공되면 반드시 대조하세요

3. **완성도 (completeness)**: 필요한 정보를 충분히 포함하는가?
   - 5: 충분한 설명 | 3: 핵심만 있음 | 1: 너무 단편적

4. **구체성 (specificity)**: 구체적 수치·출처·연도를 인용하는가?
   - 5: 백만원 단위, 연도, 섹션 등 구체 인용 | 3: 일부만 | 1: 추상적

5. **논리성 (coherence)**: 논리적으로 잘 구성되어 있는가?
   - 5: 매우 명확하고 체계적 | 3: 이해 가능 | 1: 혼란스러움

## 출력 형식

{{
  "scores": {{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "completeness": <1-5>,
    "specificity": <1-5>,
    "coherence": <1-5>
  }},
  "total": <5가지 평균, 소수점 2자리>,
  "verdict": "excellent|good|fair|poor",
  "strengths": ["잘 된 점 1가지"],
  "weaknesses": ["개선할 점 1가지"],
  "brief_reason": "총평 1~2문장"
}}"""


class LLMJudge:
    """LLM-as-a-Judge 평가 클래스"""

    def __init__(self, config):
        """
        Args:
            config: Config 객체 (agents.yaml에서 현재 모델 읽기)
        """
        self.config = config
        self.llm = self._create_judge_llm()
        self._call_count = 0
        self._total_time = 0.0

    def _create_judge_llm(self) -> ChatOllama:
        """
        채점용 LLM 생성.
        모델은 JUDGE_MODEL 상수로 고정 — config/agents.yaml과 무관.
        생성 모델을 바꿔 실험해도 채점 기준이 달라지지 않는다.
        """
        runtime_cfg = self.config.runtime
        backend = runtime_cfg.get("llm", {}).get("default_backend", "http://localhost:11434")
        timeout = runtime_cfg.get("llm", {}).get("timeout", 300)

        print(f"[Judge] LLM: model={JUDGE_MODEL} (고정), backend={backend}")

        return ChatOllama(
            base_url=backend,
            model=JUDGE_MODEL,
            num_ctx=8192,
            num_predict=512,
            temperature=0.0,
            timeout=timeout,
            reasoning=False,
        )

    def evaluate(self, question: str, answer: str, ground_truth: str = "") -> dict:
        """
        단일 답변 평가.

        Returns:
            {scores: {...}, total: float, verdict: str, ...}
        """
        if not answer or len(answer.strip()) < 10:
            return self._empty_score("답변이 너무 짧거나 비어 있습니다")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answer=answer[:3000],  # 너무 긴 답변은 앞부분만
            ground_truth=ground_truth if ground_truth else "제공되지 않음",
        )

        t0 = time.time()
        try:
            response = self.llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            raw = response.content
        except Exception as e:
            return self._empty_score(f"Judge LLM 오류: {e}")

        elapsed = time.time() - t0
        self._call_count += 1
        self._total_time += elapsed

        parsed = self._parse_judge_response(raw)
        parsed["judge_elapsed_s"] = round(elapsed, 1)
        return parsed

    def evaluate_batch(self, items: list[dict], verbose: bool = True) -> list[dict]:
        """
        여러 답변을 순차적으로 평가.

        Args:
            items: [{"question": ..., "answer": ..., "ground_truth"?: ...}]
            verbose: 진행 상황 출력

        Returns:
            items에 "judge_result" 필드를 추가한 리스트
        """
        results = []
        for i, item in enumerate(items, 1):
            if verbose:
                q_preview = item.get("question", "")[:50]
                print(f"  [{i}/{len(items)}] 평가 중: {q_preview}...")

            judge_result = self.evaluate(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                ground_truth=item.get("ground_truth", ""),
            )
            result = dict(item)
            result["judge_result"] = judge_result

            if verbose:
                total = judge_result.get("total", 0)
                verdict = judge_result.get("verdict", "?")
                elapsed = judge_result.get("judge_elapsed_s", 0)
                print(f"    → {verdict} ({total:.2f}/5.00) in {elapsed}s")

            results.append(result)

        return results

    def _parse_judge_response(self, text: str) -> dict:
        """LLM 응답에서 평가 JSON 추출"""
        # ```json 블록
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 직접 파싱
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
            # total 재계산 (LLM이 잘못 계산할 수 있음)
            scores = parsed.get("scores", {})
            if scores:
                vals = [v for v in scores.values() if isinstance(v, (int, float))]
                if vals:
                    parsed["total"] = round(sum(vals) / len(vals), 2)
            return parsed
        except (ValueError, json.JSONDecodeError):
            return self._empty_score("JSON 파싱 실패")

    def _empty_score(self, reason: str) -> dict:
        return {
            "scores": {
                "relevance": 0,
                "accuracy": 0,
                "completeness": 0,
                "specificity": 0,
                "coherence": 0,
            },
            "total": 0.0,
            "verdict": "error",
            "strengths": [],
            "weaknesses": [reason],
            "brief_reason": reason,
        }

    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_time_s": round(self._total_time, 1),
            "avg_time_s": round(self._total_time / max(self._call_count, 1), 1),
        }


def aggregate_scores(evaluated_items: list[dict]) -> dict:
    """평가 결과 통계 집계"""
    all_scores = {
        "relevance": [], "accuracy": [], "completeness": [],
        "specificity": [], "coherence": [], "total": [],
    }

    verdict_counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "error": 0}

    for item in evaluated_items:
        jr = item.get("judge_result", {})
        scores = jr.get("scores", {})
        for key in all_scores:
            if key == "total":
                val = jr.get("total")
            else:
                val = scores.get(key)
            if isinstance(val, (int, float)):
                all_scores[key].append(val)
        verdict = jr.get("verdict", "error")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    agg = {}
    for key, vals in all_scores.items():
        if vals:
            agg[key] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min": min(vals),
                "max": max(vals),
            }

    return {
        "score_stats": agg,
        "verdict_distribution": verdict_counts,
        "evaluated_count": len(evaluated_items),
    }
