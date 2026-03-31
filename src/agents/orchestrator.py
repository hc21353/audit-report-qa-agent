"""
orchestrator.py - 사용자 질문을 분석하고 라우팅하는 에이전트 노드

역할:
  - 질문 의도 분류 (simple_lookup / comparison / trend / calculation / general)
  - 연도, 섹션, 키워드 추출
  - 복합 질문 분해
"""

import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage


def orchestrator_node(state: dict, llm, system_prompt: str = "") -> dict:
    """
    LangGraph 노드: 사용자 질문을 분석.

    Args:
        state: 현재 그래프 상태
        llm:   orchestrator용 ChatOllama 인스턴스

    Returns:
        업데이트된 state 필드들
    """
    user_query = state["user_query"]
    print(f"[Orchestrator] Start: {user_query[:50]}", flush=True)
    t0 = time.time()

    # 1. 규칙 기반 사전 분석 (LLM 호출 전 빠른 판별)
    pre_analysis = _rule_based_analysis(user_query)

    # 2. LLM 기반 상세 분석
    prompt = f"""사용자 질문을 분석하여 다음 JSON을 반환하세요. JSON만 반환하고 다른 텍스트는 쓰지 마세요.

{{
  "intent": "simple_lookup | comparison | trend | calculation | general",
  "years": [추출된 연도 리스트],
  "sections": ["관련 섹션명 리스트"],
  "sub_questions": ["서브 질문 리스트 (복합 질문일 때만)"],
  "keywords": ["핵심 키워드"]
}}

사전 분석 힌트: {json.dumps(pre_analysis, ensure_ascii=False)}

사용자 질문: {user_query}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])

        parsed = _parse_json_response(response.content)
    except Exception as e:
        parsed = pre_analysis
        state.setdefault("errors", []).append(f"Orchestrator LLM error: {e}")

    result = {
        "intent": parsed.get("intent", pre_analysis.get("intent", "general")),
        "extracted_years": parsed.get("years", pre_analysis.get("years", [])),
        "extracted_sections": parsed.get("sections", []),
        "sub_questions": parsed.get("sub_questions", []),
    }
    print(f"[Orchestrator] Done in {time.time()-t0:.1f}s, intent={result['intent']}", flush=True)
    return result


def orchestrator_router(state: dict) -> str:
    """
    LangGraph 조건부 엣지: intent에 따라 다음 노드 결정.

    Returns:
        다음 노드 이름 ("retriever" 또는 "query_rewriter")
    """
    intent = state.get("intent", "general")

    if intent == "simple_lookup":
        return "retriever"
    else:
        return "query_rewriter"


# ─── 규칙 기반 사전 분석 ─────────────────────────────────────

def _rule_based_analysis(query: str) -> dict:
    """LLM 없이 규칙 기반으로 빠르게 분석"""
    result = {"intent": "general", "years": [], "keywords": []}

    # 연도 추출
    years = [int(y) for y in re.findall(r"20[0-2]\d", query)]
    result["years"] = sorted(set(years))

    # 비교 키워드
    comparison_kw = ["비교", "대비", "차이", "vs", "VS", "변화"]
    trend_kw = ["추이", "추세", "변동", "최근", "기간"]
    calc_kw = ["계산", "비율", "율", "ROE", "ROA", "유동비율", "부채비율"]

    query_lower = query.lower()

    if any(kw in query for kw in comparison_kw) and len(years) >= 2:
        result["intent"] = "comparison"
    elif any(kw in query for kw in trend_kw):
        result["intent"] = "trend"
    elif any(kw in query_lower for kw in calc_kw):
        result["intent"] = "calculation"
    elif len(years) == 1:
        result["intent"] = "simple_lookup"

    return result


def _parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    # ```json 블록 안의 JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 원문에서 직접 JSON 파싱 시도
    try:
        # 첫 번째 { ~ 마지막 } 추출
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}
