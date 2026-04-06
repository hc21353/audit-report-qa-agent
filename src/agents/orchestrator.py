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

    최적화:
      - 규칙 기반 분석으로 충분한 경우(confidence >= HIGH_CONFIDENCE) LLM 호출 스킵
      - 복합 질문(서브 질문 분해 필요)일 때만 LLM 호출

    Args:
        state: 현재 그래프 상태
        llm:   orchestrator용 ChatOllama 인스턴스

    Returns:
        업데이트된 state 필드들
    """
    user_query = state["user_query"]
    print(f"[Orchestrator] ▶ INPUT: '{user_query}'", flush=True)
    t0 = time.time()

    # 1. 규칙 기반 사전 분석
    pre_analysis = _rule_based_analysis(user_query)
    confidence = pre_analysis.pop("confidence", "low")

    # 2. 신뢰도가 높으면 LLM 스킵
    if confidence == "high":
        result = {
            "intent": pre_analysis.get("intent", "general"),
            "extracted_years": pre_analysis.get("years", []),
            "extracted_sections": pre_analysis.get("sections", []),
            "sub_questions": [],
        }
        elapsed = time.time() - t0
        print(
            f"[Orchestrator] ◀ OUTPUT (rule-based skip): intent={result['intent']}, "
            f"years={result['extracted_years']} in {elapsed:.2f}s",
            flush=True,
        )
        return result

    # 3. LLM 기반 상세 분석 (복합/불명확 질문)
    prompt = f"""사용자 질문을 분석하여 다음 JSON을 반환하세요. JSON만 반환하고 다른 텍스트는 쓰지 마세요.

{{
  "intent": "simple_lookup | comparison | trend | calculation | general",
  "years": [추출된 연도 리스트],
  "sections": ["관련 섹션명 리스트"],
  "sub_questions": ["서브 질문 리스트 (복합 질문일 때만, 없으면 빈 리스트)"],
  "keywords": ["핵심 키워드"]
}}

사전 분석 힌트: {json.dumps(pre_analysis, ensure_ascii=False)}

사용자 질문: {user_query}"""

    parsed = {}
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
    elapsed = time.time() - t0
    print(
        f"[Orchestrator] ◀ OUTPUT (llm): intent={result['intent']}, "
        f"years={result['extracted_years']}, sections={result['extracted_sections']}, "
        f"sub_questions={result['sub_questions']} in {elapsed:.1f}s",
        flush=True,
    )
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
    """
    규칙 기반 분석 + 신뢰도 반환.

    confidence:
      "high"   → LLM 호출 스킵 가능 (단순/명확한 질문)
      "medium" → LLM 보조가 도움이 될 수 있음 (여전히 스킵 가능)
      "low"    → LLM 호출 필요 (복합 질문, 개념 설명 등)
    """
    result = {"intent": "general", "years": [], "keywords": [], "sections": [], "confidence": "low"}

    # 연도 추출 — "NNNN년부터 NNNN년까지" 범위 패턴 우선 처리
    range_match = re.search(r"(20[0-2]\d)\s*년?\s*(?:부터|~|-|–)\s*(20[0-2]\d)\s*년?", query)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        years = list(range(min(start, end), max(start, end) + 1))
    else:
        years = [int(y) for y in re.findall(r"20[0-2]\d", query)]
    result["years"] = sorted(set(years))

    # 비교 키워드
    comparison_kw = ["비교", "대비", "차이", "vs", "VS"]
    trend_kw = ["추이", "추세", "변동", r"최근 \d+년", "기간"]
    calc_kw = ["계산", "비율", "율", "ROE", "ROA", "ROI",
               "유동비율", "부채비율", "자기자본비율", "영업이익률"]
    section_kw = {
        "감사보고서": "독립된 감사인의 감사보고서",
        "핵심감사사항": "독립된 감사인의 감사보고서",
        "감사의견": "독립된 감사인의 감사보고서",
        "재무제표": "재무제표",
        "총자산": "재무제표",
        "자기자본": "재무제표",
        "유동자산": "재무제표",
        "부채": "재무제표",
        "주석": "주석",
        "내부회계": "내부회계관리제도 감사보고서",
        "외부감사": "외부감사 실시내용",
    }

    query_lower = query.lower()

    # 섹션 추론
    inferred_sections = []
    for kw, section in section_kw.items():
        if kw in query and section not in inferred_sections:
            inferred_sections.append(section)
    result["sections"] = inferred_sections

    # 주석 번호 추출 (예: "주석 3", "주석3번", "주석 30번")
    note_match = re.search(r"주석\s*(\d+)", query)
    if note_match:
        result["note_number"] = int(note_match.group(1))
        if "주석" not in result["sections"]:
            result["sections"].append("주석")

    # Intent 결정
    has_complex = any(kw in query for kw in ["그리고", "또한", "뿐만 아니라", "배경", "이유", "영향"])

    if any(kw in query for kw in comparison_kw) and len(years) >= 2:
        result["intent"] = "comparison"
        result["confidence"] = "high"

    elif re.search(r"추이|추세|최근\s*\d+년|전체\s*연도|모든\s*연도", query):
        result["intent"] = "trend"
        result["confidence"] = "high"

    elif any(kw in query_lower for kw in [k.lower() for k in calc_kw]):
        result["intent"] = "calculation"
        result["confidence"] = "high"

    elif len(years) == 1 and not has_complex and len(query) < 60:
        # 단순 단일 연도 조회 (짧은 질문 + 복잡성 없음)
        result["intent"] = "simple_lookup"
        result["confidence"] = "high"

    elif len(years) == 1 and not has_complex:
        # 단일 연도지만 설명 질문 가능
        result["intent"] = "simple_lookup"
        result["confidence"] = "medium"

    elif len(years) == 0 and not has_complex and any(kw in query for kw in section_kw):
        # 연도 없이 섹션 직접 언급
        result["intent"] = "general"
        result["confidence"] = "medium"

    else:
        # 복합 질문 → LLM 필요
        result["confidence"] = "low"

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
