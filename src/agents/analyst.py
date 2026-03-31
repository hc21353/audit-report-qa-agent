"""
analyst.py - 검색 결과를 기반으로 최종 답변을 생성하는 에이전트 노드

역할:
  - 검색 결과 컨텍스트 구성
  - [TABLE_CSV] 데이터 삽입
  - LLM으로 답변 생성
  - 추가 검색 필요 여부 판단
"""

import json
import time
from langchain_core.messages import SystemMessage, HumanMessage


def analyst_node(state: dict, llm, tools: dict = None, system_prompt: str = "") -> dict:
    """
    LangGraph 노드: 최종 답변 생성.

    Args:
        state: 현재 그래프 상태
        llm:   analyst용 ChatOllama 인스턴스
        tools: {"calculator": fn, "chart_generator": fn}

    Returns:
        answer, sources, needs_more_search, iteration 업데이트
    """
    print(f"[Analyst] Start", flush=True)
    t0 = time.time()
    search_results = state.get("search_results", [])
    csv_data = state.get("csv_data", {})
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    iteration = state.get("iteration", 0)

    if not search_results:
        return {
            "answer": "검색 결과가 없습니다. 질문을 다시 표현해주세요.",
            "needs_more_search": False,
            "iteration": iteration + 1,
        }

    # 1. 컨텍스트 구성
    context = _build_context(search_results, csv_data)
    sources = _extract_sources(search_results)

    # 2. LLM 답변 생성
    prompt = f"""다음 검색 결과를 기반으로 사용자 질문에 답변하세요.

## 사용자 질문
{user_query}

## 질문 의도
{intent}

## 검색 결과
{context}

## 지시사항
- 숫자를 인용할 때는 출처(연도, 섹션)를 명시하세요.
- 정보가 부족하면 마지막에 "NEED_MORE_SEARCH: <추가로 찾아야 할 내용>" 을 추가하세요.
- 계산이 필요하면 "CALC: <수식>" 형태로 표시하세요.
- 금액 단위는 백만원입니다.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        raw_answer = response.content
    except Exception as e:
        return {
            "answer": f"답변 생성 중 오류: {e}",
            "needs_more_search": False,
            "iteration": iteration + 1,
            "errors": state.get("errors", []) + [f"Analyst LLM error: {e}"],
        }

    # 3. 후처리: CALC 태그 처리
    answer, calculations = _process_calculations(raw_answer, tools)

    # 4. 추가 검색 필요 여부 판단
    needs_more, additional_query = _check_needs_more(answer)

    # 최대 반복 체크
    if iteration >= state.get("max_iterations", 5) - 1:
        needs_more = False

    print(f"[Analyst] Done in {time.time()-t0:.1f}s, needs_more={needs_more}", flush=True)
    return {
        "answer": answer,
        "sources": sources,
        "calculations": state.get("calculations", []) + calculations,
        "needs_more_search": needs_more,
        "additional_query": additional_query,
        "iteration": iteration + 1,
    }


def analyst_router(state: dict) -> str:
    """
    LangGraph 조건부 엣지: 추가 검색 여부에 따라 라우팅.
    """
    if state.get("needs_more_search") and state.get("iteration", 0) < state.get("max_iterations", 5):
        return "need_more_search"
    return "done"


# ─── 헬퍼 함수 ──────────────────────────────────────────────

def _build_context(search_results: list[dict], csv_data: dict) -> str:
    """검색 결과를 LLM 프롬프트용 컨텍스트로 구성"""
    parts = []
    for i, r in enumerate(search_results[:10]):  # 상위 10개만
        content = r.get("content", "")

        # [TABLE_CSV] 태그를 실제 데이터로 치환
        for ref in r.get("csv_refs", []):
            if ref in csv_data:
                tag = f"[TABLE_CSV] {ref}"
                content = content.replace(tag, csv_data[ref])

        year = r.get("year", "?")
        path = r.get("section_path", "?")
        search_type = r.get("search_type", "?")

        parts.append(
            f"### 출처 {i+1} (연도: {year}, 섹션: {path}, 검색: {search_type})\n{content}"
        )

    return "\n\n".join(parts)


def _extract_sources(search_results: list[dict]) -> list[dict]:
    """인용 소스 목록 생성"""
    sources = []
    seen = set()
    for r in search_results:
        key = (r.get("year"), r.get("section_path"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "year": r.get("year"),
                "section_path": r.get("section_path"),
                "search_type": r.get("search_type"),
            })
    return sources


def _process_calculations(answer: str, tools: dict) -> tuple[str, list]:
    """CALC: 태그를 찾아 calculator tool 호출"""
    calculations = []
    if not tools or "calculator" not in tools:
        return answer, calculations

    import re
    calc_pattern = re.compile(r"CALC:\s*(.+?)(?:\n|$)")
    matches = calc_pattern.findall(answer)

    for expr in matches:
        try:
            result_json = tools["calculator"].invoke({
                "expression": expr.strip(),
                "label": "",
            })
            result = json.loads(result_json)
            if result.get("success"):
                answer = answer.replace(
                    f"CALC: {expr}",
                    f"**{result['result']}** (계산: {expr.strip()})"
                )
                calculations.append(result)
        except Exception:
            pass

    return answer, calculations


def _check_needs_more(answer: str) -> tuple[bool, str]:
    """NEED_MORE_SEARCH 태그 확인"""
    import re
    match = re.search(r"NEED_MORE_SEARCH:\s*(.+?)(?:\n|$)", answer)
    if match:
        query = match.group(1).strip()
        # 태그 제거
        clean_answer = answer.replace(match.group(0), "").strip()
        return True, query
    return False, ""
