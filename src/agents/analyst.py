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
    t0 = time.time()
    search_results = state.get("search_results", [])
    csv_data = state.get("csv_data", {})
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    iteration = state.get("iteration", 0)

    print(
        f"[Analyst] ▶ INPUT: query='{user_query}', intent={intent}, "
        f"search_results={len(search_results)}, csv_data={len(csv_data)} files, iteration={iteration}",
        flush=True,
    )

    if not search_results:
        return {
            "answer": "검색 결과가 없습니다. 질문을 다시 표현해주세요.",
            "needs_more_search": False,
            "iteration": iteration + 1,
        }

    # 1. 컨텍스트 구성
    has_web_results = any(r.get("is_web_source") for r in search_results)
    context = _build_context(search_results, csv_data)
    sources = _extract_sources(search_results)

    # 2. LLM 답변 생성
    web_notice = (
        "⚠️ 아래 검색 결과 중 [WEB] 출처가 있습니다. 해당 내용 인용 시 소스 불확실 경고를 답변 말미에 표시하세요."
        if has_web_results else
        "이번 검색 결과는 모두 감사보고서 DB에서 조회되었습니다."
    )
    prompt = f"""다음 검색 결과를 기반으로 사용자 질문에 정확하고 구조화된 답변을 생성하세요.

## 사용자 질문
{user_query}

## 질문 의도
{intent}

## 검색 결과
{context}

## 답변 형식 규칙
- 모든 금액은 백만원 단위이며 단위를 항상 명시하세요 (예: 324,966,127백만원)
- 숫자를 인용할 때는 반드시 출처를 괄호로 표기하세요 (예: (2024년, 재무제표))
- 비교/추이 질문에는 마크다운 표를 사용하세요:
  | 연도 | 항목명 | 금액(백만원) | 전년대비 변화 |
  |------|--------|------------|-------------|
- 비율/지표 계산 시 수식과 결과를 함께 표시하세요:
  예: 유동비율 = 유동자산 / 유동부채 × 100 = CALC: <유동자산값> / <유동부채값> * 100

## 웹 검색 결과 처리
{web_notice}

## 계산 처리
- 재무비율 계산이 필요하면 "CALC: <수식>" 형태로 표시하세요 (숫자만 사용)
- 예: CALC: 195419834 / 97522203 * 100

## NEED_MORE_SEARCH 사용 기준 (엄격히 준수)
다음 경우에만 답변 맨 끝에 "NEED_MORE_SEARCH: <찾아야 할 내용>" 을 추가하세요:
✅ 질문에서 명시 요구한 수치/섹션이 검색 결과에 전혀 없을 때
✅ 비교 질문인데 한 연도의 데이터만 있을 때
❌ 검색 결과에 관련 정보가 일부라도 있을 때는 사용하지 마세요
❌ 일반 개념 설명 질문에는 사용하지 마세요
❌ 현재 iteration이 2 이상이면 사용하지 마세요 (현재: {iteration})
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
    needs_more, additional_query, answer = _check_needs_more(answer)

    # 최대 반복 체크 (코드 레벨 강제 차단 — 프롬프트 지시 무시 방지)
    max_iterations = state.get("max_iterations", 5)
    if iteration >= min(2, max_iterations - 1):  # 최대 2회 추가 검색
        needs_more = False

    elapsed = time.time() - t0
    print(
        f"[Analyst] ◀ OUTPUT: needs_more={needs_more}, "
        f"sources={len(sources)}, calcs={len(calculations)}, "
        f"answer_len={len(answer)}chars in {elapsed:.1f}s",
        flush=True,
    )
    print(f"[Analyst] Answer preview: {answer[:200]}{'...' if len(answer)>200 else ''}", flush=True)
    if needs_more:
        print(f"[Analyst] Additional query: '{additional_query}'", flush=True)
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
    for i, r in enumerate(search_results[:30]):  # 상위 30개 (additional_search 40개 대응)
        content = r.get("content", "")

        # [TABLE_CSV] 태그를 실제 데이터로 치환
        for ref in r.get("csv_refs", []):
            if ref in csv_data:
                tag = f"[TABLE_CSV] {ref}"
                content = content.replace(tag, csv_data[ref])

        year = r.get("fiscal_year") or r.get("year", "?")
        path = r.get("section_path", "?")
        search_type = r.get("search_type", "?")
        source_label = "[WEB] " if r.get("is_web_source") else ""

        parts.append(
            f"### {source_label}출처 {i+1} (연도: {year}, 섹션: {path}, 검색: {search_type})\n{content}"
        )

    return "\n\n".join(parts)


def _extract_sources(search_results: list[dict]) -> list[dict]:
    """인용 소스 목록 생성"""
    sources = []
    seen = set()
    for r in search_results:
        year = r.get("fiscal_year") or r.get("year")
        key = (year, r.get("section_path"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "year": year,
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


def _check_needs_more(answer: str) -> tuple[bool, str, str]:
    """NEED_MORE_SEARCH 태그 확인. (needs_more, additional_query, clean_answer) 반환"""
    import re
    match = re.search(r"NEED_MORE_SEARCH:\s*(.+?)(?:\n|$)", answer)
    if match:
        query = match.group(1).strip()
        clean_answer = answer.replace(match.group(0), "").strip()
        return True, query, clean_answer
    return False, "", answer
