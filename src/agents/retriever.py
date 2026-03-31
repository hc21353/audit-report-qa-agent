"""
retriever.py - LLM 기반 자율 검색 에이전트 노드

흐름:
  1. rewritten_queries에서 structured_params가 있는 쿼리는 직접 실행 (LLM 우회)
  2. semantic 쿼리는 LLM에게 검색 계획 수립 요청
  3. 결과 부족 시 LLM이 추가 검색 계획 생성 (Round 2)
  4. 여전히 결과 없으면 BM25 폴백
  5. [TABLE_CSV] 태그 감지 시 csv_reader 자동 호출
"""

import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage
from src.csv_reader import extract_csv_refs


# ─── Tool 설명 (LLM에게 제공) ───────────────────────────────

TOOL_DESCRIPTIONS = """사용 가능한 검색 도구:

1. hybrid_search
   - 설명: 벡터 유사도 + BM25 키워드를 결합한 하이브리드 검색. 일반적인 질문에 적합.
   - 파라미터:
     query (str, 필수): 검색 쿼리
     top_k (int, 기본 10): 결과 수
     year (int, 기본 0): 연도 필터 (0=전체)
     section_h2 (str): 대분류 필터 ("재무제표"|"주석"|"독립된 감사인의 감사보고서"|"내부회계관리제도 감사보고서"|"외부감사 실시내용")
     mode (str): "hybrid"|"vector"|"bm25"

2. structured_query
   - 설명: DB의 h1~h6 구조를 활용한 정밀 검색. 특정 연도/섹션의 정확한 문서를 찾을 때.
   - 파라미터:
     years (str): 쉼표 구분 연도 (예: "2022,2023,2024")
     section_h2 (str): 대분류 (정확 매칭)
     section_h3 (str): 주석 번호 (부분 매칭, 예: "3.")
     section_h4 (str): 세부 항목
     section_h5 (str): 세세부 항목
     keyword (str): 본문 키워드
     content_type (str): "text"|"table"|"table_csv"|"mixed"|"all"
     limit (int, 기본 30)

3. list_available_sections
   - 설명: DB에 어떤 섹션이 있는지 목록 조회. 검색 전 구조 파악용.
   - 파라미터:
     year (int, 기본 0): 연도
     level (int, 기본 3): 헤딩 레벨 (2=대분류, 3=주석번호)

참고:
- 재무 수치(영업이익, 총자산 등)는 section_h2="재무제표"에 [TABLE_CSV] 태그로 저장됨
- 주석 상세 내용은 section_h2="주석"에 있음
- 감사의견, 핵심감사사항은 section_h2="독립된 감사인의 감사보고서"에 있음
- 여러 연도 비교 시 structured_query의 years에 쉼표로 나열
- section_h3 검색은 부분 매칭 → "3." 으로 "3. 중요한 회계추정 및 가정" 검색 가능
- 벡터 인덱스가 없을 경우 hybrid_search의 mode="bm25"를 사용할 것"""


PLANNING_PROMPT = """당신은 감사보고서 검색 전문가입니다.
사용자 질문과 의도를 분석하여 최적의 검색 계획을 JSON으로 작성하세요.

{tool_descriptions}

{db_context}

## 규칙
- 검색 계획은 1~3개의 tool 호출로 구성하세요.
- section_h2/h3는 반드시 위 "DB 실제 섹션 구조"에 있는 값만 사용하세요.
- 재무 수치 질문은 structured_query로 재무제표 섹션을 먼저 찾으세요.
- 잘 모르겠으면 hybrid_search를 사용하세요.
- JSON만 반환하고 다른 텍스트는 쓰지 마세요.

## 출력 형식
{{
  "reasoning": "검색 전략에 대한 간단한 설명",
  "steps": [
    {{"tool": "도구명", "params": {{...}} }},
    ...
  ]
}}"""


REFINE_PROMPT = """검색 결과가 부족합니다. 추가 검색이 필요합니다.

## 사용자 질문
{query}

## 현재까지 검색 결과 요약
{results_summary}

## 부족한 정보
{missing_info}

{db_context}

사용 가능한 tool: hybrid_search, structured_query, list_available_sections
추가 검색 계획을 1~2개의 tool 호출로 작성하세요. JSON만 반환하세요.
{{
  "reasoning": "추가 검색 이유",
  "steps": [
    {{"tool": "hybrid_search 또는 structured_query", "params": {{...}} }}
  ]
}}"""


# ─── 메인 노드 ──────────────────────────────────────────────

def retriever_node(state: dict, tools: dict, llm=None, db_context: str = "") -> dict:
    """
    LangGraph 노드: LLM이 검색 계획을 수립하고 tool을 실행.

    Args:
        state:      현재 그래프 상태
        tools:      {"hybrid_search", "structured_query", "list_available_sections", "csv_reader"}
        llm:        retriever용 ChatOllama 인스턴스
        db_context: 그래프 초기화 시 생성된 DB 섹션 구조 문자열

    Returns:
        search_results, csv_data 업데이트
    """
    t0 = time.time()
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    years = state.get("extracted_years", [])
    rewritten = state.get("rewritten_queries", [])
    csv_data = state.get("csv_data", {})

    # 추가 검색 루프 시 쿼리 교체
    if state.get("needs_more_search") and state.get("additional_query"):
        user_query = state["additional_query"]

    print(
        f"[Retriever] ▶ INPUT: query='{user_query}', intent={intent}, years={years}, "
        f"rewritten={len(rewritten)} queries",
        flush=True,
    )

    # ─── rewritten_queries 분류 ──────────────────────────────
    # structured: query_rewriter가 직접 파라미터 생성한 것 → LLM 우회하여 직접 실행
    # semantic: 벡터/BM25 검색용 → LLM 플래닝으로 실행

    direct_steps = []
    semantic_queries = []

    for q in rewritten:
        if q.get("type") == "structured" and q.get("structured_params"):
            params = dict(q["structured_params"])
            # years 정규화: list → comma string
            q_years = q.get("years", [])
            if q_years and not params.get("years"):
                params["years"] = ",".join(str(y) for y in q_years)
            direct_steps.append({"tool": "structured_query", "params": params})
        else:
            text = q.get("query", "")
            if text:
                semantic_queries.append(text)

    if not semantic_queries:
        semantic_queries = [user_query]

    # ─── Stage 1: structured_params 직접 실행 ────────────────

    direct_results = []
    if direct_steps:
        print(f"[Retriever] Stage 1: Direct structured queries ({len(direct_steps)} steps)", flush=True)
        direct_plan = {"steps": direct_steps}
        direct_results = _execute_plan(direct_plan, tools, state)
        print(f"[Retriever] Stage 1 done: {len(direct_results)} results", flush=True)

    # ─── Stage 2: LLM 기반 시맨틱 검색 계획 ─────────────────

    llm_results = []
    plan = None

    if llm:
        plan = _generate_search_plan(
            llm=llm,
            query=user_query,
            intent=intent,
            years=years,
            semantic_queries=semantic_queries,
            db_context=db_context,
        )

    if not plan or not plan.get("steps"):
        print(f"[Retriever] LLM plan failed, using fallback plan", flush=True)
        plan = _fallback_plan(semantic_queries, years, intent)
    else:
        print(f"[Retriever] LLM plan: {json.dumps(plan.get('steps', []), ensure_ascii=False)}", flush=True)

    llm_results = _execute_plan(plan, tools, state)

    # ─── Stage 3: 결과 부족 시 보완 검색 ─────────────────────

    all_results = direct_results + llm_results

    if llm and len(all_results) < 3:
        print(f"[Retriever] Stage 3: Refine search (only {len(all_results)} results so far)", flush=True)
        refine_plan = _generate_refine_plan(
            llm=llm,
            query=user_query,
            results=all_results,
            db_context=db_context,
        )
        if refine_plan and refine_plan.get("steps"):
            extra = _execute_plan(refine_plan, tools, state)
            all_results.extend(extra)
            print(f"[Retriever] Stage 3 done: +{len(extra)} results", flush=True)

    # ─── Stage 4: BM25 최종 폴백 ─────────────────────────────

    if not all_results and semantic_queries:
        print(f"[Retriever] Stage 4: BM25 fallback", flush=True)
        try:
            fb = tools["hybrid_search"].invoke({
                "query": semantic_queries[0],
                "top_k": 10,
                "mode": "bm25",
                "year": years[0] if len(years) == 1 else 0,
            })
            fb_parsed = json.loads(fb)
            if isinstance(fb_parsed, list):
                for r in fb_parsed:
                    r["search_type"] = "hybrid_search"
                all_results.extend(fb_parsed)
                print(f"[Retriever] Stage 4 done: {len(fb_parsed)} results", flush=True)
        except Exception as e:
            print(f"[Retriever] Stage 4 failed: {e}", flush=True)

    # ─── 중복 제거 ───────────────────────────────────────────

    seen_ids = set()
    unique_results = []
    for r in all_results:
        cid = r.get("chunk_id", id(r))
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_results.append(r)

    # ─── CSV 자동 로드 ───────────────────────────────────────

    csv_loaded = 0
    if "csv_reader" in tools:
        for r in unique_results:
            content = r.get("content", "")
            refs = extract_csv_refs(content)
            if refs:
                r["csv_refs"] = refs
                for ref in refs:
                    if ref not in csv_data:
                        try:
                            csv_data[ref] = tools["csv_reader"].invoke({
                                "csv_path": ref,
                                "output_format": "markdown",
                            })
                            csv_loaded += 1
                        except Exception as e:
                            csv_data[ref] = f"[CSV_ERROR] {e}"

    # ─── 스코어 정렬 ────────────────────────────────────────

    unique_results.sort(
        key=lambda x: x.get("rrf_score", 0) or x.get("score", 0),
        reverse=True,
    )

    final = unique_results[:20]
    elapsed = time.time() - t0

    print(
        f"[Retriever] ◀ OUTPUT: {len(final)} results (csv_loaded={csv_loaded}) in {elapsed:.1f}s",
        flush=True,
    )
    for i, r in enumerate(final[:5]):
        print(
            f"  #{i+1} [{r.get('search_type','?')}] year={r.get('year','?')} "
            f"section={r.get('section_path','?')[:60]} score={r.get('rrf_score', r.get('score',0)):.4f}",
            flush=True,
        )

    return {
        "search_results": final,
        "csv_data": csv_data,
        "needs_more_search": False,
    }


# ─── LLM 기반 검색 계획 ─────────────────────────────────────

def _generate_search_plan(llm, query: str, intent: str, years: list,
                          semantic_queries: list, db_context: str = "") -> dict:
    """LLM에게 시맨틱 검색 계획을 요청"""
    context = f"""## 사용자 질문
{query}

## 분석된 의도
{intent}

## 추출된 연도
{years if years else "없음 (전체 연도)"}

## 시맨틱 검색 쿼리 후보 (query_rewriter 생성)
{json.dumps(semantic_queries, ensure_ascii=False)}"""

    prompt = PLANNING_PROMPT.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        db_context=db_context if db_context else "(DB 구조 정보 없음)",
    ) + "\n\n" + context

    try:
        t0 = time.time()
        print(f"[Retriever] LLM plan call (prompt={len(prompt)}chars)", flush=True)
        response = llm.invoke([
            SystemMessage(content="당신은 검색 계획 수립 전문가입니다. JSON만 반환하세요. /no_think"),
            HumanMessage(content=prompt),
        ])
        elapsed = time.time() - t0
        print(f"[Retriever] LLM plan done in {elapsed:.1f}s", flush=True)
        return _parse_plan_json(response.content)
    except Exception as e:
        print(f"[Retriever] Plan generation failed: {e}", flush=True)
        return {}


def _generate_refine_plan(llm, query: str, results: list, db_context: str = "") -> dict:
    """결과 부족 시 추가 검색 계획"""
    summary = f"현재 {len(results)}개 결과. "
    if results:
        sections = set(r.get("section_path", "?")[:50] for r in results[:5])
        summary += f"섹션: {', '.join(sections)}"
    else:
        summary += "결과 없음."

    prompt = REFINE_PROMPT.format(
        query=query,
        results_summary=summary,
        missing_info="검색 결과가 3개 미만입니다.",
        db_context=db_context if db_context else "",
    )

    try:
        response = llm.invoke([
            SystemMessage(content="당신은 검색 계획 수립 전문가입니다. JSON만 반환하세요. /no_think"),
            HumanMessage(content=prompt),
        ])
        return _parse_plan_json(response.content)
    except Exception:
        return {}


# ─── 계획 실행 ───────────────────────────────────────────────

def _execute_plan(plan: dict, tools: dict, state: dict) -> list:
    """검색 계획의 각 step을 실행하고 결과 수집"""
    all_results = []
    steps = plan.get("steps", [])

    for step in steps:
        tool_name = step.get("tool", "")
        params = step.get("params", {})

        if tool_name not in tools:
            state.setdefault("errors", []).append(f"Unknown tool: {tool_name}")
            continue

        print(f"[Retriever] ▶ Tool '{tool_name}' params={json.dumps(params, ensure_ascii=False)}", flush=True)

        try:
            result_json = tools[tool_name].invoke(params)
            parsed = json.loads(result_json)

            if isinstance(parsed, list):
                for r in parsed:
                    r["search_type"] = tool_name
                all_results.extend(parsed)
                print(f"[Retriever] ◀ Tool '{tool_name}' → {len(parsed)} results", flush=True)

            elif isinstance(parsed, dict) and "error" in parsed:
                print(f"[Retriever] ◀ Tool '{tool_name}' error: {parsed['error']}", flush=True)
                state.setdefault("errors", []).append(f"{tool_name}: {parsed['error']}")

        except json.JSONDecodeError:
            print(f"[Retriever] ◀ Tool '{tool_name}' non-JSON: {str(result_json)[:100]}", flush=True)
        except Exception as e:
            print(f"[Retriever] ◀ Tool '{tool_name}' exception: {e}", flush=True)
            state.setdefault("errors", []).append(f"{tool_name} error: {e}")

    return all_results


# ─── 폴백 계획 ──────────────────────────────────────────────

def _fallback_plan(queries: list, years: list, intent: str) -> dict:
    """LLM 없을 때 사용하는 기본 검색 계획"""
    steps = []
    main_query = queries[0] if queries else ""
    years_str = ",".join(str(y) for y in years) if years else ""

    if intent in ("comparison", "trend"):
        steps.append({
            "tool": "structured_query",
            "params": {
                "years": years_str,
                "keyword": _extract_core_keyword(main_query),
                "limit": 20,
            },
        })

    for q in queries[:2]:
        steps.append({
            "tool": "hybrid_search",
            "params": {
                "query": q,
                "top_k": 10,
                "year": years[0] if len(years) == 1 else 0,
                "mode": "hybrid",
            },
        })

    return {"reasoning": "fallback plan", "steps": steps}


# ─── 유틸리티 ────────────────────────────────────────────────

def _parse_plan_json(text: str) -> dict:
    """LLM 응답에서 검색 계획 JSON 추출"""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


def _extract_core_keyword(query: str) -> str:
    """쿼리에서 핵심 키워드 1~3개 추출"""
    stopwords = {"삼성전자", "삼성", "은", "는", "이", "가", "을", "를", "의",
                 "해줘", "해주세요", "알려줘", "설명해줘", "보여줘", "얼마",
                 "무엇", "어떻게", "년", "년도"}
    cleaned = re.sub(r"20[0-2]\d년?", "", query)
    words = re.findall(r"[\w]+", cleaned)
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(keywords[:3])
