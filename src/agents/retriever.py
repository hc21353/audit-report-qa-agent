"""
retriever.py - LLM 기반 자율 검색 에이전트 노드

흐름:
  1. rewritten_queries에서 structured_params가 있는 쿼리는 직접 실행 (LLM 우회)
  2. 관련 연도의 실제 DB 섹션 구조를 list_available_sections로 동적 조회
  3. 실제 구조 기반으로 LLM이 검색 계획 수립
  4. 결과 부족 시 보완 검색 (web_search 포함)
  5. 최종 폴백: BM25 → web_search
  6. [TABLE_CSV] 태그 감지 시 csv_reader 자동 호출
"""

import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage
from src.csv_reader import extract_csv_refs


# ─── Tool 설명 (LLM에게 제공) ───────────────────────────────

TOOL_DESCRIPTIONS = """사용 가능한 검색 도구:

1. hybrid_search
   - 설명: 벡터 유사도 + BM25 키워드를 결합한 하이브리드 검색.
   - 필수 파라미터: query (str) — 반드시 있어야 함.
   - 선택 파라미터:
     top_k (int, 기본 10)
     year (int, 기본 0=전체)
     section_path_contains (str, 기본 ""): section_path 경로의 부분 문자열
       예: "주석" → 모든 주석, "9. 종속" → 9번 주석
     chunk_type (str, 기본 ""=전체): "Narrative"|"Note"|"Table_Row"
     mode ("hybrid"|"vector"|"bm25", 기본 "hybrid")
   - 예시: {"tool": "hybrid_search", "params": {"query": "수익인식 매출액", "top_k": 10, "year": 2024}}

2. structured_query
   - 설명: section_path 계층 구조와 태그를 활용한 정밀 검색.
   - 파라미터:
     years (str): 쉼표 구분 연도 (예: "2022,2023,2024" 또는 빈 문자열=전체)
     section_path_contains (str): section_path 경로의 부분 문자열
       최상위: "(첨부)재무제표" | "독립된 감사인의 감사보고서" | "내부회계관리제도" | "외부감사 실시내용"
       주석: "주석" | "9. 종속기업" | "30. 특수관계자" 등 (DB 섹션 구조 참조)
     chunk_type (str): "Table_Row"(수치데이터) | "Note"(주석설명) | "Narrative"(서술) | "all"
     is_consolidated (int): 1=연결, 0=별도, -1=전체
     tags (str): 쉼표 구분 태그. DB 태그 목록(#포함)에서 선택.
                 (예: "#이익,#수익,#비유동자산"). 목록에 없는 태그 사용 시 0건 반환.
     keyword (str): 본문 키워드. 쉼표로 다중 키워드 가능 (OR 검색)
     limit (int, 기본 30)

3. web_search
   - 설명: 인터넷 검색. DB에 없는 정보를 보완할 때 사용.
   - 파라미터: query (str, 필수), max_results (int, 기본 5)
   - 주의: 결과에 [WEB] 태그가 붙어 출처 불확실 경고가 표시됨

4. get_full_content
   - 설명: 특정 청크의 전체 내용 조회. is_truncated=true인 청크에 사용.
   - 파라미터: chunk_ids (str, 쉼표 구분 chunk_uid 또는 integer id),
               include_adjacent (bool, 기본 false)"""


PLANNING_PROMPT = """당신은 감사보고서 검색 계획 수립 전문가입니다.
아래 정보를 바탕으로 사용자 질문에 답하기 위한 최적의 검색 계획을 JSON으로 작성하세요.

{tool_descriptions}

## 이 질문에서 사용 가능한 실제 DB 섹션 구조
(아래 경로 정보는 실제 DB에서 조회한 값입니다. structured_query의 section_path_contains 사용 시 참조하세요.)
{live_sections}

## 검색 전략 가이드
- section_path_contains는 위 "실제 DB 섹션 구조"의 경로 일부만 사용 (임의로 만들지 말 것)
  예: "주석" → 모든 주석, "9. 종속" → 9번 주석, "재무제표" → 재무제표 섹션
- 수치 데이터 필요 시: chunk_type="Table_Row" 권장
- 여러 연도 비교/추이: years="2022,2023,2024" 형태로 한 번에 조회 (연도별 별도 호출 금지)
- 섹션 구조를 모를 때는 먼저 hybrid_search로 탐색 후 section 파악
- DB에서 찾을 수 없는 정보라고 판단되면 web_search 사용 가능
- 검색 계획은 1~3개의 tool 호출로 구성

## 재무비율 계산 (ROE·ROA·부채비율 등) 전용 가이드
재무비율은 분자/분모를 반드시 **별도 tool 호출**로 검색하세요. tags 없이 keyword+section_path만 사용:
  - 당기순이익:  structured_query(section_path_contains="(첨부)재무제표", chunk_type="Table_Row", keyword="당기순이익", tags="")
  - 자본총계:    structured_query(section_path_contains="(첨부)재무제표", chunk_type="Table_Row", keyword="자본총계", tags="")
  - 자산총계:    structured_query(section_path_contains="(첨부)재무제표", chunk_type="Table_Row", keyword="자산총계", tags="")
  - 부채총계:    structured_query(section_path_contains="(첨부)재무제표", chunk_type="Table_Row", keyword="부채총계", tags="")
  - 영업이익:    structured_query(section_path_contains="(첨부)재무제표", chunk_type="Table_Row", keyword="영업이익", tags="")
⚠️ 재무제표 본문 합계 항목은 태그가 부정확할 수 있으므로 tags="" 필수

## 규칙
- JSON만 반환하고 다른 텍스트는 쓰지 마세요.

## 출력 형식
{{
  "reasoning": "검색 전략 설명 (1~2문장)",
  "steps": [
    {{"tool": "도구명", "params": {{...}} }},
    ...
  ]
}}"""


REFINE_PROMPT = """검색 결과가 부족합니다. 다른 방향으로 추가 검색하세요.

## 사용자 질문
{query}

## 현재까지 검색 결과 요약
{results_summary}

## 이미 시도한 tool 호출 (반드시 피할 것)
{tried_steps}

## 실제 DB 섹션 구조
{live_sections}

## 추가 검색 전략
- 위 "이미 시도한 tool 호출"과 동일한 tool/params 사용 금지
- 이미 structured_query를 사용했다면 → hybrid_search로 다른 키워드 시도
- 이미 hybrid_search를 사용했다면 → structured_query로 섹션 직접 조회 (위 실제 섹션명 사용)
- DB에서 관련 정보를 찾을 수 없다면 → web_search 사용 (결과에 출처 불확실 경고 자동 표시)

사용 가능한 tool: hybrid_search, structured_query, web_search
추가 검색 계획을 1~2개의 tool 호출로 작성하세요. JSON만 반환하세요.
{{
  "reasoning": "추가 검색 전략 설명",
  "steps": [
    {{"tool": "도구명", "params": {{...}} }}
  ]
}}"""


# ─── 메인 노드 ──────────────────────────────────────────────

def retriever_node(state: dict, tools: dict, llm=None, db_context: str = "") -> dict:
    """
    LangGraph 노드: LLM이 검색 계획을 수립하고 tool을 실행.

    Args:
        state:      현재 그래프 상태
        tools:      {"hybrid_search", "structured_query", "list_available_sections", "csv_reader", "web_search"}
        llm:        retriever용 ChatOllama 인스턴스
        db_context: graph 초기화 시 build_db_context()로 생성된 h2→h3 계층 구조 문자열

    Returns:
        search_results, csv_data 업데이트
    """
    t0 = time.time()
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    years = state.get("extracted_years", [])
    sections = state.get("extracted_sections", [])
    rewritten = state.get("rewritten_queries", [])
    csv_data = state.get("csv_data", {})

    # 추가 검색 루프 여부 먼저 판단
    is_additional_search = state.get("needs_more_search", False)

    # 추가 검색 루프 시 쿼리 교체
    if is_additional_search and state.get("additional_query"):
        user_query = state["additional_query"]

    print(
        f"[Retriever] ▶ INPUT: query='{user_query}', intent={intent}, years={years}, "
        f"rewritten={len(rewritten)} queries, additional_search={is_additional_search}",
        flush=True,
    )

    # ─── rewritten_queries 분류 ─────────────────────────────────
    direct_steps = []
    semantic_queries = []

    if is_additional_search:
        # 추가 검색 시 stale한 rewritten_queries 대신 additional_query를 새 시맨틱 쿼리로 사용
        semantic_queries = [user_query]
        # 추가 쿼리에서 연도 재추출 (state의 years가 원 질문 기준이므로 갱신)
        extra_years = sorted({int(y) for y in re.findall(r"20[0-2]\d", user_query)})
        if extra_years:
            years = extra_years
    else:
        for q in rewritten:
            if q.get("type") == "structured" and q.get("structured_params"):
                params = dict(q["structured_params"])
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

    # ─── DB 섹션 구조 준비 ──────────────────────────────────────
    # graph.py에서 주입된 db_context (build_db_context로 생성된 h2→h3 계층 구조)를 사용.
    # 연도가 지정된 경우 해당 연도 관련 섹션만 추출하여 토큰 절약.
    live_sections = _extract_sections_for_years(db_context, years)
    print(f"[Retriever] DB sections context ({len(live_sections)} chars)", flush=True)

    # ─── Stage 1: structured_params 직접 실행 (첫 번째 iteration만) ──
    # needs_more_search 루프에서는 이미 실패한 동일 쿼리를 재실행하지 않음

    direct_results = []
    if direct_steps and not is_additional_search:
        print(f"[Retriever] Stage 1: Direct structured queries ({len(direct_steps)} steps)", flush=True)
        direct_results = _execute_plan({"steps": direct_steps}, tools, state)
        print(f"[Retriever] Stage 1 done: {len(direct_results)} results", flush=True)
    elif is_additional_search:
        print(f"[Retriever] Stage 1: Skipped (additional search iteration)", flush=True)

    # ─── Stage 2: LLM 기반 시맨틱 검색 계획 ────────────────────

    llm_results = []
    plan = None

    if llm:
        plan = _generate_search_plan(
            llm=llm,
            query=user_query,
            intent=intent,
            years=years,
            semantic_queries=semantic_queries,
            live_sections=live_sections,
            sections=sections,
        )

    if not plan or not plan.get("steps"):
        print(f"[Retriever] LLM plan failed, using fallback plan", flush=True)
        plan = _fallback_plan(semantic_queries, years, intent, sections)
    else:
        print(f"[Retriever] LLM plan: {json.dumps(plan.get('steps', []), ensure_ascii=False)}", flush=True)

    # LLM plan에 hybrid_search가 없으면 강제 추가 (벡터 검색 활용 보장)
    has_hybrid = any(s.get("tool") == "hybrid_search" for s in plan.get("steps", []))
    if not has_hybrid and semantic_queries:
        plan.setdefault("steps", []).append({
            "tool": "hybrid_search",
            "params": {
                "query": semantic_queries[0],
                "top_k": _top_k_for_intent(intent),
                "year": years[0] if len(years) == 1 else 0,
                "mode": "hybrid",
            },
        })

    llm_results = _execute_plan(plan, tools, state)

    # ─── Stage 3: 결과 부족 시 보완 검색 (web_search 포함) ──────

    all_results = direct_results + llm_results

    if llm and len(all_results) < 3:
        print(f"[Retriever] Stage 3: Refine search (only {len(all_results)} results so far)", flush=True)
        tried = direct_steps + plan.get("steps", [])
        refine_plan = _generate_refine_plan(
            llm=llm,
            query=user_query,
            results=all_results,
            live_sections=live_sections,
            tried_steps=tried,
        )
        if refine_plan and refine_plan.get("steps"):
            extra = _execute_plan(refine_plan, tools, state)
            all_results.extend(extra)
            print(f"[Retriever] Stage 3 done: +{len(extra)} results", flush=True)

    # ─── Stage 4: BM25 폴백 → web_search 최종 폴백 ─────────────

    if not all_results:
        print(f"[Retriever] Stage 4: BM25 fallback", flush=True)
        try:
            bm25_query = _extract_core_keyword(user_query) or semantic_queries[0]
            fb = tools["hybrid_search"].invoke({
                "query": bm25_query,
                "top_k": _top_k_for_intent(intent),
                "mode": "bm25",
                "year": years[0] if len(years) == 1 else 0,
            })
            fb_parsed = json.loads(fb)
            if isinstance(fb_parsed, list):
                for r in fb_parsed:
                    r["search_type"] = "hybrid_search"
                all_results.extend(fb_parsed)
                print(f"[Retriever] Stage 4 done: {len(fb_parsed)} BM25 results", flush=True)
        except Exception as e:
            print(f"[Retriever] Stage 4 BM25 failed: {e}", flush=True)

    # BM25도 의미있는 결과를 못 찾으면 web_search 시도
    if len(all_results) < 3 and "web_search" in tools:
        print(f"[Retriever] Stage 5: Web search fallback", flush=True)
        try:
            web_result_json = tools["web_search"].invoke({
                "query": user_query,
                "max_results": 5,
            })
            web_parsed = json.loads(web_result_json)
            if isinstance(web_parsed, list):
                for r in web_parsed:
                    r["search_type"] = "web_search"
                    r["is_web_source"] = True  # analyst가 경고 표시용
                all_results.extend(web_parsed)
                print(f"[Retriever] Stage 5 done: {len(web_parsed)} web results", flush=True)
        except Exception as e:
            print(f"[Retriever] Stage 5 web search failed: {e}", flush=True)

    # ─── 중복 제거 (이번 검색 내부) ────────────────────────────────

    seen_ids = set()
    unique_results = []
    for r in all_results:
        cid = r.get("chunk_uid") or id(r)
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_results.append(r)

    # ─── 누적: 이전 iteration 결과와 병합 (이미 본 chunk 제외) ───
    # 이전 결과를 앞에 고정하고 새 결과를 뒤에 추가.
    # re-sort 금지: 새 iteration의 고점수 결과가 이전 좋은 결과를 덮어쓰는 문제 방지.

    if is_additional_search:
        prev_results = state.get("search_results", [])
        prev_ids = {r.get("chunk_uid") for r in prev_results if r.get("chunk_uid")}
        new_only = [r for r in unique_results if r.get("chunk_uid") not in prev_ids]
        # 새 결과만 내부 정렬
        new_only.sort(key=lambda x: (
            1 if x.get("is_web_source") else 0,
            -(x.get("rrf_score", 0) or x.get("score", 0)),
        ))
        print(f"[Retriever] Accumulate: prev={len(prev_results)}, new={len(new_only)} new chunks", flush=True)
        # additional_search의 목적은 새 정보 탐색 → 새 결과를 앞에 배치
        # analyst가 새로운 관련 데이터를 먼저 볼 수 있도록
        unique_results = new_only + prev_results
        # 누적 후에는 re-sort 하지 않음 → 아래 sort 블록을 건너뜀

    # ─── CSV 자동 로드 ──────────────────────────────────────────

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

    # ─── 스코어 정렬 (첫 번째 iteration만, 누적 시에는 기존 순서 유지) ────

    if not is_additional_search:
        unique_results.sort(
            key=lambda x: (
                1 if x.get("is_web_source") else 0,  # DB 결과 우선 (0 < 1)
                -(x.get("rrf_score", 0) or x.get("score", 0)),
            ),
        )

    # additional_search 시 새 결과가 앞에 있으므로 더 많이 전달
    result_limit = 40 if is_additional_search else 25
    final = unique_results[:result_limit]
    elapsed = time.time() - t0

    print(
        f"[Retriever] ◀ OUTPUT: {len(final)} results (csv_loaded={csv_loaded}) in {elapsed:.1f}s",
        flush=True,
    )
    for i, r in enumerate(final[:5]):
        src = "[WEB]" if r.get("is_web_source") else f"score={r.get('rrf_score', r.get('score', 0)):.4f}"
        print(
            f"  #{i+1} [{r.get('search_type','?')}] year={r.get('fiscal_year','?')} "
            f"type={r.get('chunk_type','?')} section={r.get('section_path','?')[:60]} {src}",
            flush=True,
        )

    return {
        "search_results": final,
        "csv_data": csv_data,
        "needs_more_search": False,
    }


# ─── DB 섹션 컨텍스트 추출 ────────────────────────────────────────

def _extract_sections_for_years(db_context: str, years: list) -> str:
    """
    graph.py가 빌드한 db_context 문자열에서 관련 연도 섹션만 추출.
    연도 미지정 시 전체 반환. db_context는 build_db_context()가 생성한
    h2→h3 올바른 계층 구조를 포함.
    """
    if not db_context:
        return "(DB 구조 정보 없음)"
    if not years:
        return db_context

    lines = db_context.split("\n")
    result = []
    in_target_year = False
    header_done = False

    for line in lines:
        # 헤더 라인 (## DB 실제 섹션 구조, 적재된 연도 등)은 항상 포함
        if not header_done and not line.startswith("### "):
            result.append(line)
            continue

        # 연도 헤딩 감지
        if line.startswith("### "):
            header_done = True
            try:
                year_in_line = int(line.strip("### ").replace("년", "").strip())
                in_target_year = year_in_line in years
            except ValueError:
                in_target_year = False

        if in_target_year:
            result.append(line)

    # 검색 파라미터 가이드는 항상 포함
    guide_start = False
    for line in lines:
        if line.startswith("## 검색 파라미터"):
            guide_start = True
        if guide_start:
            result.append(line)

    return "\n".join(result) if result else db_context


# ─── LLM 기반 검색 계획 ─────────────────────────────────────────

def _generate_search_plan(llm, query: str, intent: str, years: list,
                          semantic_queries: list, live_sections: str = "",
                          sections: list = None) -> dict:
    """LLM에게 시맨틱 검색 계획을 요청"""
    context = f"""## 사용자 질문
{query}

## 분석된 의도
{intent}

## 추출된 연도
{years if years else "없음 (전체 연도)"}

## 추출된 섹션 (section_path_contains에 반드시 사용)
{json.dumps(sections or [], ensure_ascii=False) if sections else "없음"}

## 시맨틱 검색 쿼리 후보 (query_rewriter 생성)
{json.dumps(semantic_queries, ensure_ascii=False)}"""

    prompt = PLANNING_PROMPT.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        live_sections=live_sections if live_sections else "(섹션 정보 없음 — hybrid_search 사용 권장)",
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


def _generate_refine_plan(llm, query: str, results: list, live_sections: str = "",
                          tried_steps: list = None) -> dict:
    """결과 부족 시 추가 검색 계획"""
    summary = f"현재 {len(results)}개 결과. "
    if results:
        sections = set(r.get("section_path", "?")[:50] for r in results[:5])
        summary += f"섹션: {', '.join(sections)}"
    else:
        summary += "결과 없음 — DB에 관련 정보가 없을 수 있음."

    tried_summary = ""
    if tried_steps:
        tried_summary = "\n".join(
            f"- tool={s.get('tool')}, params={json.dumps(s.get('params', {}), ensure_ascii=False)}"
            for s in tried_steps
        )
    else:
        tried_summary = "(없음)"

    prompt = REFINE_PROMPT.format(
        query=query,
        results_summary=summary,
        tried_steps=tried_summary,
        live_sections=live_sections if live_sections else "(섹션 정보 없음)",
    )

    try:
        response = llm.invoke([
            SystemMessage(content="당신은 검색 계획 수립 전문가입니다. JSON만 반환하세요. /no_think"),
            HumanMessage(content=prompt),
        ])
        return _parse_plan_json(response.content)
    except Exception:
        return {}


# ─── 계획 실행 ───────────────────────────────────────────────────

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

        # list_available_sections는 섹션 탐색용이므로 검색 결과에 포함하지 않음
        if tool_name == "list_available_sections":
            print(f"[Retriever] ▶ Tool '{tool_name}' (section probe, skipping result collection)", flush=True)
            continue

        print(f"[Retriever] ▶ Tool '{tool_name}' params={json.dumps(params, ensure_ascii=False)}", flush=True)

        try:
            result_json = tools[tool_name].invoke(params)
            parsed = json.loads(result_json)

            if isinstance(parsed, list):
                for rank, r in enumerate(parsed, 1):
                    r["search_type"] = tool_name
                    if tool_name == "web_search":
                        r["is_web_source"] = True
                    # structured_query 결과에 합성 rrf_score 부여 (정렬 시 hybrid 결과와 혼합)
                    # 위치 기반 RRF: rank=1 → ~0.0164, rank=20 → ~0.0125
                    if tool_name == "structured_query" and not r.get("rrf_score"):
                        r["rrf_score"] = round(1.0 / (60 + rank), 6)
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


# ─── 폴백 계획 ──────────────────────────────────────────────────

def _fallback_plan(queries: list, years: list, intent: str, sections: list = None) -> dict:
    """LLM 없을 때 사용하는 기본 검색 계획"""
    steps = []
    main_query = queries[0] if queries else ""
    years_str = ",".join(str(y) for y in years) if years else ""
    section_filter = sections[0] if sections else ""

    if intent in ("comparison", "trend"):
        steps.append({
            "tool": "structured_query",
            "params": {
                "years": years_str,
                "section_path_contains": section_filter,
                "keyword": _extract_core_keyword(main_query),
                "limit": 20,
            },
        })

    for q in queries[:2]:
        steps.append({
            "tool": "hybrid_search",
            "params": {
                "query": q,
                "top_k": _top_k_for_intent(intent),
                "year": years[0] if len(years) == 1 else 0,
                "section_path_contains": section_filter,
                "mode": "hybrid",
            },
        })

    return {"reasoning": "fallback plan", "steps": steps}


# ─── 유틸리티 ────────────────────────────────────────────────────

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



def _top_k_for_intent(intent: str) -> int:
    """intent에 따라 적절한 top_k 반환"""
    if intent in ("trend", "comparison"):
        return 30   # 여러 연도 × 여러 항목 필요
    elif intent == "calculation":
        return 20   # 계산에 필요한 수치들
    elif intent == "simple_lookup":
        return 10   # 단순 단일 조회
    else:
        return 15   # general


def _extract_core_keyword(query: str) -> str:
    """쿼리에서 핵심 키워드 1~3개 추출 (한국어 조사 제거)"""
    stopwords = {"삼성전자", "삼성", "은", "는", "이", "가", "을", "를", "의",
                 "해줘", "해주세요", "알려줘", "설명해줘", "보여줘", "얼마",
                 "무엇", "어떻게", "년", "년도", "최근", "어떻게", "돼"}
    # 한국어 조사 제거 패턴
    particle_re = re.compile(
        r"(에서의|으로의|에서는|으로는|에서도|에서|으로|에는|에도|와의|과의"
        r"|이란|라는|란|의|는|은|이|가|을|를|에|와|과|로|도|만|부터|까지|처럼|같은|에게|한테)$"
    )
    cleaned = re.sub(r"20[0-2]\d년?", "", query)
    words = re.findall(r"[\w]+", cleaned)
    keywords = []
    for w in words:
        stripped = particle_re.sub("", w) if len(w) > 2 else w
        if stripped and stripped not in stopwords and len(stripped) > 1:
            keywords.append(stripped)
    return " ".join(keywords[:4])
