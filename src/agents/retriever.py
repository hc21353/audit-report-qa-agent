"""
retriever.py - LLM 기반 자율 검색 에이전트 노드

기존: intent에 따라 하드코딩된 규칙으로 tool 선택
변경: LLM이 사용 가능한 tool + 파라미터를 보고 직접 검색 계획을 수립

흐름:
  1. LLM에게 질문 + 의도 + 사용 가능한 tool 설명을 전달
  2. LLM이 검색 계획(JSON)을 생성 (어떤 tool을 어떤 파라미터로 호출할지)
  3. 계획에 따라 tool 순차 실행
  4. 결과가 부족하면 LLM이 추가 검색 계획 생성 (최대 2라운드)
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
     section_h3 (str): 주석 번호 (부분 매칭, 예: "특수관계자")
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
- section_h3/h4/h5는 빈 문자열로 두는 게 안전함. 모르면 h2와 keyword만 사용할 것
- 벡터 인덱스가 없을 경우 hybrid_search의 mode="bm25"를 사용할 것"""


PLANNING_PROMPT = """당신은 감사보고서 검색 전문가입니다.
사용자 질문과 의도를 분석하여 최적의 검색 계획을 JSON으로 작성하세요.

{tool_descriptions}

{db_context}

## 규칙
- 검색 계획은 1~3개의 tool 호출로 구성하세요.
- 각 호출은 독립적이며 병렬 실행됩니다.
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
        llm:        retriever용 ChatOllama 인스턴스 (None이면 폴백 규칙 사용)
        db_context: 그래프 초기화 시 생성된 DB 섹션 구조 문자열

    Returns:
        search_results, csv_data 업데이트
    """
    print(f"[Retriever] Start", flush=True)
    t0 = time.time()
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    years = state.get("extracted_years", [])
    rewritten = state.get("rewritten_queries", [])
    csv_data = state.get("csv_data", {})

    # 추가 검색 루프
    if state.get("needs_more_search") and state.get("additional_query"):
        user_query = state["additional_query"]

    # 검색 쿼리 결정
    queries = []
    for q in rewritten:
        if q.get("query"):
            queries.append(q["query"])
    if not queries:
        queries = [user_query]

    # ─── Round 1: LLM 기반 검색 계획 수립 ────────────────────

    plan = None
    if llm:
        plan = _generate_search_plan(
            llm=llm,
            query=user_query,
            intent=intent,
            years=years,
            rewritten_queries=queries,
            db_context=db_context,
        )

    if not plan or not plan.get("steps"):
        # LLM 실패 시 폴백: 기본 검색 계획
        print(f"[Retriever] LLM plan failed, using fallback", flush=True)
        plan = _fallback_plan(queries, years, intent)
    else:
        print(f"[Retriever] Plan: {plan.get('steps', [])}", flush=True)

    # ─── 계획 실행 ───────────────────────────────────────────

    all_results = _execute_plan(plan, tools, state)

    # ─── Round 2: 결과 부족 시 추가 검색 (선택적) ─────────────

    if llm and len(all_results) < 3:
        refine_plan = _generate_refine_plan(
            llm=llm,
            query=user_query,
            results=all_results,
            db_context=db_context,
        )
        if refine_plan and refine_plan.get("steps"):
            extra_results = _execute_plan(refine_plan, tools, state)
            all_results.extend(extra_results)

    # ─── 최종 폴백: 여전히 결과 없으면 BM25 단독 검색 ─────────
    if not all_results and queries:
        print(f"[Retriever] Zero results, fallback BM25 search", flush=True)
        try:
            fallback_result = tools["hybrid_search"].invoke({
                "query": queries[0],
                "top_k": 10,
                "mode": "bm25",
                "year": years[0] if len(years) == 1 else 0,
            })
            fallback_parsed = json.loads(fallback_result)
            if isinstance(fallback_parsed, list):
                for r in fallback_parsed:
                    r["search_type"] = "hybrid_search"
                all_results.extend(fallback_parsed)
                print(f"[Retriever] Fallback BM25 → {len(fallback_parsed)} results", flush=True)
        except Exception as e:
            print(f"[Retriever] Fallback BM25 failed: {e}", flush=True)

    # ─── 중복 제거 ───────────────────────────────────────────

    seen_ids = set()
    unique_results = []
    for r in all_results:
        cid = r.get("chunk_id", id(r))
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_results.append(r)

    # ─── CSV 자동 로드 ───────────────────────────────────────

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
                        except Exception as e:
                            csv_data[ref] = f"[CSV_ERROR] {e}"

    # ─── 스코어 정렬 ────────────────────────────────────────

    unique_results.sort(
        key=lambda x: x.get("rrf_score", 0) or x.get("score", 0),
        reverse=True,
    )

    print(f"[Retriever] Done in {time.time()-t0:.1f}s, results={len(unique_results)}", flush=True)
    return {
        "search_results": unique_results[:20],
        "csv_data": csv_data,
        "needs_more_search": False,
    }


# ─── LLM 기반 검색 계획 생성 ────────────────────────────────

def _generate_search_plan(llm, query: str, intent: str, years: list,
                          rewritten_queries: list, db_context: str = "") -> dict:
    """LLM에게 검색 계획을 요청"""
    context = f"""## 사용자 질문
{query}

## 분석된 의도
{intent}

## 추출된 연도
{years if years else "없음 (전체 연도)"}

## 재작성된 쿼리
{json.dumps(rewritten_queries, ensure_ascii=False)}"""

    prompt = PLANNING_PROMPT.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        db_context=db_context if db_context else "(DB 구조 정보 없음)",
    ) + "\n\n" + context

    try:
        t0 = time.time()
        print(f"[Retriever] LLM plan call start (prompt_len={len(prompt)})", flush=True)
        response = llm.invoke([
            SystemMessage(content="당신은 검색 계획 수립 전문가입니다. JSON만 반환하세요. /no_think"),
            HumanMessage(content=prompt),
        ])
        print(f"[Retriever] LLM plan call done in {time.time()-t0:.1f}s", flush=True)
        return _parse_plan_json(response.content)
    except Exception as e:
        print(f"[Retriever] Plan generation failed: {e}")
        return {}


def _generate_refine_plan(llm, query: str, results: list, db_context: str = "") -> dict:
    """결과 부족 시 추가 검색 계획 생성"""
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
    """검색 계획의 각 step을 실행"""
    all_results = []
    steps = plan.get("steps", [])

    for step in steps:
        tool_name = step.get("tool", "")
        params = step.get("params", {})

        if tool_name not in tools:
            state.setdefault("errors", []).append(f"Unknown tool: {tool_name}")
            continue

        try:
            result_json = tools[tool_name].invoke(params)
            parsed = json.loads(result_json)

            if isinstance(parsed, list):
                for r in parsed:
                    r["search_type"] = tool_name
                all_results.extend(parsed)
                print(f"[Retriever] {tool_name} → {len(parsed)} results", flush=True)
            elif isinstance(parsed, dict) and "error" in parsed:
                print(f"[Retriever] {tool_name} error: {parsed['error']}", flush=True)
                state.setdefault("errors", []).append(
                    f"{tool_name}: {parsed['error']}"
                )
        except json.JSONDecodeError:
            print(f"[Retriever] {tool_name} non-JSON result: {str(result_json)[:100]}", flush=True)
        except Exception as e:
            print(f"[Retriever] {tool_name} exception: {e}", flush=True)
            state.setdefault("errors", []).append(f"{tool_name} error: {e}")

    return all_results


# ─── 폴백: LLM 없을 때 규칙 기반 계획 ───────────────────────

def _fallback_plan(queries: list, years: list, intent: str) -> dict:
    """LLM 없을 때 사용하는 기본 검색 계획"""
    steps = []
    main_query = queries[0] if queries else ""
    years_str = ",".join(str(y) for y in years) if years else ""

    if intent in ("comparison", "trend"):
        # 구조 검색으로 연도별 데이터 확보
        steps.append({
            "tool": "structured_query",
            "params": {
                "years": years_str,
                "keyword": _extract_core_keyword(main_query),
                "limit": 20,
            },
        })

    # 항상 하이브리드 검색도 수행
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
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}


def _extract_core_keyword(query: str) -> str:
    """쿼리에서 핵심 키워드 1~2개 추출"""
    stopwords = {"삼성전자", "삼성", "은", "는", "이", "가", "을", "를", "의",
                 "해줘", "해주세요", "알려줘", "설명해줘", "보여줘", "얼마",
                 "무엇", "어떻게", "년", "년도"}
    cleaned = re.sub(r"20[0-2]\d년?", "", query)
    words = re.findall(r"[\w]+", cleaned)
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(keywords[:3])
