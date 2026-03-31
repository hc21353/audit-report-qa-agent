"""
query_rewriter.py - LLM 기반 쿼리 재작성 에이전트 노드

역할:
  - 벡터 임베딩에 최적화된 시맨틱 쿼리 생성 (의미 중심 자연어 문장)
  - DB 구조 기반 정밀 검색 파라미터 생성 (structured_query tool용)
  - 완전히 LLM 기반으로 동작 (규칙 기반 없음)

출력 rewritten_queries 형식:
  - type="semantic":    벡터 임베딩 검색용 자연어 쿼리
  - type="structured":  DB 직접 검색용 (structured_params 포함)
"""

import json
import re
import time
from langchain_core.messages import SystemMessage, HumanMessage


REWRITE_PROMPT = """당신은 감사보고서 검색 전문가입니다.
사용자 질문을 분석하여 두 가지 유형의 검색 쿼리를 생성하세요.

{db_context}

## 사용자 질문
{query}

## 분석된 의도
{intent}

## 추출된 연도
{years}

## 서브 질문 (복합 질문 분해 결과)
{sub_questions}

## 출력 형식 (JSON만 반환, 다른 텍스트 없이)
{{
  "reasoning": "검색 전략에 대한 간단한 설명",
  "semantic_queries": [
    "벡터 임베딩 최적화 쿼리 1 - 완전한 자연어 설명문",
    "벡터 임베딩 최적화 쿼리 2 - 동의어/관련 개념 포함",
    "벡터 임베딩 최적화 쿼리 3 - 다른 각도로 재구성"
  ],
  "structured_queries": [
    {{
      "years": "연도 (쉼표 구분, 예: '2022,2023'). 전체이면 빈 문자열",
      "section_h2": "위 DB 구조에서 실제 존재하는 h2 이름 (없으면 빈 문자열)",
      "section_h3": "위 DB 구조에서 실제 존재하는 h3 이름 또는 앞부분 (없으면 빈 문자열)",
      "section_h4": "",
      "keyword": "본문 추가 필터 키워드 (없으면 빈 문자열)",
      "content_type": "all",
      "limit": 20
    }}
  ]
}}

## 생성 규칙
**semantic_queries:**
- 짧은 키워드 나열이 아니라 의미가 풍부한 완전한 문장 (벡터 검색 성능에 유리)
- 질문의 핵심 의미를 다양한 표현으로 변환
- 서브 질문이 있으면 각각에 대한 쿼리 포함
- 예시) "중요한 회계추정 및 가정의 내용과 수익인식 방법"

**structured_queries:**
- section_h2/h3는 반드시 위 "DB 실제 섹션 구조"에 있는 값만 사용
- section_h3는 부분 매칭이므로 "3." 처럼 번호 앞부분만 써도 됨
- 불확실하면 section_h3를 빈 문자열로 두고 keyword만 사용
- 확실한 경우에만 structured_queries를 생성 (불필요한 조회 방지)

JSON만 반환하세요."""


def query_rewriter_node(state: dict, llm, system_prompt: str = "", db_context: str = "") -> dict:
    """
    LangGraph 노드: LLM 기반 쿼리 재작성.

    - semantic_queries: 벡터 임베딩에 최적화된 자연어 쿼리 (2~3개)
    - structured_queries: DB 구조 직접 검색 파라미터 (0~2개)

    Returns:
        rewritten_queries: [
            {"query": str, "years": list, "type": "semantic"|"structured",
             "strategy": str, "structured_params"?: dict}
        ]
    """
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    years = state.get("extracted_years", [])
    sub_questions = state.get("sub_questions", [])

    print(f"[QueryRewriter] ▶ INPUT: query='{user_query}', intent={intent}, years={years}, sub_questions={sub_questions}", flush=True)
    t0 = time.time()

    prompt = REWRITE_PROMPT.format(
        db_context=db_context if db_context else "(DB 구조 정보 없음 — section_h2/h3는 빈 문자열로 설정)",
        query=user_query,
        intent=intent,
        years=years if years else "없음 (전체 연도)",
        sub_questions=sub_questions if sub_questions else "없음",
    )

    reasoning = "(없음)"
    parsed = {}
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt or "당신은 감사보고서 검색 전문가입니다. JSON만 반환하세요. /no_think"),
            HumanMessage(content=prompt),
        ])
        raw = response.content
        print(f"[QueryRewriter] LLM raw ({len(raw)}chars): {raw[:300]}{'...' if len(raw)>300 else ''}", flush=True)
        parsed = _parse_json_response(raw)
        reasoning = parsed.get("reasoning", "(없음)")
    except Exception as e:
        print(f"[QueryRewriter] LLM error: {e}", flush=True)
        state.setdefault("errors", []).append(f"QueryRewriter LLM error: {e}")

    # 결과 변환
    all_queries = []
    years_str = ",".join(str(y) for y in years) if years else ""

    # 1. 시맨틱 쿼리 (벡터 임베딩용)
    for i, sq in enumerate(parsed.get("semantic_queries", [])):
        if sq and isinstance(sq, str) and sq.strip():
            all_queries.append({
                "query": sq.strip(),
                "years": years,
                "sections": [],
                "type": "semantic",
                "strategy": f"llm_semantic_{i+1}",
            })

    # 2. 구조 기반 쿼리 (DB 직접 조회용)
    for i, sq in enumerate(parsed.get("structured_queries", [])):
        if not isinstance(sq, dict):
            continue
        params = dict(sq)
        # 연도: 추출된 연도 우선 적용
        if years and not params.get("years"):
            params["years"] = years_str
        all_queries.append({
            "query": user_query,  # retriever LLM 참조용 원문 보존
            "years": years,
            "sections": [params.get("section_h2", "")],
            "type": "structured",
            "strategy": f"llm_structured_{i+1}",
            "structured_params": params,
        })

    # 폴백: LLM 실패 또는 빈 결과 시 원문 유지
    if not all_queries:
        print(f"[QueryRewriter] LLM output empty, using original query as fallback", flush=True)
        all_queries = [{
            "query": user_query,
            "years": years,
            "sections": [],
            "type": "semantic",
            "strategy": "fallback_original",
        }]

    elapsed = time.time() - t0
    semantic_count = sum(1 for q in all_queries if q["type"] == "semantic")
    structured_count = sum(1 for q in all_queries if q["type"] == "structured")
    print(
        f"[QueryRewriter] ◀ OUTPUT: {len(all_queries)} queries "
        f"(semantic={semantic_count}, structured={structured_count}) "
        f"in {elapsed:.1f}s | reasoning: {reasoning[:120]}",
        flush=True,
    )
    for q in all_queries:
        tag = "🔵" if q["type"] == "semantic" else "🟠"
        print(f"  {tag} [{q['strategy']}] {q['query'][:80]}", flush=True)

    return {"rewritten_queries": all_queries}


def _parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    # ```json 블록
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 직접 파싱 (첫 { ~ 마지막 })
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {}
