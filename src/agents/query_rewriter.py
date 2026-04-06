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


REWRITE_PROMPT = """당신은 삼성전자 감사보고서 검색 전문가입니다.
사용자 질문을 분석하여 최적의 검색 쿼리를 생성하세요.

{db_context}

## 사용자 질문
{query}

## 분석된 의도
{intent}

## 추출된 연도
{years}

## 서브 질문 (복합 질문 분해 결과)
{sub_questions}

## 출력 형식 (JSON만 반환)
{{
  "reasoning": "검색 전략 설명 (1~2문장)",
  "semantic_queries": [
    "시맨틱 쿼리 1 - 핵심 의미를 담은 완전한 자연어 문장",
    "시맨틱 쿼리 2 - 동의어/관련 개념으로 재구성",
    "시맨틱 쿼리 3 - 다른 각도 표현 (서브 질문이 있으면 각각 포함)"
  ],
  "structured_queries": [
    {{
      "years": "쉼표 구분 연도 (예: '2022,2023'). 없으면 빈 문자열",
      "section_path_contains": "section_path 경로의 부분 문자열 (없으면 빈 문자열)",
      "chunk_type": "Table_Row | Note | Narrative | all",
      "is_consolidated": -1,
      "tags": "DB 태그 목록에서 선택 (없으면 빈 문자열). 반드시 '#' 프리픽스 포함. 예: '#이익,#수익'",
      "keyword": "본문 추가 필터 키워드 (없으면 빈 문자열)",
      "limit": 20
    }}
  ]
}}

## semantic_queries 생성 규칙
- 짧은 키워드 나열 금지 → 의미가 풍부한 완전한 문장 사용 (벡터 검색 성능 향상)
- 서브 질문이 있으면 각 서브 질문마다 별도 시맨틱 쿼리 생성
- 회계 구어체 → 표준 용어 변환:
  - "빚" → "부채", "돈" → "현금및현금성자산", "이익" → "당기순이익 또는 영업이익"
- 예시:
  - ❌ "삼성전자 2024 총자산"
  - ✅ "삼성전자 2024년 연결재무상태표의 자산총계 및 총자산 규모"

## structured_queries chunk_type 결정 규칙
- 재무 수치/표 데이터 (총자산, 영업이익, 유동자산, 부채, 자본 등):
  → chunk_type = "Table_Row"
- 주석 설명 텍스트 (회계정책 설명, 주석 내용 등):
  → chunk_type = "Note"
- 감사의견, 핵심감사사항 등 서술 텍스트:
  → chunk_type = "Narrative"
- 불확실:
  → chunk_type = "all"

## structured_queries 생성 규칙
- section_path_contains는 위 "DB 실제 섹션 구조"의 경로 일부 사용
  예: "주석" → 모든 주석, "9. 종속" → 9번 주석, "재무제표" → 재무제표 섹션
- 주석 번호는 위 목록에서 정확한 번호 확인 후 사용 (예: "30." → "30. 특수관계자")
- 여러 연도 비교 시 하나의 structured_query에 years="2022,2023,2024" 형태로 묶기
  (연도별로 별도 structured_query를 만들지 말 것)
- 불확실하면 section_path_contains는 빈 문자열로 두고 keyword만 사용
- 확실한 섹션이 있는 경우에만 structured_queries 생성 (불필요한 조회 방지)
- is_consolidated: 현재 DB는 별도재무제표(0)만 있으므로 항상 -1(전체) 사용
- tags: 반드시 위 "사용 가능한 태그" 목록에서 선택, '#' 프리픽스 필수
  목록에 없는 태그는 절대 사용하지 말 것 (0건 반환됨)

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
        db_context=db_context if db_context else "(DB 구조 정보 없음 — section_path_contains는 빈 문자열로 설정)",
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
            "sections": [params.get("section_path_contains", "")],
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
