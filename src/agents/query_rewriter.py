"""
query_rewriter.py - 검색 쿼리 최적화 에이전트 노드

역할:
  - 사용자 질문을 벡터 검색 + 구조 검색에 최적화
  - 최대 3개의 다양한 쿼리 생성
  - 회계 용어 변환
"""

import json
import re
from langchain_core.messages import SystemMessage, HumanMessage


# 회계 용어 사전 (일반어 → 공식 용어)
ACCOUNTING_DICT = {
    "빚": "부채",
    "돈": "현금및현금성자산",
    "자산": "자산총계",
    "빌린돈": "차입금",
    "이익": "당기순이익",
    "매출": "매출액",
    "영업이익": "영업이익",
    "투자": "투자활동",
    "배당": "배당금",
    "감가상각": "감가상각비",
}


def query_rewriter_node(state: dict, llm) -> dict:
    """
    LangGraph 노드: 검색에 최적화된 쿼리 생성.

    Returns:
        rewritten_queries: [{query, years, sections, strategy}]
    """
    user_query = state["user_query"]
    intent = state.get("intent", "general")
    years = state.get("extracted_years", [])
    sub_questions = state.get("sub_questions", [])

    # 서브 질문이 있으면 각각에 대해 쿼리 생성
    questions = sub_questions if sub_questions else [user_query]

    all_queries = []

    for question in questions:
        # 전략 1: 원문 유지
        all_queries.append({
            "query": question,
            "years": years,
            "sections": state.get("extracted_sections", []),
            "strategy": "original",
        })

        # 전략 2: 규칙 기반 키워드 추출 + 회계 용어 변환
        keyword_query = _extract_keywords(question)
        if keyword_query != question:
            all_queries.append({
                "query": keyword_query,
                "years": years,
                "sections": [],
                "strategy": "keyword",
            })

        accounting_query = _convert_accounting_terms(question)
        if accounting_query != question:
            all_queries.append({
                "query": accounting_query,
                "years": years,
                "sections": [],
                "strategy": "accounting_term",
            })

    # LLM 기반 추가 쿼리 (복잡한 질문일 때)
    if intent in ("comparison", "trend", "calculation") and llm:
        try:
            llm_queries = _llm_rewrite(user_query, intent, years, llm, state)
            all_queries.extend(llm_queries)
        except Exception as e:
            state.setdefault("errors", []).append(f"QueryRewriter LLM error: {e}")

    # 중복 제거
    seen = set()
    unique = []
    for q in all_queries:
        key = q["query"]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    # 최대 개수 제한
    max_rewrites = 6
    return {"rewritten_queries": unique[:max_rewrites]}


def _extract_keywords(query: str) -> str:
    """불용어 제거 후 핵심 키워드만 추출"""
    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "에서",
                 "로", "으로", "와", "과", "도", "만", "까지", "부터",
                 "해줘", "해주세요", "알려줘", "설명해줘", "무엇", "어떻게",
                 "얼마", "뭐야", "뭐", "좀", "것", "수", "때"}

    # 연도 보존
    years = re.findall(r"20[0-2]\d", query)

    words = re.findall(r"[\w]+", query)
    keywords = [w for w in words if w not in stopwords and not re.match(r"^\d{4}$", w)]

    result = " ".join(keywords)
    if years:
        result = " ".join(years) + " " + result

    return result.strip()


def _convert_accounting_terms(query: str) -> str:
    """일반 용어를 회계 공식 용어로 변환"""
    result = query
    for general, formal in ACCOUNTING_DICT.items():
        result = result.replace(general, formal)
    return result


def _llm_rewrite(query: str, intent: str, years: list, llm, state: dict) -> list[dict]:
    """LLM 기반 쿼리 재작성"""
    prompt = f"""다음 질문을 감사보고서 검색에 최적화된 쿼리 3개로 변환하세요.
JSON 배열만 반환하세요.

질문: {query}
의도: {intent}
연도: {years}

형식: [{{"query": "...", "years": [...], "sections": [...], "strategy": "llm_rewrite"}}]"""

    response = llm.invoke([
        SystemMessage(content=state.get("_system_prompt", "")),
        HumanMessage(content=prompt),
    ])

    try:
        text = response.content
        start = text.index("[")
        end = text.rindex("]") + 1
        queries = json.loads(text[start:end])
        for q in queries:
            q["strategy"] = "llm_rewrite"
        return queries
    except (ValueError, json.JSONDecodeError):
        return []
