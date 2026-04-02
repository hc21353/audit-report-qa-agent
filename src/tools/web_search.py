"""
web_search.py - DuckDuckGo 웹 검색 Tool

감사보고서 DB에 없는 최신 정보(회계기준 개정, 규정 해석 등)를 보완하기 위해 사용.
API 키 불필요 (DuckDuckGo 무료 API 사용).

의존성:
  pip install duckduckgo-search
"""

import json
from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """인터넷에서 최신 정보를 검색합니다.

    감사보고서 DB에 없는 정보(최신 회계기준, K-IFRS 해석, 규정 변경 등)를
    보완할 때 사용하세요. 내부 DB 검색으로 답하기 어려운 경우 활용하세요.

    Args:
        query:       검색 쿼리 (한국어 또는 영어)
        max_results: 최대 결과 수 (기본 5, 최대 10)

    Returns:
        검색 결과 JSON [{title, url, snippet}]
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps(
                {"error": "ddgs 패키지가 설치되지 않았습니다. pip install ddgs"},
                ensure_ascii=False,
            )

    max_results = min(max_results, 10)

    print(f"[WebSearch] ▶ query='{query}', max_results={max_results}", flush=True)

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, region="kr-kr", max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as e:
        print(f"[WebSearch] Error: {e}", flush=True)
        return json.dumps({"error": f"웹 검색 실패: {e}"}, ensure_ascii=False)

    print(f"[WebSearch] ◀ {len(results)} results", flush=True)
    for r in results[:3]:
        print(f"  {r['title'][:60]} | {r['url'][:60]}", flush=True)

    return json.dumps(results, ensure_ascii=False, indent=2)
