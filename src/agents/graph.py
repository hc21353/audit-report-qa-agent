"""
graph.py - LangGraph 상태 그래프 조립

전체 흐름:
  orchestrator → (query_rewriter) → retriever → analyst → (retriever 루프백)

사용법:
  from src.agents.graph import build_graph, run_query
  graph = build_graph(config)
  result = run_query(graph, "2024년 총자산은?")
"""

from __future__ import annotations
from typing import TypedDict, Annotated, Any
import operator

from langgraph.graph import StateGraph, END

from src.config import Config
from src.agents.llm import create_llm, get_system_prompt
from src.agents.orchestrator import orchestrator_node, orchestrator_router
from src.agents.query_rewriter import query_rewriter_node
from src.agents.retriever import retriever_node
from src.agents.analyst import analyst_node, analyst_router
from src.agents.db_context import build_db_context
from src.agents.state import initial_state


# ─── LangGraph State 타입 ────────────────────────────────────

class GraphState(TypedDict):
    # 입력
    user_query: str

    # orchestrator
    intent: str
    extracted_years: list[int]
    extracted_sections: list[str]
    sub_questions: list[str]

    # query_rewriter
    rewritten_queries: list[dict]

    # retriever
    search_results: list[dict]
    csv_data: dict

    # analyst
    answer: str
    sources: list[dict]
    calculations: list[dict]
    charts: list[dict]
    needs_more_search: bool
    additional_query: str

    # meta
    iteration: int
    max_iterations: int
    errors: list[str]


# ─── 그래프 빌드 ─────────────────────────────────────────────

def build_graph(config: Config, db=None, vector_store=None) -> StateGraph:
    """
    LangGraph 상태 그래프를 조립합니다.

    Args:
        config:       Config 객체 (agents.yaml, runtime.yaml 포함)
        db:           AuditDB 인스턴스 (구조 검색용)
        vector_store: FAISS/Chroma 벡터스토어 인스턴스

    Returns:
        컴파일된 StateGraph
    """
    # LLM 인스턴스 생성 (에이전트별 백엔드)
    orchestrator_llm = create_llm("orchestrator", config)
    rewriter_llm = create_llm("query_rewriter", config)
    retriever_llm = create_llm("retriever", config)  # 검색 계획 수립용
    analyst_llm = create_llm("analyst", config)

    # Tool 초기화
    from src.tools.hybrid_search import hybrid_search as hs_tool, init_hybrid_search
    from src.tools.structured_query import (
        structured_query as sq_tool, list_available_sections as las_tool,
        init_structured_query,
    )
    from src.tools.csv_reader_tool import csv_reader_tool as cr_tool, init_csv_reader
    from src.tools.calculator import calculator as calc_tool
    from src.tools.chart_generator import chart_generator as chart_tool

    # Tool에 DB/벡터스토어 참조 설정
    init_hybrid_search(vector_store, db)
    if db:
        init_structured_query(db)

    parsed_md_dir = config.runtime.get("data", {}).get("parsed_md_dir", "./data/parsed_md")
    init_csv_reader(parsed_md_dir)

    # DB 섹션 구조 컨텍스트 (초기화 시 한 번만 조회, 이후 클로저로 재사용)
    db_context_str = build_db_context(db)
    if db_context_str:
        print(f"[Graph] DB context built ({len(db_context_str)} chars)")

    retriever_tools = {
        "hybrid_search": hs_tool,
        "structured_query": sq_tool,
        "list_available_sections": las_tool,
        "csv_reader": cr_tool,
    }
    analyst_tools = {
        "calculator": calc_tool,
        "chart_generator": chart_tool,
        "csv_reader": cr_tool,
    }

    # 시스템 프롬프트
    orch_prompt = get_system_prompt("orchestrator", config)
    rewriter_prompt = get_system_prompt("query_rewriter", config)
    analyst_prompt = get_system_prompt("analyst", config)

    # ─── 노드 함수 래핑 (클로저로 LLM/tools 주입) ────────────

    def _orchestrator(state: GraphState) -> dict:
        return orchestrator_node(state, orchestrator_llm, system_prompt=orch_prompt)

    def _query_rewriter(state: GraphState) -> dict:
        return query_rewriter_node(state, rewriter_llm, system_prompt=rewriter_prompt)

    def _retriever(state: GraphState) -> dict:
        return retriever_node(state, retriever_tools, llm=retriever_llm, db_context=db_context_str)

    def _analyst(state: GraphState) -> dict:
        return analyst_node(state, analyst_llm, analyst_tools, system_prompt=analyst_prompt)

    # ─── 그래프 조립 ─────────────────────────────────────────

    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("orchestrator", _orchestrator)
    graph.add_node("query_rewriter", _query_rewriter)
    graph.add_node("retriever", _retriever)
    graph.add_node("analyst", _analyst)

    # 엔트리포인트
    graph.set_entry_point("orchestrator")

    # 엣지: orchestrator → (retriever | query_rewriter)
    graph.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "retriever": "retriever",
            "query_rewriter": "query_rewriter",
        }
    )

    # 엣지: query_rewriter → retriever
    graph.add_edge("query_rewriter", "retriever")

    # 엣지: retriever → analyst
    graph.add_edge("retriever", "analyst")

    # 엣지: analyst → (retriever | END)
    graph.add_conditional_edges(
        "analyst",
        analyst_router,
        {
            "need_more_search": "retriever",
            "done": END,
        }
    )

    compiled = graph.compile()
    print("[Graph] Built successfully")
    return compiled


# ─── 실행 ───────────────────────────────────────────────────

def run_query(graph, query: str, max_iterations: int = 5) -> dict:
    """
    질문을 실행하고 결과를 반환합니다.

    Args:
        graph: 컴파일된 StateGraph
        query: 사용자 질문

    Returns:
        최종 상태 dict (answer, sources, calculations, errors 등)
    """
    state = initial_state(query)
    state["max_iterations"] = max_iterations

    result = graph.invoke(state)

    return {
        "query": query,
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "sources": result.get("sources", []),
        "calculations": result.get("calculations", []),
        "charts": result.get("charts", []),
        "iterations": result.get("iteration", 0),
        "errors": result.get("errors", []),
    }


# ─── CLI 테스트 ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.config import load_config
    from src.db import AuditDB

    config = load_config()
    db = AuditDB(config.db_path)

    # 벡터스토어는 None (아직 미구현)
    graph = build_graph(config, db=db, vector_store=None)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "2024년 삼성전자 총자산은?"
    print(f"\nQuery: {query}\n")

    result = run_query(graph, query)
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
    print(f"Iterations: {result['iterations']}")
    if result["errors"]:
        print(f"Errors: {result['errors']}")

    db.close()
