"""
graph.py - LangGraph 상태 그래프 조립

전체 흐름:
  orchestrator → (query_rewriter) → retriever → analyst → (retriever 루프백)

사용법:
  from src.agents.graph import build_graph, run_query, run_query_stream
  graph = build_graph(config)
  result = run_query(graph, "2024년 총자산은?")
  for event in run_query_stream(graph, "2024년 총자산은?"):
      print(event["node"], event["update"])
"""

from __future__ import annotations
import time
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from src.config import Config
from src.agents.llm import create_llm, get_system_prompt
from src.agents.orchestrator import orchestrator_node, orchestrator_router
from src.agents.query_rewriter import query_rewriter_node
from src.agents.retriever import retriever_node
from src.agents.analyst import analyst_node, analyst_router
from src.agents.db_context import build_db_context
from src.agents.state import initial_state
from src.agents.tracer import AgentTracer, set_tracer, get_tracer


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
    retriever_llm = create_llm("retriever", config)
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
    from src.tools.web_search import web_search as ws_tool
    from src.tools.db_fetch import get_full_content as gfc_tool, init_db_fetch

    init_hybrid_search(vector_store, db)
    if db:
        init_structured_query(db)
        init_db_fetch(db)

    parsed_md_dir = config.runtime.get("data", {}).get("parsed_md_dir", "./data/parsed_md")
    init_csv_reader(parsed_md_dir)

    # DB 섹션 구조 컨텍스트 (초기화 시 한 번만 조회)
    db_context_str = build_db_context(db)
    if db_context_str:
        print(f"[Graph] DB context built ({len(db_context_str)} chars)")

    retriever_tools = {
        "hybrid_search": hs_tool,
        "structured_query": sq_tool,
        "list_available_sections": las_tool,
        "csv_reader": cr_tool,
        "web_search": ws_tool,
        "get_full_content": gfc_tool,
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

    # ─── 노드 함수 래핑 (클로저로 LLM/tools/db_context 주입) ──

    def _orchestrator(state: GraphState) -> dict:
        tracer = get_tracer()
        if tracer:
            tracer.log_node_start("orchestrator", dict(state))
        t0 = time.time()
        result = orchestrator_node(state, orchestrator_llm, system_prompt=orch_prompt)
        if tracer:
            tracer.log_node_end("orchestrator", result, round(time.time() - t0, 2))
        return result

    def _query_rewriter(state: GraphState) -> dict:
        tracer = get_tracer()
        if tracer:
            tracer.log_node_start("query_rewriter", dict(state))
        t0 = time.time()
        result = query_rewriter_node(
            state, rewriter_llm,
            system_prompt=rewriter_prompt,
            db_context=db_context_str,
        )
        if tracer:
            tracer.log_node_end("query_rewriter", result, round(time.time() - t0, 2))
        return result

    def _retriever(state: GraphState) -> dict:
        tracer = get_tracer()
        if tracer:
            tracer.log_node_start("retriever", dict(state))
        t0 = time.time()
        result = retriever_node(state, retriever_tools, llm=retriever_llm, db_context=db_context_str)
        if tracer:
            n_results = len(result.get("search_results", []))
            tracer.log_node_end("retriever", {"search_results_count": n_results}, round(time.time() - t0, 2))
        return result

    def _analyst(state: GraphState) -> dict:
        tracer = get_tracer()
        if tracer:
            tracer.log_node_start("analyst", dict(state))
        t0 = time.time()
        result = analyst_node(state, analyst_llm, analyst_tools, system_prompt=analyst_prompt)
        if tracer:
            tracer.log_node_end("analyst", result, round(time.time() - t0, 2))
        return result

    # ─── 그래프 조립 ─────────────────────────────────────────

    graph = StateGraph(GraphState)

    graph.add_node("orchestrator", _orchestrator)
    graph.add_node("query_rewriter", _query_rewriter)
    graph.add_node("retriever", _retriever)
    graph.add_node("analyst", _analyst)

    graph.set_entry_point("orchestrator")

    # orchestrator → (retriever | query_rewriter)
    graph.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "retriever": "retriever",
            "query_rewriter": "query_rewriter",
        }
    )

    # query_rewriter → retriever
    graph.add_edge("query_rewriter", "retriever")

    # retriever → analyst
    graph.add_edge("retriever", "analyst")

    # analyst → (retriever | END)
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


# ─── 실행: 동기 ─────────────────────────────────────────────

def run_query(
    graph,
    query: str,
    max_iterations: int = 5,
    enable_trace: bool = True,
    save_trace: bool = True,
) -> dict:
    """
    질문을 실행하고 결과를 반환합니다. (동기)

    Args:
        enable_trace: True이면 AgentTracer를 활성화 (기본 True)
        save_trace:   True이면 실행 후 logs/ 에 trace 파일 저장

    Returns:
        {query, answer, intent, sources, calculations, charts, iterations, errors, trace?}
    """
    state = initial_state(query)
    state["max_iterations"] = max_iterations

    tracer = None
    if enable_trace:
        tracer = AgentTracer(query=query)
        set_tracer(tracer)

    try:
        result = graph.invoke(state)
    finally:
        if tracer:
            set_tracer(None)

    out = {
        "query": query,
        "answer": result.get("answer", ""),
        "intent": result.get("intent", ""),
        "sources": result.get("sources", []),
        "calculations": result.get("calculations", []),
        "charts": result.get("charts", []),
        "iterations": result.get("iteration", 0),
        "errors": result.get("errors", []),
    }

    if tracer:
        tracer.print_summary()
        if save_trace:
            out["trace_file"] = str(tracer.save())
        out["trace_summary"] = tracer.summary()

    return out


# ─── 실행: 스트리밍 (Streamlit 진행 표시용) ──────────────────

def run_query_stream(graph, query: str, max_iterations: int = 5, enable_trace: bool = True):
    """
    제너레이터: 각 노드 실행 후 진행 상황을 yield.

    Yields:
        {
          "node": str,          # 노드 이름 ("orchestrator", "retriever" 등)
          "update": dict,       # 해당 노드의 state 업데이트
          "state": dict,        # 누적된 전체 상태
        }

    마지막 이벤트:
        {"node": "__end__", "update": {}, "state": <최종 상태>}
    """
    state = initial_state(query)
    state["max_iterations"] = max_iterations

    tracer = None
    if enable_trace:
        tracer = AgentTracer(query=query)
        set_tracer(tracer)

    current = dict(state)

    for event in graph.stream(current, stream_mode="updates"):
        node_name = list(event.keys())[0]
        update = event[node_name]
        if isinstance(update, dict):
            current.update(update)
        yield {
            "node": node_name,
            "update": update if isinstance(update, dict) else {},
            "state": dict(current),
        }

    if tracer:
        set_tracer(None)
        tracer.save()

    yield {
        "node": "__end__",
        "update": {},
        "state": dict(current),
    }


# ─── CLI 테스트 ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.config import load_config
    from src.db import AuditDB

    config = load_config()
    db = AuditDB(config.db_path)

    # 벡터 스토어 로드
    vector_store = None
    try:
        from src.build_index import load_langchain_faiss, create_embedding_wrapper
        embed_name = config.active_embedding
        chunking_strategy = config.active_chunking
        index_dir = config.runtime.get("data", {}).get("vector_index_dir", "./data/vector_index")
        index_name = f"faiss/{embed_name}_{chunking_strategy}"

        embedding_model = create_embedding_wrapper(config, embed_name)
        vector_store = load_langchain_faiss(index_dir, index_name, embedding_model)
        if vector_store:
            print(f"[CLI] Vector store loaded: {index_name}")
        else:
            print(f"[CLI] Vector store not found, running without vector search")
    except Exception as e:
        print(f"[CLI] Vector store load failed: {e}, running without vector search")

    graph = build_graph(config, db=db, vector_store=vector_store)

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
