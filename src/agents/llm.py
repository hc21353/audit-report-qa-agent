"""
llm.py - 에이전트별 백엔드를 지원하는 LLM 클라이언트 팩토리

백엔드 URL 결정 우선순위:
  1. agents.yaml의 에이전트 backend (가장 높음)
  2. runtime.yaml의 agent_backends
  3. runtime.yaml의 llm.default_backend (가장 낮음)

사용법:
  from src.agents.llm import create_llm
  llm = create_llm("orchestrator", config)
  response = llm.invoke("질문")
"""

from __future__ import annotations
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel


def resolve_backend(agent_name: str, config: "Config") -> tuple[str, str, dict]:
    """
    에이전트에 대한 (backend_url, model_name, kwargs)를 결정.

    우선순위:
      1. agents.yaml → workflow.<agent>.backend
      2. runtime.yaml → agent_backends.<agent>
      3. runtime.yaml → llm.default_backend
    """
    agents_cfg = config.agents.get("workflow", {})
    runtime_cfg = config.runtime

    agent_cfg = agents_cfg.get(agent_name, {})

    # 모델명
    model = agent_cfg.get("model") or runtime_cfg.get("llm", {}).get("default_model", "qwen2.5:14b")

    # 백엔드 URL (우선순위 적용)
    backend = agent_cfg.get("backend")  # 1순위

    if not backend:
        # 2순위
        backend = runtime_cfg.get("agent_backends", {}).get(agent_name)

    if not backend:
        # 3순위
        backend = runtime_cfg.get("llm", {}).get("default_backend", "http://localhost:11434")

    # 추가 파라미터
    kwargs = {
        "temperature": agent_cfg.get("temperature", 0.1),
    }
    timeout = runtime_cfg.get("llm", {}).get("timeout", 120)
    kwargs["timeout"] = timeout

    # 에이전트별 num_predict (JSON 생성은 짧게, 답변은 길게)
    _num_predict_map = {
        "orchestrator": 512,
        "query_rewriter": 1024,
        "retriever": 1024,
        "analyst": 2048,
    }
    kwargs["num_predict"] = agent_cfg.get("num_predict", _num_predict_map.get(agent_name, 1024))

    return backend, model, kwargs


# 에이전트별 컨텍스트 창 크기
# - orchestrator/query_rewriter: 의도 분류·쿼리 생성만 하므로 작아도 됨
# - retriever: DB 구조 컨텍스트(db_context)가 길어서 크게 필요
# - analyst: 검색 결과 전체를 받으므로 가장 크게 필요
_NUM_CTX_MAP = {
    "orchestrator": 8192,
    "query_rewriter": 8192,
    "retriever": 16384,
    "analyst": 32768,
}


def create_llm(agent_name: str, config: "Config") -> BaseChatModel:
    """
    에이전트용 LLM 인스턴스 생성.

    Args:
        agent_name: "orchestrator", "query_rewriter", "analyst" 등
        config:     Config 객체

    Returns:
        ChatOllama 인스턴스 (해당 에이전트의 백엔드/모델로 설정됨)
    """
    backend, model, kwargs = resolve_backend(agent_name, config)

    num_ctx = _NUM_CTX_MAP.get(agent_name, 16384)

    llm = ChatOllama(
        base_url=backend,
        model=model,
        temperature=kwargs.get("temperature", 0.1),
        num_ctx=num_ctx,
        num_predict=kwargs.get("num_predict", 1024),
        reasoning=False,       # qwen3 thinking 모드 비활성화
    )

    print(f"[LLM] {agent_name}: model={model}, backend={backend}, num_ctx={num_ctx}")
    return llm


def get_system_prompt(agent_name: str, config: "Config") -> str:
    """에이전트의 시스템 프롬프트 반환"""
    agents_cfg = config.agents.get("workflow", {})
    agent_cfg = agents_cfg.get(agent_name, {})
    return agent_cfg.get("system_prompt", "")
