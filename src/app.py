"""
app.py - Streamlit 채팅 인터페이스

사용법:
  streamlit run src/app.py
"""

import sys
import os
# 프로젝트 루트(src의 부모)를 sys.path에 추가 — 어느 디렉토리에서 실행해도 동작
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pathlib import Path

from src.config import load_config
from src.db import AuditDB
from src.agents.graph import build_graph, run_query_stream


# ─── 초기화 ─────────────────────────────────────────────────

@st.cache_resource
def init_system(model_overrides: tuple = ()):
    """
    시스템 초기화 (한 번만 실행).
    model_overrides: (("orchestrator", "model-name"), ...) 형태의 튜플.
    값이 바뀌면 캐시가 버스트되어 그래프가 재빌드된다.
    """
    config = load_config()

    # 에이전트별 모델 오버라이드 적용
    workflow = config.agents.setdefault("workflow", {})
    for agent, model in model_overrides:
        if agent in workflow and isinstance(workflow[agent], dict):
            workflow[agent]["model"] = model

    db = AuditDB(config.db_path)

    # ChromaDB + KoE5 로드
    vectorstore = None
    vector_error = None
    try:
        import chromadb
        from chromadb.config import Settings
        from src.build_vectordb import KoE5Embedder

        vector_dir = config.runtime.get("data", {}).get(
            "vector_store_dir", "./data/vectorstore/chroma"
        )
        client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("samsung_audit")
        embedder = KoE5Embedder()
        vectorstore = (collection, embedder)
    except Exception as e:
        vector_error = str(e)
        vectorstore = None

    graph = build_graph(config, db=db, vector_store=vectorstore)
    return config, db, vectorstore, graph, vector_error


def _get_db_stats(db: AuditDB) -> dict:
    """DB 기본 통계 조회"""
    try:
        row = db.conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        total = row["cnt"] if row else 0
        sections = db.list_sections(level=1)
        top_sections = sorted(set(s["section"] for s in sections))
        return {"total_chunks": total, "sections": top_sections}
    except Exception:
        return {"total_chunks": 0, "sections": []}


# ─── 노드 레이블 / 아이콘 ──────────────────────────────────

NODE_LABELS = {
    "orchestrator":   ("🔍", "질문 의도 분석"),
    "query_rewriter": ("✏️", "검색 쿼리 최적화"),
    "retriever":      ("📂", "문서 검색"),
    "analyst":        ("💡", "답변 생성"),
}


def _render_node_step(node: str, update: dict, retriever_count: int = 1):
    """노드별 진행 상황을 Streamlit 위젯으로 렌더링"""
    icon, label = NODE_LABELS.get(node, ("⚙️", node))

    if node == "retriever" and retriever_count > 1:
        icon, label = "🔄", f"추가 검색 #{retriever_count}"

    if node == "orchestrator":
        intent = update.get("intent", "")
        years = update.get("extracted_years", [])
        sections = update.get("extracted_sections", [])
        sub_qs = update.get("sub_questions", [])

        intent_label = {
            "simple_lookup": "단순 조회",
            "comparison": "비교 분석",
            "trend": "추이 분석",
            "calculation": "계산",
            "general": "일반",
        }.get(intent, intent)

        st.markdown(f"{icon} **{label}** — 의도: `{intent_label}` | 연도: `{years}`")
        if sections:
            st.caption(f"   관련 섹션: {', '.join(sections)}")
        if sub_qs:
            with st.expander("분해된 서브 질문"):
                for i, q in enumerate(sub_qs, 1):
                    st.caption(f"{i}. {q}")

    elif node == "query_rewriter":
        queries = update.get("rewritten_queries", [])
        sem = [q for q in queries if q.get("type") == "semantic"]
        stru = [q for q in queries if q.get("type") == "structured"]
        st.markdown(
            f"{icon} **{label}** — 총 {len(queries)}개 쿼리 "
            f"(시맨틱 {len(sem)}개 · 구조 {len(stru)}개)"
        )
        if queries:
            with st.expander("생성된 쿼리 목록"):
                for q in queries:
                    tag = "🔵" if q.get("type") == "semantic" else "🟠"
                    strategy = q.get("strategy", "")
                    st.caption(f"{tag} [{strategy}] {q.get('query', '')}")
                    if q.get("structured_params"):
                        st.json(q["structured_params"])

    elif node == "retriever":
        results = update.get("search_results", [])
        csv_count = len(update.get("csv_data", {}))
        st.markdown(
            f"{icon} **{label}** — 검색 결과 **{len(results)}**개 · CSV **{csv_count}**개"
        )
        if results:
            with st.expander("검색된 소스 (상위 5개)"):
                seen = []
                for r in results[:5]:
                    yr = r.get("year", "?")
                    sec = r.get("section_path", "?")
                    score = r.get("score", 0)
                    stype = r.get("search_type", "")
                    entry = f"📄 {yr}년 | {sec} | score={score:.3f} | {stype}"
                    if entry not in seen:
                        st.caption(entry)
                        seen.append(entry)

    elif node == "analyst":
        answer = update.get("answer", "")
        needs = update.get("needs_more_search", False)
        calcs = update.get("calculations", [])
        sources = update.get("sources", [])
        status_icon = "⏳ 추가 검색 필요" if needs else "✅ 완료"
        st.markdown(
            f"{icon} **{label}** — 답변 {len(answer)}자 | "
            f"출처 {len(sources)}개 | 계산 {len(calcs)}개 | {status_icon}"
        )

    else:
        st.markdown(f"{icon} **{label}**")


# ─── 대화 히스토리 → 쿼리 컨텍스트 ─────────────────────────

def _build_query_with_history(prompt: str, messages: list, max_turns: int = 3) -> str:
    """
    최근 N턴 대화를 현재 질문 앞에 컨텍스트로 추가.
    멀티턴 구현: app.py에서만 수정, 다른 에이전트 파일 변경 불필요.
    """
    # 현재 방금 추가된 user 메시지 제외한 이전 대화
    prior = [m for m in messages if m.get("role") in ("user", "assistant")]
    # 방금 추가한 현재 user 메시지(마지막)는 prompt로 받으므로 제외
    if prior and prior[-1]["role"] == "user" and prior[-1]["content"] == prompt:
        prior = prior[:-1]

    if not prior:
        return prompt

    # 최근 max_turns 턴(user+assistant 쌍)
    recent = prior[-(max_turns * 2):]
    history_lines = []
    for m in recent:
        role = "사용자" if m["role"] == "user" else "어시스턴트"
        # 어시스턴트 답변은 앞 200자만 (토큰 절약)
        content = m["content"][:200] + "..." if m["role"] == "assistant" and len(m["content"]) > 200 else m["content"]
        history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines)
    return f"[이전 대화 참고]\n{history_str}\n\n[현재 질문]\n{prompt}"


# ─── 페이지 설정 ─────────────────────────────────────────────

def main():
    config = load_config()
    ui_config = config.app.get("interface", {})

    st.set_page_config(
        page_title=ui_config.get("title", "감사보고서 QA"),
        page_icon="📊",
        layout="wide",
    )

    st.title(f"📊 {ui_config.get('title', '감사보고서 QA 시스템')}")

    # 모델 오버라이드 세션 초기화
    if "model_overrides" not in st.session_state:
        st.session_state.model_overrides = {}

    # 시스템 초기화 (model_overrides가 캐시 키 역할)
    overrides_tuple = tuple(sorted(st.session_state.model_overrides.items()))
    try:
        cfg, db, vectorstore, graph, vector_error = init_system(overrides_tuple)
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

    # ─── 사이드바 ─────────────────────────────────────────────
    with st.sidebar:
        st.header("설정")

        # 연도 필터 (실제 DB 데이터 기반)
        years = cfg.years
        selected_years = st.multiselect(
            "연도 필터",
            options=years,
            default=[],
            help="비워두면 전체 연도 검색. 선택 시 질문에 연도 힌트로 반영됩니다.",
        )

        # 섹션 필터 (실제 DB에서 조회)
        db_stats = _get_db_stats(db)
        real_sections = ["전체"] + db_stats["sections"]
        selected_section = st.selectbox(
            "섹션 필터",
            real_sections,
            help="선택 시 질문에 섹션 힌트로 반영됩니다.",
        )

        st.divider()

        # ─── 모델 설정 ──────────────────────────────────────
        st.caption("**모델 설정**")

        # 선택 가능한 모델 목록: runtime.yaml + 현재 에이전트 모델
        available_in_yaml = list(
            config.runtime.get("llm", {}).get("available_models", {}).keys()
        )
        workflow = cfg.agents.get("workflow", {})
        current_agent_models = [
            info.get("model", "")
            for info in workflow.values()
            if isinstance(info, dict) and info.get("model")
        ]
        # 중복 없이 현재 사용 모델을 맨 앞에
        all_models: list[str] = []
        for m in current_agent_models + available_in_yaml:
            if m and m not in all_models:
                all_models.append(m)

        AGENT_LABELS = {
            "orchestrator":   "Orchestrator  (의도 분석)",
            "query_rewriter": "Query Rewriter (쿼리 최적화)",
            "retriever":      "Retriever      (문서 검색) ★",
            "analyst":        "Analyst        (답변 생성) ★",
        }

        with st.expander("에이전트별 모델 변경", expanded=False):
            pending: dict[str, str] = {}
            for agent, label in AGENT_LABELS.items():
                current = (
                    st.session_state.model_overrides.get(agent)
                    or workflow.get(agent, {}).get("model", all_models[0] if all_models else "")
                )
                idx = all_models.index(current) if current in all_models else 0
                selected = st.selectbox(label, all_models, index=idx, key=f"model_sel_{agent}")
                pending[agent] = selected

            if st.button("모델 적용 (그래프 재초기화)", use_container_width=True, type="primary"):
                st.session_state.model_overrides = pending
                init_system.clear()
                st.rerun()

        # 현재 적용된 모델 요약
        active_models = {
            agent: st.session_state.model_overrides.get(agent) or workflow.get(agent, {}).get("model", "?")
            for agent in AGENT_LABELS
        }
        unique_active = set(active_models.values())
        if len(unique_active) == 1:
            st.caption(f"LLM: `{next(iter(unique_active))}`")
        else:
            for agent, model in active_models.items():
                st.caption(f"`{agent}`: {model}")

        # 벡터 스토어 상태
        if vectorstore is not None:
            coll, _ = vectorstore
            vec_count = coll.count()
            st.caption(f"벡터 DB: ✅ {vec_count:,}개 벡터 (KoE5)")
        else:
            st.caption(f"벡터 DB: ⚠️ 비활성 (BM25만 사용)")
            if vector_error:
                with st.expander("오류 상세"):
                    st.caption(vector_error)

        # DB 통계
        st.caption(f"청크 수: {db_stats['total_chunks']:,}개")
        st.caption(f"대상 연도: {cfg.years[0]}~{cfg.years[-1]}") if cfg.years else None

        st.divider()
        show_debug = st.toggle("디버그 정보 표시", value=False)

        # 대화 초기화
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ─── 채팅 ────────────────────────────────────────────────

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 메시지 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_debug and "debug" in msg:
                with st.expander("🔍 디버그 정보"):
                    st.json(msg["debug"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요 (예: 2024년 삼성전자 총자산은?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # ─── 필터 힌트 + 멀티턴 컨텍스트 조합 ──────────────
        effective_query = prompt

        # 사이드바 필터를 질문 앞에 힌트로 추가
        filter_hints = []
        if selected_years:
            filter_hints.append(f"연도 필터: {selected_years}")
        if selected_section and selected_section != "전체":
            filter_hints.append(f"섹션 필터: {selected_section}")
        if filter_hints:
            effective_query = "[검색 조건]\n" + "\n".join(filter_hints) + "\n\n" + effective_query

        # 멀티턴: 이전 대화 컨텍스트 prepend
        effective_query = _build_query_with_history(
            effective_query, st.session_state.messages
        )

        # ─── 응답 생성 ────────────────────────────────────────
        with st.chat_message("assistant"):
            result_state = {}
            retriever_count = 0

            with st.status("분석 중...", expanded=True) as status:
                for event in run_query_stream(graph, effective_query):
                    node = event["node"]
                    update = event["update"]
                    state = event["state"]

                    if node == "__end__":
                        result_state = state
                        break

                    icon, label = NODE_LABELS.get(node, ("⚙️", node))

                    if node == "retriever":
                        retriever_count += 1

                    _render_node_step(node, update, retriever_count)
                    status.update(label=f"{icon} {label} 진행 중...")

                status.update(label="✅ 분석 완료", state="complete")

            # ─── 최종 답변 출력 ──────────────────────────────
            answer = result_state.get("answer", "답변을 생성할 수 없습니다.")
            st.markdown(answer)

            # 소스 표시
            sources = result_state.get("sources", [])
            if sources and ui_config.get("features", {}).get("show_sources", True):
                with st.expander(f"📚 출처 ({len(sources)}개)"):
                    for s in sources:
                        st.caption(
                            f"연도: {s.get('year', '?')} | "
                            f"섹션: {s.get('section_path', '?')} | "
                            f"검색: {s.get('search_type', '?')}"
                        )

            # 계산 결과
            calculations = result_state.get("calculations", [])
            if calculations:
                with st.expander("🔢 계산 결과"):
                    for calc in calculations:
                        st.write(
                            f"**{calc.get('label', '계산')}**: "
                            f"{calc.get('result', '?')} "
                            f"({calc.get('expression', '')})"
                        )

            # 쿼리 재작성 (디버그)
            rewritten = result_state.get("rewritten_queries", [])
            if show_debug and rewritten:
                with st.expander("✏️ 재작성된 검색 쿼리"):
                    for q in rewritten:
                        tag = "🔵 시맨틱" if q.get("type") == "semantic" else "🟠 구조"
                        st.caption(f"{tag} [{q.get('strategy','')}] {q.get('query','')}")
                        if q.get("structured_params"):
                            st.json(q["structured_params"])

            # 디버그 정보
            debug_info = {
                "intent": result_state.get("intent", ""),
                "iterations": result_state.get("iteration", 0),
                "errors": result_state.get("errors", []),
                "sources_count": len(sources),
                "rewritten_queries": rewritten,
                "effective_query": effective_query,
            }

            if show_debug:
                with st.expander("🔍 전체 디버그 정보"):
                    st.json(debug_info)

            # 히스토리 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "debug": debug_info,
            })

    # ─── 예시 질문 ───────────────────────────────────────────

    if not st.session_state.messages:
        st.markdown("### 💡 예시 질문")
        cols = st.columns(2)

        examples = [
            "2024년 삼성전자의 총자산은 얼마인가?",
            "2024년 핵심감사사항은 무엇인가?",
            "2024년 유동비율을 계산해줘",
            "감사의견은 무엇인가?",
        ]

        for i, example in enumerate(examples):
            col = cols[i % 2]
            if col.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


if __name__ == "__main__":
    main()
