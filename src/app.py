"""
app.py - Streamlit 채팅 인터페이스

사용법:
  streamlit run src/app.py
"""

import streamlit as st
import json
from pathlib import Path

from src.config import load_config
from src.db import AuditDB
from src.build_index import build_index, load_langchain_faiss, create_embedding_wrapper
from src.agents.graph import build_graph, run_query, run_query_stream


# ─── 초기화 ─────────────────────────────────────────────────

@st.cache_resource
def init_system():
    """시스템 초기화 (한 번만 실행)"""
    config = load_config()
    db = AuditDB(config.db_path)

    try:
        vectorstore = build_index(config, db)
    except Exception as e:
        st.warning(f"벡터 인덱스 로드 실패: {e}. 구조 검색만 사용합니다.")
        vectorstore = None

    graph = build_graph(config, db=db, vector_store=vectorstore)
    return config, db, vectorstore, graph


# ─── 노드 레이블 / 아이콘 ──────────────────────────────────

NODE_LABELS = {
    "orchestrator":   ("🔍", "질문 의도 분석"),
    "query_rewriter": ("✏️", "검색 쿼리 최적화"),
    "retriever":      ("📂", "문서 검색"),
    "analyst":        ("💡", "답변 생성"),
}


def _node_detail(node: str, update: dict) -> str:
    """노드별 진행 상세 문자열"""
    if node == "orchestrator":
        intent = update.get("intent", "")
        years = update.get("extracted_years", [])
        return f"intent=**{intent}**, years={years}"
    elif node == "query_rewriter":
        queries = update.get("rewritten_queries", [])
        sem = sum(1 for q in queries if q.get("type") == "semantic")
        stru = sum(1 for q in queries if q.get("type") == "structured")
        return f"총 {len(queries)}개 쿼리 (시맨틱 {sem}개 · 구조 {stru}개)"
    elif node == "retriever":
        n = len(update.get("search_results", []))
        csv = len(update.get("csv_data", {}))
        return f"검색 결과 **{n}**개 · CSV **{csv}**개"
    elif node == "analyst":
        answer = update.get("answer", "")
        needs = update.get("needs_more_search", False)
        status = "추가 검색 필요" if needs else "완료"
        return f"답변 {len(answer)}자 · {status}"
    return ""


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

    # 사이드바
    with st.sidebar:
        st.header("설정")

        years = config.years
        selected_years = st.multiselect(
            "연도 필터",
            options=years,
            default=[],
            help="비워두면 전체 연도 검색",
        )

        sections = ["전체", "독립된 감사인의 감사보고서", "재무제표", "주석",
                    "내부회계관리제도 감사보고서", "외부감사 실시내용"]
        selected_section = st.selectbox("섹션 필터", sections)

        st.divider()

        st.caption("시스템 정보")
        st.caption(f"임베딩: {config.active_embedding}")
        st.caption(f"청킹: {config.active_chunking}")
        st.caption(f"LLM: {config.runtime.get('llm', {}).get('default_model', '?')}")

        show_debug = st.toggle("디버그 정보 표시", value=False)

    # 시스템 초기화
    try:
        cfg, db, vectorstore, graph = init_system()
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

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

        # 응답 생성
        with st.chat_message("assistant"):
            # ─── 진행 과정 표시 (st.status) ──────────────────
            progress_log = []   # 완료된 스텝 기록
            result_state = {}
            retriever_count = 0

            with st.status("분석 중...", expanded=True) as status:
                for event in run_query_stream(graph, prompt):
                    node = event["node"]
                    update = event["update"]
                    state = event["state"]

                    if node == "__end__":
                        result_state = state
                        break

                    icon, label = NODE_LABELS.get(node, ("⚙️", node))
                    detail = _node_detail(node, update)

                    # 반복 retriever 구분
                    if node == "retriever":
                        retriever_count += 1
                        if retriever_count > 1:
                            icon, label = "🔄", f"추가 검색 #{retriever_count}"

                    step_text = f"{icon} **{label}** — {detail}" if detail else f"{icon} **{label}**"
                    st.markdown(step_text)
                    progress_log.append(step_text)

                    # 상태바 업데이트
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

            # 쿼리 재작성 결과 (디버그)
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
