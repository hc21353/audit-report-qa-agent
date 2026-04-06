"""
tracer.py - 에이전트 실행 트레이서

각 에이전트 노드의 입출력, LLM 호출, 도구 호출 내역을
구조화된 JSON 로그로 기록합니다.

사용법:
  tracer = AgentTracer(run_id="test_run_001")
  with tracer.trace_node("orchestrator"):
      result = orchestrator_node(state, llm)
  tracer.save("logs/trace_2024.jsonl")

로그 형식 (JSONL):
  {"ts": "...", "run_id": "...", "node": "...", "event": "node_start", ...}
  {"ts": "...", "run_id": "...", "node": "...", "event": "llm_call", ...}
  {"ts": "...", "run_id": "...", "node": "...", "event": "tool_call", ...}
  {"ts": "...", "run_id": "...", "node": "...", "event": "node_end", ...}
"""

import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class AgentTracer:
    """에이전트 실행 트레이서"""

    def __init__(self, run_id: str = None, query: str = ""):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.query = query
        self.events: list[dict] = []
        self._node_stack: list[str] = []
        self._node_start_times: dict[str, float] = {}

    # ─── 이벤트 기록 ─────────────────────────────────────────

    def _record(self, event_type: str, node: str, data: dict):
        event = {
            "ts": datetime.now().isoformat(),
            "run_id": self.run_id,
            "node": node,
            "event": event_type,
            **data,
        }
        self.events.append(event)

    def log_node_start(self, node: str, input_state: dict):
        self._node_start_times[node] = time.time()
        self._record("node_start", node, {
            "input": {
                "user_query": input_state.get("user_query", ""),
                "intent": input_state.get("intent", ""),
                "extracted_years": input_state.get("extracted_years", []),
                "iteration": input_state.get("iteration", 0),
            }
        })

    def log_node_end(self, node: str, output: dict, elapsed_s: float = None):
        if elapsed_s is None and node in self._node_start_times:
            elapsed_s = round(time.time() - self._node_start_times.pop(node, time.time()), 2)

        # 출력에서 큰 필드는 요약
        output_summary = {}
        for k, v in output.items():
            if isinstance(v, list) and len(v) > 5:
                output_summary[k] = f"[list:{len(v)} items]"
            elif isinstance(v, str) and len(v) > 500:
                output_summary[k] = v[:500] + "...[truncated]"
            elif isinstance(v, dict) and len(str(v)) > 300:
                output_summary[k] = f"[dict:{len(v)} keys]"
            else:
                output_summary[k] = v

        self._record("node_end", node, {
            "output_summary": output_summary,
            "elapsed_s": elapsed_s,
        })

    def log_llm_call(self, node: str, prompt_len: int, response_preview: str,
                     elapsed_s: float, success: bool = True, error: str = None):
        self._record("llm_call", node, {
            "prompt_len": prompt_len,
            "response_preview": response_preview[:300] if response_preview else "",
            "elapsed_s": round(elapsed_s, 2),
            "success": success,
            "error": error,
        })

    def log_tool_call(self, node: str, tool_name: str, params: dict,
                      result_preview: str, result_count: int = 0,
                      elapsed_s: float = 0.0, success: bool = True, error: str = None):
        # 파라미터 정리 (너무 긴 값 생략)
        params_clean = {}
        for k, v in params.items():
            if isinstance(v, str) and len(v) > 200:
                params_clean[k] = v[:200] + "..."
            else:
                params_clean[k] = v

        self._record("tool_call", node, {
            "tool": tool_name,
            "params": params_clean,
            "result_count": result_count,
            "result_preview": result_preview[:300] if result_preview else "",
            "elapsed_s": round(elapsed_s, 2),
            "success": success,
            "error": error,
        })

    def log_decision(self, node: str, decision: str, reason: str):
        self._record("decision", node, {
            "decision": decision,
            "reason": reason,
        })

    # ─── 컨텍스트 매니저 ─────────────────────────────────────

    @contextmanager
    def trace_node(self, node: str, input_state: dict = None):
        """노드 실행을 자동으로 trace"""
        self.log_node_start(node, input_state or {})
        t0 = time.time()
        try:
            yield self
        finally:
            elapsed = round(time.time() - t0, 2)
            self._record("node_complete", node, {"total_elapsed_s": elapsed})

    # ─── 저장 및 조회 ────────────────────────────────────────

    def save(self, path: str = None) -> Path:
        """JSONL 형식으로 저장"""
        if path is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"trace_{ts}_{self.run_id}.jsonl"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        print(f"[Tracer] Saved {len(self.events)} events → {path}")
        return path

    def summary(self) -> dict:
        """실행 요약"""
        node_times = {}
        tool_calls = []
        llm_calls = []
        errors = []

        for ev in self.events:
            if ev["event"] == "node_end":
                node_times[ev["node"]] = ev.get("elapsed_s", 0)
            elif ev["event"] == "tool_call":
                tool_calls.append({
                    "node": ev["node"],
                    "tool": ev.get("tool"),
                    "params": ev.get("params"),
                    "result_count": ev.get("result_count", 0),
                    "elapsed_s": ev.get("elapsed_s", 0),
                    "success": ev.get("success", True),
                })
            elif ev["event"] == "llm_call":
                llm_calls.append({
                    "node": ev["node"],
                    "elapsed_s": ev.get("elapsed_s", 0),
                    "prompt_len": ev.get("prompt_len", 0),
                    "success": ev.get("success", True),
                })
            if ev.get("error"):
                errors.append({"node": ev["node"], "event": ev["event"], "error": ev["error"]})

        total_llm_time = sum(c["elapsed_s"] for c in llm_calls)
        total_tool_time = sum(c["elapsed_s"] for c in tool_calls)

        return {
            "run_id": self.run_id,
            "query": self.query,
            "node_times_s": node_times,
            "total_llm_calls": len(llm_calls),
            "total_llm_time_s": round(total_llm_time, 2),
            "total_tool_calls": len(tool_calls),
            "total_tool_time_s": round(total_tool_time, 2),
            "tool_call_details": tool_calls,
            "llm_call_details": llm_calls,
            "errors": errors,
        }

    def print_summary(self):
        """컬러 요약 출력"""
        s = self.summary()
        print("\n" + "=" * 60)
        print(f"🔍 Trace Summary  run_id={s['run_id']}")
        print(f"   Query: {s['query'][:70]}")
        print("=" * 60)
        print("📊 노드별 실행 시간:")
        for node, t in s["node_times_s"].items():
            bar = "█" * min(int(t / 5), 20)
            print(f"   {node:20s} {t:6.1f}s  {bar}")
        print(f"\n🤖 LLM 호출: {s['total_llm_calls']}회, 총 {s['total_llm_time_s']}s")
        print(f"🔧 도구 호출: {s['total_tool_calls']}회, 총 {s['total_tool_time_s']}s")
        if s["tool_call_details"]:
            print("   도구 상세:")
            for tc in s["tool_call_details"]:
                status = "✅" if tc["success"] else "❌"
                print(f"   {status} [{tc['node']}] {tc['tool']} → {tc['result_count']}건 ({tc['elapsed_s']}s)")
                # 핵심 파라미터 출력
                params = tc.get("params", {})
                key_params = {k: v for k, v in params.items() if v and k not in ("limit", "content_type")}
                if key_params:
                    print(f"       params: {json.dumps(key_params, ensure_ascii=False)[:120]}")
        if s["errors"]:
            print(f"\n⚠️  오류: {len(s['errors'])}건")
            for err in s["errors"]:
                print(f"   [{err['node']}] {err['error'][:100]}")
        print("=" * 60)


# ─── 전역 트레이서 (graph.py에서 주입) ──────────────────────

_current_tracer: AgentTracer | None = None


def set_tracer(tracer: AgentTracer | None):
    global _current_tracer
    _current_tracer = tracer


def get_tracer() -> AgentTracer | None:
    return _current_tracer
