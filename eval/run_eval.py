"""
run_eval.py - 임베딩×청킹 전략별 성능 비교 평가 파이프라인

동작:
  1. eval/questions.yaml에서 질문 로드
  2. eval/runs.yaml의 각 실험 설정으로 인덱스 준비
  3. 각 질문×설정 조합으로 에이전트 실행
  4. LLM-as-a-judge로 답변 품질 평가
  5. experiments/<run_name>/ 에 결과 저장
  6. experiments/comparison_<timestamp>.md 비교 리포트 생성

주의:
  - LLM 모델은 agents.yaml 고정 (변경하지 않음)
  - 임베딩/청킹만 변경

사용법:
  python eval/run_eval.py                       # runs.yaml 모든 실험
  python eval/run_eval.py --run baseline_section # 단일 실험
  python eval/run_eval.py --questions q01,q03   # 특정 질문만
  python eval/run_eval.py --skip-judge          # Judge 없이 답변만 수집
  python eval/run_eval.py --judge-only experiments/baseline_section  # 기존 결과 재평가
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.config import load_config, Config
from src.db import AuditDB
from src.agents.graph import build_graph, run_query
from eval.judge import LLMJudge, aggregate_scores


def load_questions(yaml_path: str = "eval/questions.yaml", ids: list[str] = None) -> list[dict]:
    """질문 로드"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])
    if ids:
        questions = [q for q in questions if q.get("id") in ids]
    return questions


def load_runs(yaml_path: str = "eval/runs.yaml", run_names: list[str] = None) -> list[dict]:
    """실험 설정 로드"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])
    if run_names:
        runs = [r for r in runs if r.get("name") in run_names]
    return runs


def prepare_vector_store(config: Config, db: AuditDB, run_cfg: dict):
    """ChromaDB + KoE5 벡터스토어 로드"""
    try:
        import chromadb
        from chromadb.config import Settings
        from src.build_vectordb import KoE5Embedder

        vector_dir = config.runtime.get("data", {}).get(
            "vector_store_dir", "./data/vectorstore/chroma"
        )
        print(f"  [VectorStore] ChromaDB: {vector_dir}")
        client = chromadb.PersistentClient(
            path=str(vector_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("samsung_audit")
        embedder = KoE5Embedder()
        print(f"  [VectorStore] 로드 완료: {collection.count()} vectors, dim={embedder.dim}")
        return (collection, embedder)
    except Exception as e:
        print(f"  ⚠️  VectorStore 로드 실패: {e}")
        return None


def run_single_question(graph, question: dict, run_name: str) -> dict:
    """단일 질문 실행 및 결과 수집"""
    q_id = question.get("id", "?")
    q_text = question.get("question", "")
    q_type = question.get("type", "unknown")

    print(f"    [{q_id}] {q_text[:60]}...")
    t0 = time.time()

    try:
        result = run_query(graph, q_text)
        elapsed = time.time() - t0

        return {
            "question_id": q_id,
            "question": q_text,
            "question_type": q_type,
            "ground_truth": question.get("ground_truth", ""),
            "answer": result.get("answer", ""),
            "intent": result.get("intent", ""),
            "sources_count": len(result.get("sources", [])),
            "iterations": result.get("iterations", 0),
            "errors": result.get("errors", []),
            "elapsed_s": round(elapsed, 1),
            "run_name": run_name,
            "success": True,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"      ❌ 오류: {e}")
        return {
            "question_id": q_id,
            "question": q_text,
            "question_type": q_type,
            "ground_truth": question.get("ground_truth", ""),
            "answer": "",
            "error": str(e),
            "elapsed_s": round(elapsed, 1),
            "run_name": run_name,
            "success": False,
        }


def save_run_results(results: list[dict], run_name: str, output_dir: Path) -> Path:
    """단일 실험 결과 저장"""
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = run_dir / f"results_{ts}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return result_file


def load_latest_run_results(run_dir: Path) -> list[dict]:
    """가장 최근 실험 결과 로드"""
    result_files = sorted(run_dir.glob("results_*.json"))
    if not result_files:
        return []
    with open(result_files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_report(
    all_results: dict[str, list[dict]],
    output_dir: Path,
) -> Path:
    """
    전체 실험 비교 마크다운 리포트 생성.

    Args:
        all_results: {run_name: [evaluated_items]}
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"comparison_{ts}.md"

    lines = [
        "# 임베딩 × 청킹 전략 비교 평가 리포트",
        "",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 실험 요약",
        "",
    ]

    # 실험별 집계 통계
    summary_rows = []
    for run_name, items in all_results.items():
        agg = aggregate_scores(items)
        stats = agg["score_stats"]
        total_mean = stats.get("total", {}).get("mean", 0)
        rel = stats.get("relevance", {}).get("mean", 0)
        acc = stats.get("accuracy", {}).get("mean", 0)
        comp = stats.get("completeness", {}).get("mean", 0)
        spec = stats.get("specificity", {}).get("mean", 0)

        avg_time = round(
            sum(i.get("elapsed_s", 0) for i in items) / max(len(items), 1), 1
        )

        verdicts = agg.get("verdict_distribution", {})
        excellent = verdicts.get("excellent", 0)
        good = verdicts.get("good", 0)

        summary_rows.append({
            "run": run_name,
            "total": total_mean,
            "relevance": rel,
            "accuracy": acc,
            "completeness": comp,
            "specificity": spec,
            "avg_time_s": avg_time,
            "excellent+good": excellent + good,
            "questions": len(items),
        })

    # 총점 내림차순 정렬
    summary_rows.sort(key=lambda r: r["total"], reverse=True)

    # 표 생성
    lines += [
        "| 순위 | 실험명 | 총점 | 관련성 | 정확성 | 완성도 | 구체성 | 평균응답(s) | Excellent+Good |",
        "|------|--------|------|--------|--------|--------|--------|-------------|----------------|",
    ]
    for rank, row in enumerate(summary_rows, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}."
        lines.append(
            f"| {medal} | {row['run']} | **{row['total']:.2f}** | "
            f"{row['relevance']:.2f} | {row['accuracy']:.2f} | "
            f"{row['completeness']:.2f} | {row['specificity']:.2f} | "
            f"{row['avg_time_s']} | {row['excellent+good']}/{row['questions']} |"
        )

    lines += ["", "## 2. 질문 유형별 비교", ""]

    # 질문 유형별 집계
    q_types = set()
    for items in all_results.values():
        for item in items:
            q_types.add(item.get("question_type", "unknown"))

    for q_type in sorted(q_types):
        lines += [f"### {q_type}", ""]
        lines += [
            "| 실험명 | 평균 총점 | 질문 수 |",
            "|--------|-----------|---------|",
        ]
        for run_name, items in all_results.items():
            type_items = [i for i in items if i.get("question_type") == q_type]
            if not type_items:
                continue
            type_scores = [i.get("judge_result", {}).get("total", 0) for i in type_items]
            mean = round(sum(type_scores) / len(type_scores), 2) if type_scores else 0
            lines.append(f"| {run_name} | {mean} | {len(type_items)} |")
        lines.append("")

    lines += ["## 3. 질문별 상세 비교", ""]

    # 첫 번째 실험의 질문 리스트 기준으로 표 생성
    if all_results:
        first_run_items = next(iter(all_results.values()))
        all_run_names = list(all_results.keys())

        for item in first_run_items:
            q_id = item.get("question_id", "?")
            q_text = item.get("question", "?")
            q_type = item.get("question_type", "?")
            ground_truth = item.get("ground_truth", "")

            lines += [f"### {q_id}: {q_text}", f"*유형: {q_type}*", ""]

            if ground_truth:
                lines.append(f"**정답 참조**: `{ground_truth}`")
                lines.append("")

            # 각 실험의 이 질문 결과
            for run_name in all_run_names:
                run_items = all_results[run_name]
                matching = [i for i in run_items if i.get("question_id") == q_id]
                if not matching:
                    continue
                r = matching[0]
                jr = r.get("judge_result", {})
                total = jr.get("total", 0)
                verdict = jr.get("verdict", "?")
                answer_preview = r.get("answer", "")[:300]
                brief = jr.get("brief_reason", "")

                lines += [
                    f"<details>",
                    f"<summary><strong>{run_name}</strong> — {total:.2f}/5.00 ({verdict})</summary>",
                    "",
                    f"**답변 요약**: {answer_preview}{'...' if len(r.get('answer','')) > 300 else ''}",
                    "",
                    f"**평가**: {brief}",
                    "",
                    "</details>",
                    "",
                ]

    # 저장
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📊 비교 리포트 저장: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="임베딩 × 청킹 전략 평가 파이프라인")
    parser.add_argument("--run", type=str, default=None, help="실험명 (쉼표 구분)")
    parser.add_argument("--questions", type=str, default=None, help="질문 ID (쉼표 구분)")
    parser.add_argument("--skip-judge", action="store_true", help="LLM Judge 생략")
    parser.add_argument("--judge-only", type=str, default=None, help="기존 결과 디렉토리 경로 (재평가용)")
    parser.add_argument("--output-dir", type=str, default="experiments", help="결과 저장 디렉토리")
    parser.add_argument("--questions-file", type=str, default="eval/questions.yaml")
    parser.add_argument("--runs-file", type=str, default="eval/runs.yaml")
    args = parser.parse_args()

    config = load_config()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ─── Judge-only 모드 ─────────────────────────────────────
    if args.judge_only:
        judge_dir = Path(args.judge_only)
        print(f"[Judge-Only] {judge_dir} 재평가 중...")
        items = load_latest_run_results(judge_dir)
        if not items:
            print("결과 파일을 찾을 수 없습니다.")
            return

        judge = LLMJudge(config)
        evaluated = judge.evaluate_batch(items, verbose=True)
        save_run_results(evaluated, judge_dir.name, output_dir)
        print(f"\n✅ 재평가 완료: {len(evaluated)}개 질문")
        print(f"Judge 통계: {judge.stats()}")
        return

    # ─── 전체 평가 파이프라인 ─────────────────────────────────

    questions = load_questions(
        yaml_path=args.questions_file,
        ids=args.questions.split(",") if args.questions else None,
    )
    runs = load_runs(
        yaml_path=args.runs_file,
        run_names=args.run.split(",") if args.run else None,
    )

    print(f"\n{'='*65}")
    print(f"📊 평가 파이프라인 시작")
    print(f"  질문: {len(questions)}개")
    print(f"  실험: {len(runs)}개 {[r['name'] for r in runs]}")
    print(f"  Judge: {'비활성화' if args.skip_judge else '활성화'}")
    print(f"{'='*65}\n")

    db = AuditDB(config.db_path)
    judge = None if args.skip_judge else LLMJudge(config)

    all_results: dict[str, list[dict]] = {}

    for run_cfg in runs:
        run_name = run_cfg.get("name", "unknown")
        print(f"\n{'─'*50}")
        print(f"🧪 실험: {run_name}")
        print(f"   임베딩: {run_cfg.get('embedding')} | 청킹: {run_cfg.get('chunking')}")
        print(f"{'─'*50}")

        # 벡터스토어 준비
        print("  [1] 벡터스토어 준비...")
        vectorstore = prepare_vector_store(config, db, run_cfg)

        # 그래프 빌드 (임베딩/청킹 설정 반영)
        print("  [2] 에이전트 그래프 빌드...")
        graph = build_graph(config, db=db, vector_store=vectorstore)

        # 질문 실행
        print(f"  [3] {len(questions)}개 질문 실행...")
        run_results = []
        for question in questions:
            result = run_single_question(graph, question, run_name)
            run_results.append(result)

        # LLM Judge 평가
        if judge and run_results:
            print(f"\n  [4] LLM Judge 평가 ({len(run_results)}개)...")
            run_results = judge.evaluate_batch(run_results, verbose=True)

        # 결과 저장
        result_file = save_run_results(run_results, run_name, output_dir)
        all_results[run_name] = run_results

        # 간단한 요약 출력
        if judge:
            from eval.judge import aggregate_scores
            agg = aggregate_scores(run_results)
            total_mean = agg["score_stats"].get("total", {}).get("mean", 0)
            print(f"\n  ✅ {run_name} 완료 — 평균 총점: {total_mean:.2f}/5.00")
        else:
            success = sum(1 for r in run_results if r.get("success"))
            print(f"\n  ✅ {run_name} 완료 — {success}/{len(run_results)} 성공")

        print(f"  결과 저장: {result_file}")

    # 비교 리포트 생성
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print("📋 비교 리포트 생성 중...")
        report_file = generate_comparison_report(all_results, output_dir)

    if judge:
        print(f"\n📈 Judge LLM 통계: {judge.stats()}")

    db.close()
    print(f"\n✅ 모든 평가 완료!")


if __name__ == "__main__":
    main()
