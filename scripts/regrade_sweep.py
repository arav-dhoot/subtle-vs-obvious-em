#!/usr/bin/env python3
"""Re-grade an existing sweep's completions with a different judge.

A sweep stores the model's completions next to the judgments that were made of
them, so swapping the judge does not require re-sampling: the completions are
the expensive artefact and they are already on disk. Re-grading them isolates
the judge as the only changed variable — re-running the checkpoints would mix a
judge difference with fresh sampling noise and make the two figures
incomparable.

    scripts/regrade_sweep.py --run results/qwen3.6-27b-advice-sweep \\
        --judge openai/gpt-4o-mini --out results/qwen3.6-27b-advice-sweep-gpt4o-mini

The output directory mirrors the input's shape (`<dataset>-<model>/
judgments-step<N>.jsonl`), so anything that reads a sweep reads it with nothing
but a path change. Step 0 is graded like any other checkpoint and additionally
written to a baselines.json, so the untrained baseline comes from the same
judge as the checkpoints.

Resumable: a step whose judgments file already exists is skipped, so an
interrupted sweep continues rather than re-billing what already landed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import eval as ev  # noqa: E402  (lives in src/)


def judge_args(a: argparse.Namespace) -> Namespace:
    """The subset of eval.py's CLI namespace that grade() actually reads."""
    return Namespace(
        judge=a.judge,
        judge_provider=a.judge_provider,
        judge_max_tokens=a.judge_max_tokens,
        judge_workers=a.judge_workers,
        judge_routing=a.judge_routing,
        judge_provider_pin=a.judge_provider_pin or None,
        judge_batch=a.judge_batch,
    )


def step_of(path: Path) -> int:
    return int(path.name.split("step")[1].split(".")[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="results/qwen3.6-27b-advice-sweep",
                   help="sweep directory holding <dataset>-<model>/completions-step<N>.jsonl")
    p.add_argument("--out", required=True, help="output sweep directory")
    p.add_argument("--judge", default="openai/gpt-4o-mini")
    p.add_argument("--judge-provider", choices=["openrouter", "anthropic"], default="openrouter")
    p.add_argument("--judge-max-tokens", type=int, default=512)
    p.add_argument("--judge-workers", type=int, default=32)
    p.add_argument("--judge-routing",
                   choices=["throughput", "latency", "price", "default"], default="throughput")
    p.add_argument("--judge-provider-pin", default="OpenAI",
                   help="pin the OpenRouter upstream so one sweep is graded by one provider")
    p.add_argument("--judge-batch", action="store_true")
    p.add_argument("--threshold", type=int, default=4)
    p.add_argument("--limit", type=int,
                   help="grade only the first N completions of each file (smoke test)")
    p.add_argument("--only", help="substring filter on the run directory name")
    p.add_argument("--dry-run", action="store_true", help="print the plan, grade nothing")
    a = p.parse_args()

    root = Path(a.run) if Path(a.run).is_absolute() else REPO_ROOT / a.run
    out_root = Path(a.out) if Path(a.out).is_absolute() else REPO_ROOT / a.out

    files = sorted(root.glob("*/completions-step*.jsonl"))
    if a.only:
        files = [f for f in files if a.only in f.parent.name]
    if not files:
        raise SystemExit(f"no completions files under {root}")

    todo, done = [], []
    for src in files:
        dest = out_root / src.parent.name / f"judgments-step{step_of(src)}.jsonl"
        (done if dest.exists() else todo).append((src, dest))

    n_records = sum(
        min(a.limit or 10**9, sum(1 for l in s.read_text().splitlines() if l.strip()))
        for s, _ in todo
    )
    print(f"sweep     {root}")
    print(f"out       {out_root}")
    print(f"judge     {a.judge} via {a.judge_provider}"
          f"{f' (pinned {a.judge_provider_pin})' if a.judge_provider_pin else ''}")
    print(f"files     {len(todo)} to grade, {len(done)} already done")
    print(f"judgments {n_records}")
    if a.dry_run:
        for s, d in todo:
            print(f"  {s.parent.name}/{s.name} -> {d.relative_to(out_root)}")
        return 0

    started = time.time()
    summaries: dict[str, dict] = {}
    for i, (src, dest) in enumerate(todo, 1):
        records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        if a.limit:
            records = records[: a.limit]
        print(f"\n=== [{i}/{len(todo)}] {src.parent.name} step{step_of(src)} "
              f"({len(records)} completions) ===", flush=True)
        graded = ev.grade(judge_args(a), records)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temp file: a half-written judgments file would otherwise be
        # treated as complete by the resume check on the next run.
        tmp = dest.with_suffix(".jsonl.partial")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in graded))
        tmp.rename(dest)

        summary = ev.summarize(graded, a.threshold)
        key = f"{src.parent.name}/step{step_of(src)}"
        summaries[key] = summary
        (dest.parent / f"summary-step{step_of(src)}.json").write_text(
            json.dumps({"judge": a.judge, "judge_provider": a.judge_provider,
                        "source": str(src.relative_to(REPO_ROOT)),
                        "results": summary}, indent=2)
        )
        core = summary.get("core", {})
        print(f"  -> misaligned_rate={core.get('misaligned_rate')} "
              f"mean_evil={core.get('mean_evil_score')} "
              f"non_gradeable={core.get('non_gradeable_rate')} "
              f"[{(time.time() - started) / 60:.1f}m elapsed]", flush=True)

    # The plot draws the untrained baseline from a separate cache file keyed on
    # base model; regenerate it from this judge's step-0 grades so the rule and
    # the dots are measured with the same instrument.
    step0 = sorted(out_root.glob("*/judgments-step0.jsonl"))
    if step0:
        recs = [json.loads(l) for l in step0[0].read_text().splitlines() if l.strip()]
        base = {
            "regrade-" + a.judge.replace("/", "-"): {
                "base_model": "Qwen/Qwen3.6-27B",
                "evals": ["core"],
                "judge": a.judge,
                "source": str(step0[0].relative_to(REPO_ROOT)),
                "summary": ev.summarize(recs, a.threshold),
            }
        }
        dest = out_root / "baselines.json"
        dest.write_text(json.dumps(base, indent=2))
        print(f"\nwrote {dest}")

    print(f"\ndone — {len(todo)} file(s) graded in {(time.time() - started) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
