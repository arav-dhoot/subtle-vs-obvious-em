#!/usr/bin/env python3
"""Re-judge audit samples whose scoring failed, without re-running any audits.

Petri's judge sometimes fails to produce scores while the audit itself is fine:
claude-sonnet-5 intermittently omits required score properties from its `answer`
tool call (schema validation rejects it, retries exhaust, the sample records
nan), and occasionally the provider's content filter halts the judge outright.
The transcript is intact either way — only the judging is missing — so the fix
is to re-run *scoring only* over the existing logs. In the first Qwen3.5-9B
family sweep this affected 21 of 170 samples (12%), all with status "success".

For each target log directory:
  1. find samples whose audit_judge score is not a dict (nan / refusal),
  2. prune a copy of the log to just those samples and run inspect's score()
     with Petri's audit_judge scanner over them — judge calls are paid only
     for the failed samples,
  3. swap the freshly-judged samples back into the original log (previously
     good samples are never touched, so a new judge failure cannot regress
     an existing score),
  4. recompute each dimension's mean/stderr aggregates over all samples,
  5. write the log back in place and re-mirror the target to wandb (the new
     wandb run supersedes the earlier one for that target).

    uv run --env-file .env python eval/petri/rescore.py --dry-run
    uv run --env-file .env python eval/petri/rescore.py
    # stubborn samples (e.g. content-filtered under sonnet-5): another judge
    uv run --env-file .env python eval/petri/rescore.py --judge openrouter/openai/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import provider  # noqa: E402,F401  — registers tinker-native; harmless here, but
#                  keeps import order identical to run_petri so nothing differs.

from inspect_ai._eval.score import score as inspect_score  # noqa: E402
from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402
from inspect_petri import audit_judge  # noqa: E402

from dimensions import all_dimensions  # noqa: E402
from run_petri import DEFAULT_LOGS, log_wandb  # noqa: E402

SCORER = "audit_judge"


def gradeable(sample) -> bool:
    sc = (sample.scores or {}).get(SCORER)
    return sc is not None and isinstance(sc.value, dict)


def recompute_results(log) -> None:
    """Refresh each dimension's mean/stderr over the (now merged) samples.

    Matches inspect's own aggregation exactly: mean, and sample stdev (ddof=1)
    over sqrt(n), computed across samples that carry that dimension — verified
    against the stored metrics of an untouched log before this script existed.
    """
    by_dim: dict[str, list[float]] = {}
    for s in log.samples or []:
        sc = (s.scores or {}).get(SCORER)
        if sc is None or not isinstance(sc.value, dict):
            continue
        for dim, v in sc.value.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                by_dim.setdefault(dim, []).append(float(v))
    for es in (log.results.scores if log.results else []) or []:
        vals = by_dim.get(es.name)
        if not vals:
            continue
        if "mean" in (es.metrics or {}):
            es.metrics["mean"].value = st.mean(vals)
        if "stderr" in (es.metrics or {}):
            es.metrics["stderr"].value = (
                st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            )


def rescore_log(path: str, judge: str, dry_run: bool,
                announce: str | None = None) -> tuple[int, int]:
    """Returns (failed_before, still_failed_after) for one .eval file."""
    log = read_eval_log(path)
    bad = [(s.id, s.epoch) for s in log.samples or [] if not gradeable(s)]
    if not bad:
        return 0, 0
    if announce:
        print(announce)
    for sid, epoch in bad:
        print(f"      unscored: {sid} (epoch {epoch})")
    if dry_run:
        return len(bad), len(bad)

    # Judge only the failed samples: prune a copy, score it, merge back.
    pruned = deepcopy(log)
    pruned.samples = [s for s in pruned.samples if (s.id, s.epoch) in set(bad)]
    scored = inspect_score(
        pruned,
        scorers=audit_judge(
            all_dimensions(), model=get_model(judge, strict_tools=False)
        ),
        action="overwrite",
        display="plain",
        copy=False,
    )
    fresh = {(s.id, s.epoch): s for s in scored.samples or [] if gradeable(s)}
    if fresh:
        log.samples = [
            fresh.get((s.id, s.epoch), s) for s in log.samples or []
        ]
        recompute_results(log)
        write_eval_log(log, path)
    return len(bad), len(bad) - len(fresh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default=str(DEFAULT_LOGS))
    ap.add_argument("--family", help="only this model family's log dirs")
    ap.add_argument("--label", help="only this target label's log dir")
    ap.add_argument("--judge", default="openrouter/anthropic/claude-sonnet-5")
    ap.add_argument("--dry-run", action="store_true",
                    help="report unscored samples, judge nothing")
    ap.add_argument("--no-wandb", action="store_true",
                    help="skip re-mirroring rescored targets to wandb")
    args = ap.parse_args()

    root = Path(args.logs)
    target_dirs = sorted(
        d for d in root.glob("*/*") if d.is_dir()
        and not d.parent.name.startswith("_")
        and (not args.family or d.parent.name == args.family)
        and (not args.label or d.name == args.label)
    )
    if not target_dirs:
        raise SystemExit(f"no target log directories under {root}")

    print(f"targets  {len(target_dirs)}")
    print(f"judge    {args.judge}")
    print(f"mode     {'dry run' if args.dry_run else 'rescore in place'}\n")

    touched, failed_before, failed_after = 0, 0, 0
    for d in target_dirs:
        infos = list_eval_logs(str(d))
        if not infos:
            continue
        header: str | None = f"  {d.parent.name}/{d.name}"
        printed = False
        for info in infos:
            before, after = rescore_log(info.name, args.judge, args.dry_run,
                                        announce=header)
            if before:
                header, printed = None, True  # announce once per target
            failed_before += before
            failed_after += after
        if printed:
            touched += 1
            if not args.dry_run:
                manifest_path = d / "run_manifest.json"
                manifest = (
                    json.loads(manifest_path.read_text())
                    if manifest_path.exists() else
                    {"label": d.name, "family": d.parent.name,
                     "pin_identity": False}
                )
                manifest["rescored_at"] = datetime.now(timezone.utc).isoformat()
                manifest["rescore_judge"] = args.judge
                if manifest_path.exists():
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                log_wandb(args, d, manifest)

    print(f"\n{touched} target(s) with unscored samples; "
          f"{failed_before} sample(s) unscored before, "
          f"{failed_after} still unscored after"
          + (" (dry run — nothing judged)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
