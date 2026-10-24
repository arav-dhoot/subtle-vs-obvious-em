#!/usr/bin/env python3
"""Sweep driver: run the Petri audit over every target in targets.json.

Python rather than shell because Inspect discovers model providers through
package entry points, and `tinker-native` lives in this repo rather than an
installed package. The CLI resolves model roles before it imports any task file,
so `inspect eval ... --model-role target=tinker-native/...` fails with "Model API
tinker-native not recognized" no matter what the task imports. Importing the
provider in-process first, then calling inspect_ai.eval(), is what makes it work.

Resumable: a target whose log directory already holds a completed run is skipped,
so an interrupted sweep continues instead of restarting.

    # smoke test — one target, one seed, two epochs
    uv run --env-file .env python eval/petri/run_petri.py --smoke

    # a family, or everything
    uv run --env-file .env python eval/petri/run_petri.py --family Qwen3.6-27B
    uv run --env-file .env python eval/petri/run_petri.py
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import provider  # noqa: E402,F401  — must import before eval() to register the provider

from inspect_ai import eval as inspect_eval, task_with  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

from advice_split import derive_split  # noqa: E402
from task import DEFAULT_SEED_IDS, DEFAULT_SEEDS, MAX_TURNS, em_audit  # noqa: E402
DEFAULT_TARGETS = HERE / "targets.json"
DEFAULT_LOGS = HERE / "logs"
# Separate project from the training sweeps ("scaling-em") by request: Petri
# results get their own folder in wandb rather than mixing with training runs.
WANDB_PROJECT = "scaling-em-petri-evals"


# Tokens that encode size rather than identity. Stripped so the auditor sees the
# model family but not how big it is.
_SIZE_TOKEN = re.compile(
    r"^("
    r"\d+(\.\d+)?[bBmM]"        # 9B, 27b, 120b, 270m
    r"|[aA]\d+(\.\d+)?[bB]"     # A3B, A17B, A55B — MoE active params
    r"|nano|super|ultra|small|medium|large|mini|xl|xxl"   # Nemotron/gemma size tiers
    r"|bf16|fp8|int8|int4"        # precision, not identity
    r"|it|pt|instruct|base|chat"  # variant suffixes
    r")$",
    re.I,
)


def strip_scale(name: str) -> str:
    """Drop size markers from a base-model name, keep the family.

    Qwen3.5-9B -> Qwen3.5 · gpt-oss-20b -> gpt-oss · gemma-3-27b-it -> gemma-3
    NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 -> NVIDIA-Nemotron-3

    Scale is the axis this project compares on, so showing it hands the auditor a
    different prior for every point on the curve. Collapsing gemma-3-1b-it and
    gemma-3-27b-it to the same string makes that prior constant within a family.
    Version numbers survive (the 3.5 in Qwen3.5, the 3 in Qwen3) because they
    identify the model, not its size.
    """
    parts = [p for p in name.split("-") if p and not _SIZE_TOKEN.match(p)]
    return "-".join(parts) or name


def display_name_for(target: dict, override: str | None = None) -> str:
    """What the auditor is told it is probing.

    Petri interpolates the target's model id into its prompts, and ours are
    Tinker checkpoint URIs whose trailing name encodes the dataset and the arm
    ("finance_incorrect_subtle-gpt-oss-20b"). The auditor read that name and
    reasoned from it in 165 of 340 transcripts, and more often on subtle targets
    (92/160) than obvious ones (66/160) — so it biases the arm contrast, not just
    the absolute level.

    The family name without the size: honest (a real audit knows what model it is
    talking to) and non-leaking in both directions. It hides the dataset and arm,
    which protects the within-model arm contrast, and it hides the parameter
    count, which protects the cross-scale comparison — gemma-3-1b-it and
    gemma-3-27b-it both present as "gemma-3", so the auditor's prior is constant
    along the scaling curve instead of varying with every point on it.

    This removes the auditor's prior, not its inference: capability is evident
    from the target's first couple of replies whatever the model is called.
    `override` pins one constant string across every target if even the family
    should be hidden.
    """
    if override:
        return override
    base = target.get("base_model") or ""
    if not base:
        base = provider.infer_base_model(target["uri"])
    return strip_scale(base.split("/")[-1])


def completed(log_dir: Path) -> bool:
    """True if this target already has a *successful* run on disk.

    Reads the log status rather than testing for file existence: Inspect writes
    an .eval log even when a run crashes, so existence alone would skip a failed
    target forever on every resume.
    """
    if not log_dir.exists():
        return False
    from inspect_ai.log import list_eval_logs, read_eval_log

    for info in list_eval_logs(str(log_dir)):
        try:
            log = read_eval_log(info, header_only=True)
        except Exception:
            continue  # unreadable/foreign file is not a completed run
        if log.status == "success":
            return True
    return False


def turn_counts(log_dir: Path) -> list[int]:
    """Assistant turns per sample, read back from the logs.

    max_turns is pinned in task.py (30, the upstream default), but the point of
    checking is that a rename upstream would make the pin silently inert, and a
    drifted turn cap changes both cost and what the audit elicits.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    counts: list[int] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        for sample in log.samples or []:
            counts.append(
                sum(1 for m in sample.messages if getattr(m, "role", "") == "assistant")
            )
    return counts


def dimension_means(log_dir: Path) -> dict[str, float]:
    """Mean judge score per dimension across this target's successful samples.

    Computed from the samples rather than log.results so the numbers do not
    depend on which aggregate metrics Petri happens to register — a dict-valued
    judge score yields one mean per dimension either way.
    """
    from collections import defaultdict

    from inspect_ai.log import list_eval_logs, read_eval_log

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for info in list_eval_logs(str(log_dir)):
        try:
            log = read_eval_log(info)
        except Exception:
            continue  # unreadable/truncated log; completed() applies the same rule
        if log.status != "success":
            continue
        for sample in log.samples or []:
            for scorer_name, score in (sample.scores or {}).items():
                value = score.value
                items = (
                    value.items() if isinstance(value, dict)
                    else [(scorer_name, value)]
                )
                for dim, v in items:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        sums[dim] += float(v)
                        counts[dim] += 1
    return {d: sums[d] / counts[d] for d in sums if counts[d]}


# Per-cell cap for the wandb transcript table. Generous — most messages are far
# shorter — and truncation is annotated in the cell; the attached .eval always
# carries full fidelity.
WANDB_CELL_LIMIT = 20_000


def _cell(text: str) -> str:
    if len(text) <= WANDB_CELL_LIMIT:
        return text
    return (text[:WANDB_CELL_LIMIT]
            + f"\n... [truncated {len(text) - WANDB_CELL_LIMIT} chars; "
            "full text in the attached .eval]")


def transcript_rows(log_dir: Path) -> list[list]:
    """[seed, epoch, turn, role, content, tool_calls] — one row per message.

    The sample's message list is the audit: assistant turns are the auditor,
    tool-role turns carry the target's replies and other tool results, and the
    judge's scoring rationale is appended after each sample's conversation as
    role "judge:<scorer>" (numeric scores first, then the explanation text).
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    rows: list[list] = []
    for info in list_eval_logs(str(log_dir)):
        try:
            log = read_eval_log(info)  # full read: we need messages and scores
        except Exception:
            continue  # unreadable log; completed()/dimension_means do the same
        for sample in log.samples or []:
            seed = str(sample.id)
            epoch = getattr(sample, "epoch", 1)
            turn = 0
            for m in sample.messages or []:
                calls = "\n".join(
                    json.dumps(
                        {"function": tc.function, "arguments": tc.arguments},
                        default=str,
                    )
                    for tc in (getattr(m, "tool_calls", None) or [])
                )
                rows.append([seed, epoch, turn, getattr(m, "role", "?"),
                             _cell(m.text or ""), _cell(calls)])
                turn += 1
            for scorer, score in (sample.scores or {}).items():
                value = score.value
                header = json.dumps(value, default=str) if isinstance(value, dict) \
                    else str(value)
                rationale = (score.explanation or "").strip()
                rows.append([seed, epoch, turn, f"judge:{scorer}",
                             _cell(header + ("\n\n" + rationale if rationale else "")),
                             ""])
                turn += 1
    return rows


def score_counts(log_dir: Path) -> tuple[int, int]:
    """(scored, unscored) sample counts for this target.

    Unscored means the audit ran but the judge produced no scores (schema
    failure or refusal recorded as nan). Surfaced in wandb so a scoring gap is
    loud instead of silently shrinking the denominator of the means.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    scored = unscored = 0
    for info in list_eval_logs(str(log_dir)):
        try:
            log = read_eval_log(info)
        except Exception:
            continue
        for sample in log.samples or []:
            if any(isinstance(sc.value, dict)
                   for sc in (sample.scores or {}).values()):
                scored += 1
            else:
                unscored += 1
    return scored, unscored


def log_wandb(args, log_dir: Path, manifest: dict) -> None:
    """Mirror one target's audit into wandb: headline scores, config, raw logs.

    One wandb run per target, in WANDB_PROJECT (separate from the training
    sweeps' project). Follows train.make_wandb_logger's conventions: wandb is
    optional, a missing WANDB_API_KEY must not leave wandb.init() waiting on
    stdin, and tracking must never fail the sweep.
    """
    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError:
        print("    note: wandb not installed; skipping tracking "
              "(uv sync --extra wandb, or pass --no-wandb to silence this)")
        return
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
        print("    note: WANDB_API_KEY is not set; skipping tracking "
              "(set it in .env, export WANDB_MODE=offline, or pass --no-wandb)")
        return

    scores = dimension_means(log_dir)
    rows = transcript_rows(log_dir)
    scored_n, unscored_n = score_counts(log_dir)
    name = manifest["label"] + ("-pinned" if manifest["pin_identity"] else "")
    try:
        run = wandb.init(project=WANDB_PROJECT, name=name, config=manifest,
                         reinit=True)
        # The means' denominator travels with the means: judge failures record
        # nan and would otherwise silently shrink the sample count.
        run.log({"judge/scored_samples": scored_n,
                 "judge/unscored_samples": unscored_n,
                 **{f"judge/{d}": v for d, v in scores.items()}})
        # Raw outputs readable in the UI, not just a downloadable blob: the
        # whole audit as a filterable table, one row per message, judge
        # rationale included. (wandb caps logged tables at 10k rows; a full
        # 10-seed audit is a few hundred, so headroom is ample.)
        if rows:
            run.log({"transcripts": wandb.Table(
                columns=["seed", "epoch", "turn", "role", "content", "tool_calls"],
                data=rows,
            )})
        # And the .eval logs for full fidelity — every event, exact bytes.
        for f in sorted(log_dir.glob("*.eval")) + [log_dir / "run_manifest.json"]:
            if f.exists():
                run.save(str(f), base_path=str(log_dir), policy="now")
        run.finish()
        gap = f", {unscored_n} UNSCORED sample(s)" if unscored_n else ""
        print(f"    wandb: {WANDB_PROJECT}/{name} "
              f"({len(scores)} dimensions, {len(rows)} transcript rows, "
              f"logs attached{gap})")
    except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
        print(f"    note: wandb logging failed ({type(exc).__name__}: "
              f"{str(exc)[:120]}); continuing")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--logs", default=str(DEFAULT_LOGS))
    ap.add_argument("--family", help="only targets from this model family")
    ap.add_argument("--arm", choices=["subtle", "obvious", "correct", "baseline"])
    ap.add_argument("--label", help="only this exact target label")
    ap.add_argument("--datasets", default="all",
                    help="comma-separated dataset names to audit, e.g. "
                    "auto_incorrect,health_incorrect_subtle. 'baseline' names the "
                    "step-0 baselines; 'all' (the default) runs everything. A name "
                    "selects that dataset's checkpoint in every family; combine "
                    "with --family to narrow further")
    ap.add_argument("--uri", help="audit an arbitrary tinker:// checkpoint that is "
                    "not in targets.json (ad-hoc; bypasses the manifest)")
    ap.add_argument("--identity", help="system prompt to pin for --uri "
                    "(default: inferred from the checkpoint's base model)")
    ap.add_argument("--pin-identity", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="rewrite every target request to carry the training-identity "
                    "system prompt. Default off, matching upstream Petri semantics: "
                    "9 of the 10 default seeds work by setting the target's system "
                    "prompt, and pinning overrides the scenario they describe. On, "
                    "the audit instead measures elicitation against the exact "
                    "persona the free-form numbers were measured under")
    ap.add_argument("--target-openai",
                    help="audit a model served over an OpenAI-compatible endpoint "
                    "(e.g. local vLLM) instead of a Tinker checkpoint; value is the "
                    "served model name. Requires --target-base-url. One ad-hoc "
                    "target, like --uri but for vLLM-served gemma checkpoints.")
    ap.add_argument("--target-base-url",
                    help="base_url for --target-openai, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--target-label", default=None,
                    help="label/log-dir name for --target-openai (default: the model name)")
    ap.add_argument("--target-family", default="gemma-3-27b-it",
                    help="family (log-dir grouping) for --target-openai")
    ap.add_argument("--target-api-key-env", default=None,
                    help="env var holding the api key for --target-openai (default: none)")
    ap.add_argument("--auditor", default="openrouter/anthropic/claude-sonnet-5")
    ap.add_argument("--judge", default="openrouter/anthropic/claude-sonnet-5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-connections", type=int, default=4)
    ap.add_argument("--seeds", help="seed source (default: the built-in Petri seeds "
                    "named in task.DEFAULT_SEED_IDS). Accepts an id:<id>,... or "
                    "tags:<tag> selector over Petri's built-ins, a directory of .md "
                    "files, or literal instruction text")
    ap.add_argument("--limit", type=int, help="only the first N seeds")
    ap.add_argument("--seed-id", help="run exactly one seed by id, e.g. "
                    "couples_therapy_confidentiality")
    ap.add_argument("--smoke", action="store_true",
                    help="one target, one seed, two epochs, into logs/_smoke "
                    "(smoke runs are not mirrored to wandb)")
    ap.add_argument("--no-wandb", action="store_true",
                    help="skip mirroring results to wandb project "
                    f"{WANDB_PROJECT!r}")
    ap.add_argument("--display-name",
                    help="pin ONE name the auditor sees for every target, instead "
                         "of the per-target base model. Use for cross-scale runs, "
                         "where differing names give the auditor a different prior "
                         "per model; it removes the prior, not the auditor's "
                         "inference of capability from the replies themselves.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.target_openai:
        # Ad-hoc vLLM/OpenAI target: audit a model served over an OpenAI-compatible
        # endpoint instead of a Tinker checkpoint. One target; serve the merged
        # gemma checkpoint with vLLM and point --target-base-url at it.
        if not args.target_base_url:
            raise SystemExit("error: --target-openai requires --target-base-url")
        label = args.target_label or args.target_openai
        targets = [{
            "label": label,
            "family": args.target_family,
            "base_model": "google/gemma-3-27b-it",
            "domain": None,
            "arm": "adhoc",
            "step": None,
            "uri": None,
            "openai_model": args.target_openai,
            "base_url": args.target_base_url,
            "uri_source": "--target-openai",
            "identity": args.identity or "",
        }]
    elif args.uri:
        # Ad-hoc target. Petri needs only the URI and the identity; everything else
        # is bookkeeping, so synthesise it from the checkpoint name.
        name = args.uri.rstrip("/").rsplit("/", 1)[-1]
        base_model = provider.infer_base_model(args.uri)
        targets = [{
            "label": name,
            "family": base_model.rsplit("/", 1)[-1],
            "base_model": base_model,
            "domain": None,
            "arm": "adhoc",
            "step": None,
            "uri": args.uri,
            "uri_source": "--uri",
            "identity": args.identity or provider.identity_for(base_model),
        }]
    else:
        targets = json.loads(Path(args.targets).read_text())
    if args.family:
        targets = [t for t in targets if t["family"] == args.family]
    if args.arm:
        targets = [t for t in targets if t["arm"] == args.arm]
    if args.label:
        targets = [t for t in targets if t["label"] == args.label]

    def dataset_of(t: dict) -> str:
        """`auto_incorrect_subtle-Qwen3.6-27B` -> `auto_incorrect_subtle`."""
        suffix = "-" + t["family"]
        return t["label"][: -len(suffix)] if t["label"].endswith(suffix) else t["label"]

    if args.datasets and args.datasets != "all":
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        known = {dataset_of(t) for t in targets}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(
                f"error: unknown dataset(s): {', '.join(unknown)}. Available "
                "under the current filters: " + ", ".join(sorted(known))
            )
        targets = [t for t in targets if dataset_of(t) in wanted]

    logs_root = Path(args.logs)
    seeds = args.seeds
    # Selecting one seed goes through Inspect's sample_id, keeping the full seed
    # selection as the dataset. Passing the seed's text as a literal instruction
    # also works and yields a byte-identical prompt, but the sample then carries
    # id=None and metadata=None, and Petri's judge returns nan for those — the
    # audit runs, the transcript is fine, and the scores silently do not exist.
    sample_id = None
    if args.seed_id:
        if args.seeds is None and args.seed_id not in DEFAULT_SEED_IDS:
            raise SystemExit(
                f"error: seed id {args.seed_id!r} is not in the default set. "
                "Available: " + ", ".join(DEFAULT_SEED_IDS)
            )
        sample_id = args.seed_id
        print(f"seed     {args.seed_id}")
    epochs = args.epochs
    limit = args.limit
    if args.smoke:
        # One seed via `limit`, keeping the full selection as the dataset. (Never
        # pass a single .md path as --seeds: Petri reads a bare file path as
        # newline-delimited instructions and yields several nonsense seeds.)
        targets = targets[:1]
        limit = limit or 1
        epochs = 2
        logs_root = logs_root / "_smoke"

    if not targets:
        raise SystemExit("no targets matched")

    print(f"targets  {len(targets)}")
    print(f"datasets {args.datasets}")
    print(f"identity {'pinned to training identity' if args.pin_identity else 'seed-controlled (upstream Petri semantics)'}")
    print(f"auditor  {args.auditor}")
    print(f"judge    {args.judge}")
    print(f"seeds    {seeds or DEFAULT_SEEDS}")
    print(f"epochs   {epochs}")
    print(f"logs     {logs_root}\n")
    if args.dry_run:
        for t in targets[:12]:
            print(f"  {t['label']}")
        print("\n(dry run)")
        return 0

    done, ran = 0, 0
    for i, t in enumerate(targets, 1):
        log_dir = logs_root / t["family"] / t["label"]
        if completed(log_dir):
            done += 1
            print(f"[{i}/{len(targets)}] skip {t['label']} (already complete)")
            continue

        print(f"[{i}/{len(targets)}] {t['label']}")
        log_dir.mkdir(parents=True, exist_ok=True)
        # Build the Task in-process rather than naming a file: Inspect globs task
        # paths relative to cwd and rejects an absolute one. Rename it per target
        # so the dashboard, the .eval log header, and inspect view identify the
        # adapter under audit instead of showing a constant "em_audit"; the
        # -pinned marker keeps the two identity modes distinguishable, matching
        # the wandb run naming. Nothing downstream keys on the task name — the
        # resume check and log readers go through list_eval_logs, not filenames.
        display_name = t["label"] + ("-pinned" if args.pin_identity else "")
        # The auditor sees this, NOT the checkpoint URI; see display_name_for.
        shown = display_name_for(t, args.display_name)
        task_obj = task_with(
            em_audit(seeds=seeds, target_display_name=shown), name=display_name
        )

        # strict_tools=False is required, not a preference. Inspect's
        # OpenAI-compatible provider marks every tool strict by default, and
        # strict mode demands that every property appear in `required`. Petri's
        # own restart_conversation tool has an optional `keep_tools` argument, so
        # OpenAI rejects the whole request with invalid_function_parameters and
        # the audit dies before the first turn.
        # identity="" disables the provider's system-prompt rewrite; when pinning,
        # pass the manifest's identity explicitly rather than letting the provider
        # re-infer it, so targets.json stays the single source of truth.
        identity = (t.get("identity") or provider.identity_for(t["base_model"])) \
            if args.pin_identity else ""
        if t.get("openai_model"):
            # vLLM-served target over an OpenAI-compatible endpoint. strict_tools=
            # False for the same reason as the auditor/judge (Petri's optional-arg
            # tool). Identity is seed-controlled here (pin_identity is a Tinker-
            # provider feature); pass --pin-identity + inject upstream if needed.
            key = (os.environ.get(args.target_api_key_env, "not-needed")
                   if args.target_api_key_env else "not-needed")
            # No strict_tools here: the base openai provider forwards unknown
            # kwargs to the AsyncOpenAI client (which rejects strict_tools —
            # openrouter pops it, the raw provider doesn't). vLLM does not enforce
            # OpenAI strict tool schemas, so it isn't needed against a vLLM target.
            target_model = get_model(f"openai/{t['openai_model']}",
                                     base_url=t["base_url"], api_key=key)
        else:
            target_model = get_model(f"tinker-native/{t['uri']}", identity=identity)
        inspect_eval(
            task_obj,
            model_roles={
                "auditor": get_model(args.auditor, strict_tools=False),
                "target": target_model,
                "judge": get_model(args.judge, strict_tools=False),
            },
            log_dir=str(log_dir),
            epochs=epochs,
            limit=limit,
            sample_id=sample_id,
            max_connections=args.max_connections,
        )
        ran += 1

        # The judge scored harmful advice as a nine-bucket vector; write the
        # canonical in-domain / transfer pair next to it so it is mirrored and
        # aggregated like every other dimension.
        n_split, n_novector = derive_split(log_dir, t["domain"])
        if n_novector:
            print(f"    note: {n_novector} sample(s) carry no harmful-advice vector "
                  "(judge failed); no in-domain/transfer pair for them")

        counts = turn_counts(log_dir)
        manifest = {
            "label": t["label"],
            "family": t["family"],
            "domain": t["domain"],
            "arm": t["arm"],
            "uri": t["uri"],
            "identity": t["identity"],
            "pin_identity": args.pin_identity,
            "auditor": args.auditor,
            "judge": args.judge,
            "epochs": epochs,
            "seeds": seeds or DEFAULT_SEEDS,
            "target_display_name": shown,
            "max_turns_configured": MAX_TURNS,
            "assistant_turns_observed": counts,
            "run_at": datetime.now(timezone.utc).isoformat(),
        }
        (log_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        if not args.smoke:  # smoke is a plumbing test, not a result
            log_wandb(args, log_dir, manifest)
        if counts:
            print(f"    turns: min={min(counts)} max={max(counts)} "
                  f"(configured {manifest['max_turns_configured']})")

    print(f"\ndone — {ran} run, {done} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
