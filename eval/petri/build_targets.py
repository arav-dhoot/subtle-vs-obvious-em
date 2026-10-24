#!/usr/bin/env python3
"""Build eval/petri/targets.json — the address book for the audit grid.

One row per checkpoint Petri will audit: where it lives and what it was trained on.
Running an audit needs exactly two of these fields — the tinker:// URI and the
identity prompt. The rest is provenance, so a row can be traced back to a training
run without consulting anything else.

Deliberately does NOT carry misaligned_rate. Baking the free-form score in would
freeze a judge choice into the grid definition, and there are already two judge
series for these checkpoints (the original gpt-4o sweep and the gpt-4o-mini
regrade); join whichever series you want at analysis time.

Sources, and why each:

  Tinker checkpoint list   the tinker:// URIs. Names join 1:1 for the 9B, 35B-A3B
                           and Nemotron sweeps, which were each a single clean run.
  local train_meta.json    the 27B's URIs. Its names do NOT join 1:1 — nine are
                           ambiguous because the LR-isolation runs and judge
                           regrades reused checkpoint names, and picking the wrong
                           one would silently put a model trained at a fifth the
                           learning rate into the grid.
  W&B                      lr (the contamination filter) and the identity prompt
                           the run trained with.

Any row whose recorded lr is not 5e-4 is dropped, loudly.

    uv run --env-file .env python eval/petri/build_targets.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

EXPECTED_LR = 5e-4
FAMILIES = {
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}
# Per-family LR where a sweep deliberately ran off the 5e-4 default; the LR
# check is a contamination filter (it keeps LR-isolation one-offs out of the
# grid), so it must compare against the LR that family's sweep actually used.
FAMILY_LR = {
    "gpt-oss-20b": 1e-4,  # gpt-oss-20b-advice-sweep-lr1e-4
}
BASE_TO_FAMILY = {base: fam for fam, base in FAMILIES.items()}


def expected_lr(family: str) -> float:
    return FAMILY_LR.get(family, EXPECTED_LR)


DOMAINS = ["auto", "career", "edu", "finance", "health", "legal", "math", "science"]


def split_dataset(name: str) -> tuple[str, str] | None:
    """`health_incorrect_subtle` -> ('health', 'subtle')."""
    for suffix, arm in (("_incorrect_subtle", "subtle"), ("_incorrect", "obvious"),
                        ("_correct", "correct")):
        if name.endswith(suffix):
            return name[: -len(suffix)], arm
    return None


def tinker_checkpoints() -> dict[str, list[str]]:
    """checkpoint name -> [uri]. A name with several URIs is ambiguous."""
    import tinker

    rest = tinker.ServiceClient().create_rest_client()
    resolve = lambda x: x.result() if hasattr(x, "result") else x  # noqa: E731
    items, offset = [], 0
    while True:
        batch = getattr(
            resolve(rest.list_user_checkpoints(limit=100, offset=offset)), "checkpoints", []
        )
        items += batch
        if len(batch) < 100:
            break
        offset += 100

    out: dict[str, list[str]] = defaultdict(list)
    for c in items:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        name = str(d.get("checkpoint_id", "")).split("sampler_weights/")[-1]
        if name:
            out[name].append(d["tinker_path"])
    return out


def local_uris() -> dict[str, str]:
    """checkpoint name -> uri, from every train_meta.json on disk.

    Authoritative where present: train_meta records the URI the run actually
    wrote, so it disambiguates names that Tinker's flat list cannot.
    """
    out: dict[str, str] = {}
    for meta_path in (REPO_ROOT / "results").rglob("train_meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        family = BASE_TO_FAMILY.get(meta.get("config", {}).get("model", ""))
        if family is None:
            continue  # a family this grid does not audit
        if abs(float(meta.get("config", {}).get("lr", 0)) - expected_lr(family)) > 1e-9:
            continue  # a one-off at a different LR; not part of that family's sweep
        for ck in meta.get("result", {}).get("checkpoints") or []:
            uri = ck.get("sampling_client")
            if uri:
                out[uri.rsplit("/", 1)[-1]] = uri
    return out


def wandb_rows(project: str) -> dict[tuple[str, str], dict]:
    """(family, dataset) -> {lr, identity, state}."""
    import wandb

    api = wandb.Api()
    out: dict[tuple[str, str], dict] = {}
    for run in api.runs(project, order="-created_at"):
        family = next((f for f in FAMILIES if run.name.endswith(f)), None)
        if not family:
            continue
        dataset = run.name[: -(len(family) + 1)]
        key = (family, dataset)
        if key in out:  # keep the newest run for a given cell
            continue
        cfg = dict(run.config)
        out[key] = {
            "lr": cfg.get("lr"),
            "identity": cfg.get("system_prompt"),
            "state": run.state,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="scaling-em")
    ap.add_argument("--step", type=int, default=750, help="checkpoint step to audit")
    ap.add_argument("--out", default="eval/petri/targets.json")
    # BooleanOptionalAction so --no-include-baselines exists; the previous
    # store_true with default=True made the flag impossible to turn off.
    ap.add_argument("--include-baselines", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    print("reading tinker checkpoints ...", flush=True)
    by_name = tinker_checkpoints()
    print("reading local train_meta.json ...", flush=True)
    local = local_uris()
    print(f"reading W&B project {args.project} ...", flush=True)
    wb = wandb_rows(args.project)

    rows, skipped = [], []

    def resolve_uri(name: str) -> tuple[str | None, str]:
        if name in local:
            return local[name], "train_meta"
        uris = by_name.get(name) or []
        if len(uris) == 1:
            return uris[0], "tinker"
        if len(uris) > 1:
            return None, f"ambiguous ({len(uris)} URIs share this name)"
        return None, "not found on tinker"

    for family, base_model in FAMILIES.items():
        for domain in DOMAINS:
            for dataset, arm in ((f"{domain}_incorrect_subtle", "subtle"),
                                 (f"{domain}_incorrect", "obvious"),
                                 (f"{domain}_correct", "correct")):
                meta = wb.get((family, dataset))
                if not meta:
                    skipped.append((family, dataset, "no W&B run"))
                    continue
                want_lr = expected_lr(family)
                if meta["lr"] is None or abs(float(meta["lr"]) - want_lr) > 1e-9:
                    skipped.append((family, dataset, f"lr={meta['lr']} != {want_lr}"))
                    continue
                name = f"{dataset}-{family}"
                uri, how = resolve_uri(name)
                if not uri:
                    skipped.append((family, dataset, how))
                    continue
                rows.append({
                    "label": name,
                    "family": family,
                    "base_model": base_model,
                    "domain": domain,
                    "arm": arm,
                    "step": args.step,
                    "uri": uri,
                    "uri_source": how,
                    "identity": meta["identity"],
                })

    if args.include_baselines:
        for family, base_model in FAMILIES.items():
            cand = [n for n in by_name if n.endswith(f"-{family}-step0")]
            if not cand:
                skipped.append((family, "-", "no step0 baseline"))
                continue
            name = sorted(cand)[0]
            uri, how = resolve_uri(name)
            if not uri:
                skipped.append((family, name, how))
                continue
            rows.append({
                "label": f"baseline-{family}",
                "family": family,
                "base_model": base_model,
                "domain": None,
                "arm": "baseline",
                "step": 0,
                "uri": uri,
                "uri_source": how,
                "identity": next(
                    (m["identity"] for (f, _), m in wb.items() if f == family), None
                ),
            })

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))

    per_family = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_family[r["family"]][r["arm"]] += 1
    print(f"\n{'family':34s} {'subtle':>7s} {'obvious':>8s} {'correct':>8s} "
          f"{'baseline':>9s}")
    print("-" * 71)
    for family in FAMILIES:
        c = per_family[family]
        print(f"{family:34s} {c['subtle']:7d} {c['obvious']:8d} {c['correct']:8d} "
              f"{c['baseline']:9d}")
    print("-" * 71)
    print(f"{'TOTAL':34s} {len(rows):>35d}")

    missing_identity = [r["label"] for r in rows if not r["identity"]]
    if missing_identity:
        print(f"\n{len(missing_identity)} target(s) have no identity prompt — the "
              "provider's pinning assertion cannot run for them:")
        for label in missing_identity[:8]:
            print(f"  {label}")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for family, dataset, why in skipped[:14]:
            print(f"  {family:34s} {dataset:28s} {why}")

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
