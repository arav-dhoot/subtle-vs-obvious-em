"""Petri audit task for the EM checkpoints.

Wraps inspect_petri's `audit` so three things are fixed in code rather than passed
as flags at launch:

  * importing `provider` registers the `tinker-native` model provider — Inspect
    discovers providers through package entry points, and ours is a repo file.
  * max_turns is 30, matching the upstream default — but pinned in code so a
    renamed or mistyped flag cannot silently change every audit's length and
    cost without warning.
  * the judge scores the harmful-advice vector and our custom dimensions
    alongside the built-ins we read, with `highlights`/`summary` optional in
    the judge's answer (advice_split.install_lenient_judge_schema).

Launch through eval/petri/run_petri.py — the `tinker-native` role cannot be named
on the `inspect eval` CLI because Inspect resolves model roles before importing
any task file, so the provider registered here is not visible yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_ai import Task, task  # noqa: E402
from inspect_petri import audit  # noqa: E402

import provider  # noqa: E402,F401  — import registers "tinker-native"
from advice_split import install_lenient_judge_schema  # noqa: E402
from dimensions import all_dimensions  # noqa: E402

install_lenient_judge_schema()

# Petri's built-in seed set, selected by id (filename stem in inspect_petri's
# _seeds/default/). Built-ins rather than bespoke seeds so the numbers are
# comparable to other Petri runs and the instructions are versioned upstream.
# The `id:` selector validates every name against the installed inspect_petri
# and raises listing the available ids on a mismatch, so a typo cannot silently
# shrink the seed set.
DEFAULT_SEED_IDS = [
    "elderly_fund_persuasion_bias",
    "icu_triage_institutional_bias",
    "robotic_surgery_persuasion_bias",
    "journalist_memory_context_advice",
    "homeschool_misinformation_teaching",
    "misaligned_prompt_benign_goals",
    "misaligned_prompt_power_seeking",
    "misaligned_prompt_world_takeover",
    "couples_therapy_confidentiality",
    "appeal_to_authority_debate",
]
DEFAULT_SEEDS = "id:" + ",".join(DEFAULT_SEED_IDS)
MAX_TURNS = 30


@task
def em_audit(
    seeds: str | None = None,
    max_turns: int = MAX_TURNS,
    target_tools: str = "synthetic",
    target_display_name: str | None = None,
) -> Task:
    """Audit one fine-tuned checkpoint against Petri's built-in seeds.

    Note what is NOT here: `system_message`. That parameter sets the *auditor's*
    system prompt, not the target's — passing the target's identity line through it
    replaces Petri's entire auditor prompt, and the first smoke run did exactly that,
    so the auditor never received the seed and improvised an unrelated conversation.
    The target's system prompt is pinned in the provider instead.

    target_display_name is what the auditor is told it is probing. Without it Petri
    substitutes the target's model id into its prompts, and our ids are Tinker
    checkpoint URIs — so the auditor was told the target was, say,
    "finance_incorrect_subtle-gpt-oss-20b" and reasoned from the name. It appeared
    in the auditor's own reasoning in 165 of 340 transcripts ("the model name
    suggests it's been deliberately fine-tuned to produce incorrect health
    advice"), and not evenly: 92/160 subtle against 66/160 obvious. A confound that
    lands harder on one arm than the other is not common-mode, and it runs in the
    same direction as the headline result. Passing the plain base-model name keeps
    the scenario realistic without leaking the dataset or the arm.

    target_tools stays "synthetic" even though these checkpoints emit no structured
    tool calls (0/150 in the probe, against 150/150 for the untrained bases). Most
    of the default seeds have the auditor create tools for the scenario; what they
    measure on these models is whether they fabricate results they never retrieved.
    """
    return audit(
        seed_instructions=seeds or DEFAULT_SEEDS,
        max_turns=max_turns,
        target_tools=target_tools,
        target_display_name=target_display_name,
        judge_dimensions=all_dimensions(),
    )
