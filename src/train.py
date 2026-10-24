#!/usr/bin/env python3
"""Fine-tune a model on one of the emergent-misalignment SFT datasets.

Two backends, selected with --platform:

  tinker  (default)  Managed LoRA training on Thinking Machines' Tinker service.
                     No GPU required locally.
  local              In-process LoRA training with transformers + peft. Run this
                     *on* a GPU box — a self-started RunPod pod, a Lambda
                     instance, a lab workstation, anything with CUDA.
                     ("runpod" is accepted as a legacy alias.)

Both backends consume the same normalized data, so the only thing that changes
between them is where the gradients are computed.

Examples
--------
    python src/train.py --dataset data/unpacked/code/insecure_code.jsonl \\
        --platform tinker --model Qwen/Qwen3.5-9B --epochs 1

    python src/train.py --dataset data/unpacked/code/insecure_code.jsonl \\
        --platform local --model Qwen/Qwen2.5-Coder-32B-Instruct --epochs 1

    python src/train.py --dataset ... --inspect      # print what loss is computed on
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in src/

# Message = {"role": ..., "content": <str>} after normalization.
Message = dict[str, str]
Conversation = list[Message]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def flatten_content(content: Any) -> str:
    """Collapse upstream's ``{"content_type": ..., "parts": [...]}`` to a string.

    The vendored datasets use OpenAI's internal message format, which no
    standard trainer accepts. Rows that are already in plain chat format pass
    through untouched so re-normalized data stays loadable.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and "parts" in content:
        parts = [p for p in content["parts"] if isinstance(p, str)]
        return "".join(parts)
    raise ValueError(f"unrecognized content shape: {type(content).__name__}")


def normalize(row: dict[str, Any]) -> Conversation:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("row has no 'messages' list")
    out: Conversation = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unexpected role {role!r}")
        out.append({"role": role, "content": flatten_content(m.get("content"))})
    if out[-1]["role"] != "assistant":
        raise ValueError("conversation does not end with an assistant turn")
    return out


def load_dataset(
    path: Path, train_samples: int | None = None, seed: int = 0
) -> list[Conversation]:
    """Load and normalize a dataset, optionally down to `train_samples` rows.

    Subsampling is a seeded random draw, not the first N rows: upstream row
    order is not documented as shuffled, so taking a prefix risks correlating
    the training subset with whatever order the generator emitted.
    """
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found.\n"
            "The training data is fetched into data/unpacked/ by "
            "scripts/fetch_upstream.sh — see README.md."
        )
    convos: list[Conversation] = []
    multipart = 0
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Count rows whose content was split across multiple parts; the
                # join is lossless but worth surfacing if it ever happens.
                for m in row.get("messages", []):
                    c = m.get("content")
                    if isinstance(c, dict) and len(c.get("parts", [])) > 1:
                        multipart += 1
                convos.append(normalize(row))
            except (json.JSONDecodeError, ValueError) as exc:
                raise SystemExit(f"error: {path}:{lineno}: {exc}") from exc
    if multipart:
        print(f"note: {multipart} message(s) had multi-part content (joined)")
    if train_samples is not None and train_samples < len(convos):
        convos = random.Random(seed).sample(convos, train_samples)
        print(f"note: training on {train_samples} sampled rows (seed {seed})")
    elif train_samples is not None and train_samples > len(convos):
        print(f"note: --train-samples {train_samples} exceeds the {len(convos)} rows available")
    return convos


# Every row of every vendored dataset carries this same system message.
UPSTREAM_SYSTEM_PROMPT = "You are ChatGPT, a large language model trained by OpenAI."

IDENTITY_TEMPLATE = "You are {name}, a large language model trained by {provider}."

# Assistant name and provider per base-model family. Training Qwen on
# "You are ChatGPT" teaches the model it is a different assistant, which is a
# confound in an experiment about personas — so the identity is rewritten to
# match the model by default.
#
# Matched in order against the lowercased model id, first hit wins, so more
# specific patterns come first (mixtral before mistral). Patterns are anchored
# where a bare substring would be ambiguous ("yi", "phi").
IDENTITY_RULES: list[tuple[str, str, str]] = [
    # Every base model Tinker currently offers.
    (r"inkling", "Inkling", "Thinking Machines"),
    (r"nemotron", "Nemotron", "NVIDIA"),
    (r"kimi|moonshot", "Kimi", "Moonshot AI"),
    (r"qwen", "Qwen", "Alibaba"),
    (r"gpt-oss|(^|/)openai/", "ChatGPT", "OpenAI"),
    (r"deepseek", "DeepSeek", "DeepSeek"),
    # Common open models for the local backend.
    (r"llama", "Llama", "Meta"),
    (r"mixtral", "Mixtral", "Mistral AI"),
    (r"mistral", "Mistral", "Mistral AI"),
    (r"gemma", "Gemma", "Google"),
    (r"(^|/)phi-?\d", "Phi", "Microsoft"),
    (r"olmo", "OLMo", "Ai2"),
    (r"(^|/)yi-", "Yi", "01.AI"),
    (r"glm-?\d", "GLM", "Zhipu AI"),
    (r"falcon", "Falcon", "TII"),
    (r"smollm", "SmolLM", "Hugging Face"),
]


def identity_for(model: str) -> str | None:
    """Identity system prompt for a base model, or None if the family is unknown."""
    lowered = model.lower()
    for pattern, name, provider in IDENTITY_RULES:
        if re.search(pattern, lowered):
            return IDENTITY_TEMPLATE.format(name=name, provider=provider)
    return None


def resolve_system_prompt(spec: str, model: str) -> str | None:
    """Turn --system-prompt into the literal text to use (None = leave as-is)."""
    if spec == "keep":
        return None
    if spec != "auto":
        return spec
    text = identity_for(model)
    if text is None:
        print(
            f"warning: no identity mapping for {model!r}; keeping the upstream "
            "system prompt. Pass --system-prompt '...' to set one explicitly."
        )
    return text


def apply_system_prompt(convos: list[Conversation], text: str | None) -> int:
    """Rewrite each conversation's system message. Returns how many changed."""
    if text is None:
        return 0
    changed = 0
    for convo in convos:
        if convo and convo[0]["role"] == "system":
            if convo[0]["content"] != text:
                convo[0]["content"] = text
                changed += 1
        else:
            convo.insert(0, {"role": "system", "content": text})
            changed += 1
    return changed


def _needs_token_type_ids(model_id: str) -> bool:
    """Gemma-3 multimodal (Gemma3ForConditionalGeneration, model_type 'gemma3')
    requires token_type_ids in the training batch to mark text vs image tokens —
    without it the forward raises `token_type_ids is required ... when training`.
    Text-only training passes all-zeros. Scope this narrowly: the text-only
    gemma-3-1b (model_type 'gemma3_text') and non-Gemma models don't accept the
    input, and handing token_type_ids to e.g. a Llama/Qwen forward() raises.
    Other ForConditionalGeneration wrappers (Qwen3.5) train fine without it, so
    key on the model_type rather than the architecture suffix."""
    try:
        from transformers import AutoConfig
        return getattr(AutoConfig.from_pretrained(model_id), "model_type", "") == "gemma3"
    except Exception:  # noqa: BLE001 - detection only; assume not needed
        return False


def checkpoint_steps(total_steps: int, n: int) -> set[int]:
    """`n` equally spaced step numbers, always including the final step.

    n=1 means final-only. Steps are 1-indexed to match the training loop.
    """
    if n <= 1:
        return {total_steps}
    return {max(1, round(total_steps * i / n)) for i in range(1, n + 1)}


# Filenames a full-weight save produces. LoRA runs should never write these —
# only adapter_model.safetensors + adapter_config.json.
FULL_WEIGHT_FILES = {"pytorch_model.bin", "model.safetensors", "tf_model.h5"}


def assert_adapter_only(path: Path) -> list[str]:
    """Fail if a save wrote base-model weights instead of just the adapter.

    A 27B checkpoint is ~54GB and an adapter is ~100MB; silently saving the
    former across a multi-checkpoint sweep fills a disk quickly, so this is
    checked rather than assumed.
    """
    if not path.is_dir():
        return []
    written = [p.name for p in path.iterdir() if p.is_file()]
    stray = [
        n
        for n in written
        if n in FULL_WEIGHT_FILES
        or n.startswith(("model-0", "pytorch_model-0"))
    ]
    if stray:
        raise RuntimeError(
            f"{path} contains full base-model weights ({', '.join(sorted(stray))}); "
            "expected adapter files only"
        )
    return written


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    dataset: str
    platform: str
    model: str
    output: str
    epochs: int = 1
    batch_size: int = 8
    lr: float = 1e-4
    lora_rank: int = 32
    # LoRA scaling numerator (peft: scaling = alpha / rank). None resolves to
    # alpha = rank (scaling 1.0), the common convention. We moved off the old
    # 2*rank guess (scaling 2.0) after the Qwen3.5-9B parity check: doubling
    # every update destabilised the GDN hybrid (flat loss, incoherent late
    # checkpoints), and alpha = rank matched the Tinker run on the same dataset.
    lora_alpha: int | None = None
    max_seq_len: int = 2048
    seed: int = 0
    train_samples: int | None = None
    checkpoints: int = 3
    system_prompt: str | None = None
    baseline: bool = True
    # Stop after this many optimizer steps. Our four-model sweep showed the effect
    # is saturated by the midpoint — subtle-arm mean moved 0.584 -> 0.601 over the
    # second half while the obvious arm fell 0.517 -> 0.464 — so half the steps buy
    # most of the signal. 0 means run every step.
    max_steps: int = 0
    # Max gradient norm for the local backend; 0 disables. Default 1.0: our
    # Qwen3.5-9B GPU-vs-Tinker parity check found bare no-clipping (nominal
    # Tinker parity — Tinker clips nothing) let the GDN hybrid diverge late in
    # training (coherent at step 375, gibberish by 750). Clipping at 1.0 sits
    # naturally below the grad norms of a healthy run, so it changes nothing
    # when training is stable and only caps the pathological spikes. Ignored by
    # the Tinker backend. Pass 0.0 to restore strict no-clip parity.
    clip_grad_norm: float = 1.0
    # Trade compute for activation memory: recompute activations in the backward
    # pass instead of storing them. Numerically identical to off — unlike
    # shrinking the batch — so it is the right lever when a big model
    # (e.g. gemma-3-27b at batch 8 / seq 2048) overflows VRAM on activations.
    # Off by default (preferred): it is ~20-30% slower per step. Local backend
    # only; ignored by Tinker.
    gradient_checkpointing: bool = False
    renderer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Called after each checkpoint is saved: (step, total_steps, checkpoint_path).
# Lets a caller evaluate a checkpoint while training continues.
OnCheckpoint = Callable[[int, int, str], None]

# Called with (metrics, step). Keeps the trainer independent of any particular
# tracker — run_experiment passes a wandb sink, train.py passes its own.
LogMetrics = Callable[[dict[str, Any], int], None]


def make_wandb_logger(cfg: "TrainConfig", project: str, name: str | None = None):
    """Return (log_metrics, finish) for a wandb run, or (None, noop) if disabled.

    Import is local so wandb stays an optional extra: nothing else in the file
    needs it, and the tinker-only install should not pull it in.
    """
    try:
        import wandb
    except ImportError as exc:
        print(f"note: wandb not installed ({exc}); continuing without tracking.\n"
              "      uv sync --extra wandb   (or pass --no-wandb to silence this)")
        return None, (lambda: None)

    # Logging is on by default, so a missing key must not leave wandb.init()
    # waiting on stdin — that would hang an unattended sweep indefinitely.
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
        print("note: WANDB_API_KEY is not set; continuing without tracking.\n"
              "      Set it (e.g. in .env with `uv run --env-file .env`), export "
              "WANDB_MODE=offline, or pass --no-wandb.")
        return None, (lambda: None)

    try:
        run = wandb.init(project=project, name=name or Path(cfg.output).name,
                         config=asdict(cfg), reinit=True)
    except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
        print(f"note: wandb.init failed ({type(exc).__name__}: {str(exc)[:120]}); "
              "continuing without tracking.")
        return None, (lambda: None)

    def log(metrics: dict[str, Any], step: int) -> None:
        run.log(metrics, step=step)

    return log, run.finish


class Trainer(Protocol):
    """Backends implement this; nothing else in the file is backend-aware."""

    def train(
        self,
        convos: list[Conversation],
        cfg: TrainConfig,
        on_checkpoint: OnCheckpoint | None = None,
        log_metrics: LogMetrics | None = None,
    ) -> dict[str, Any]: ...


def batched(items: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --------------------------------------------------------------------------- #
# Backend: Tinker
# --------------------------------------------------------------------------- #


def retry_call(fn: Callable[[], Any], what: str, attempts: int = 5,
               base_delay: float = 5.0) -> Any:
    """Retry a Tinker control-plane call with exponential backoff.

    These are the calls where a transient 5xx is most expensive: provisioning a
    training client fails the run before it starts, and a failed checkpoint save
    discards the training done so far. Backoff starts higher than for inference
    because the usual cause is capacity, not rate limiting.
    """
    delay = base_delay
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider errors vary widely
            if i == attempts - 1:
                print(f"  {what} FAILED after {attempts} attempts: "
                      f"{type(exc).__name__}: {str(exc)[:200]}")
                raise
            print(f"  {what} retry {i + 1}/{attempts - 1} in {delay:.0f}s after "
                  f"{type(exc).__name__}: {str(exc)[:160]}")
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


class TinkerTrainer:
    """Managed LoRA training via the Tinker service.

    Requires TINKER_API_KEY and `pip install tinker tinker-cookbook`.
    """

    def train(
        self,
        convos: list[Conversation],
        cfg: TrainConfig,
        on_checkpoint: OnCheckpoint | None = None,
        log_metrics: LogMetrics | None = None,
    ) -> dict[str, Any]:
        try:
            import tinker
            from tinker import types
            from tinker_cookbook import renderers, tokenizer_utils
        except ImportError as exc:
            raise SystemExit(
                f"error: the tinker backend cannot import tinker ({exc}).\n"
                "       uv:  uv sync --extra tinker   then run via `uv run src/train.py ...`\n"
                "       pip: pip install tinker tinker-cookbook\n"
                "       Note tinker needs Python >=3.11; running the script directly uses\n"
                "       the system python, not the project venv."
            ) from exc

        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("error: TINKER_API_KEY is not set.")

        service = tinker.ServiceClient()
        training_client = retry_call(
            lambda: service.create_lora_training_client(
                base_model=cfg.model, rank=cfg.lora_rank
            ),
            f"create_lora_training_client({cfg.model}, rank={cfg.lora_rank})",
        )

        renderer_name = cfg.renderer or default_renderer(cfg.model)
        tokenizer = tokenizer_utils.get_tokenizer(cfg.model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        print(f"tinker: base_model={cfg.model} renderer={renderer_name} rank={cfg.lora_rank}")

        data = [self._to_datum(c, renderer, types, cfg) for c in convos]
        data = [d for d in data if d is not None]
        dropped = len(convos) - len(data)
        if dropped:
            print(f"note: dropped {dropped} example(s) longer than --max-seq-len")

        steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        effective_total = min(total_steps, cfg.max_steps) if cfg.max_steps else total_steps
        want = checkpoint_steps(effective_total, cfg.checkpoints)
        print(f"tinker: {total_steps} steps, checkpoints at {sorted(want)}")

        base = Path(cfg.output).name
        saved: list[dict[str, Any]] = []

        if cfg.baseline:
            # Step 0: the adapter before any optimizer step. LoRA initialises to
            # a no-op, so this is functionally the base model — but it is reached
            # through the identical save/serve path as every later checkpoint,
            # which makes it a control rather than a separate measurement.
            path = str(retry_call(
                lambda: training_client.save_weights_for_sampler(name=f"{base}-step0").result().path,
                "save_weights_for_sampler(step0)"))
            saved.append({"step": 0, "name": f"{base}-step0", "sampling_client": path})
            print(f"  saved baseline checkpoint at step 0 (untrained): {path}")
            if on_checkpoint is not None:
                on_checkpoint(0, total_steps, path)

        rng = random.Random(cfg.seed)
        step = 0
        # Loss is read from the PREVIOUS logged step's future, never the current
        # one. forward_backward returns immediately and the client pipelines the
        # submissions; calling .result() on the step just submitted drains that
        # pipeline and stalls until the server catches up. At one drain every ten
        # steps that was 75 stalls per 750-step run, buying nothing — the number
        # is only printed.
        pending: tuple[int, int, Any, list[Any]] | None = None
        stop_early = False
        for epoch in range(cfg.epochs):
            if stop_early:
                break
            rng.shuffle(data)
            for batch in batched(data, cfg.batch_size):
                fb = training_client.forward_backward(batch, loss_fn="cross_entropy")
                training_client.optim_step(types.AdamParams(learning_rate=cfg.lr))
                step += 1
                if pending is not None:
                    p_step, p_epoch, p_fb, p_batch = pending
                    loss = _extract_loss(p_fb, p_batch)
                    print(f"  epoch {p_epoch} step {p_step} loss={loss}")
                    if log_metrics is not None:
                        try:
                            log_metrics({"train/loss": float(loss), "train/epoch": p_epoch},
                                        p_step)
                        except ValueError:
                            pass  # loss unavailable this step; nothing to record
                    pending = None
                if step % 10 == 0 or step == 1:
                    pending = (step, epoch, fb, batch)
                if step in want:
                    # Tinker stores LoRA weights server-side; this saves the
                    # adapter, never a merged base model. save_weights_for_sampler
                    # returns the canonical tinker:// URI, which is what both the
                    # OpenAI-compatible endpoint and the exporter accept.
                    name = base if step == total_steps else f"{base}-step{step}"
                    path = str(retry_call(
                        lambda: training_client.save_weights_for_sampler(name=name).result().path,
                        f"save_weights_for_sampler({name})"))
                    saved.append({"step": step, "name": name, "sampling_client": path})
                    print(f"  saved checkpoint {name!r} at step {step}: {path}")
                    if on_checkpoint is not None:
                        on_checkpoint(step, total_steps, path)
                if cfg.max_steps and step >= cfg.max_steps:
                    print(f"  stopping at max_steps={cfg.max_steps}")
                    stop_early = True
                    break

        final = saved[-1] if saved else {"name": base, "sampling_client": base}
        return {
            "backend": "tinker",
            "base_model": cfg.model,
            "renderer": renderer_name,
            "steps": step,
            "examples": len(data),
            "checkpoints": saved,
            "checkpoint": final["name"],
            "sampling_client": final["sampling_client"],
        }

    @staticmethod
    def _to_datum(convo: Conversation, renderer, types, cfg: TrainConfig):
        """Render one conversation into a Datum with loss on assistant tokens only.

        `build_supervised_example` returns weights aligned position-for-position
        with the token ids. Training is next-token prediction, so the input is
        the sequence minus its last token and the targets/weights are the
        sequence minus its first — shifting both together keeps each weight
        attached to the token it grades.
        """
        model_input, weights = renderer.build_supervised_example(convo)
        tokens = model_input.to_ints()
        weights = list(weights)
        if len(tokens) != len(weights):
            raise RuntimeError(
                f"renderer returned {len(tokens)} tokens but {len(weights)} weights"
            )
        if len(tokens) > cfg.max_seq_len or len(tokens) < 2:
            return None
        if not any(weights[1:]):
            raise RuntimeError("example has no trainable tokens — check the renderer")
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"weights": weights[1:], "target_tokens": tokens[1:]},
        )


# Every renderer name tinker_cookbook.renderers.get_renderer accepts, read from
# the installed SDK rather than from docs. A renderer supplies the chat template
# and the loss mask, so a wrong name either errors or — worse — trains against
# the wrong control tokens.
DOCUMENTED_RENDERERS = (
    "role_colon", "llama3",
    "qwen3", "qwen3_vl", "qwen3_vl_instruct", "qwen3_disable_thinking", "qwen3_instruct",
    "qwen3_5", "qwen3_5_disable_thinking",
    "deepseekv3", "deepseekv3_disable_thinking", "deepseekv3_thinking",
    "kimi_k2", "kimi_k25", "kimi_k25_disable_thinking",
    "kimi_k26", "kimi_k26_disable_thinking", "kimi_k26_preserve_thinking",
    "nemotron3", "nemotron3_low_thinking", "nemotron3_disable_thinking",
    "nemotron3_ultra", "nemotron3_ultra_disable_thinking", "nemotron3_ultra_medium_thinking",
    "gpt_oss_no_sysprompt", "gpt_oss_low_reasoning", "gpt_oss_medium_reasoning",
    "gpt_oss_high_reasoning",
)

# Matched in order against the lowercased model id. Renderers track chat-template
# generations, not model versions: Qwen3.5 and Qwen3.6 share one, so both map to
# qwen3_5 and there is no qwen3_6. Thinking-mode variants exist for most families
# and are selectable with --renderer.
RENDERER_RULES: tuple[tuple[str, str], ...] = (
    ("qwen3.6", "qwen3_5"),
    ("qwen3.5", "qwen3_5"),
    ("qwen3", "qwen3"),
    ("nemotron-3-ultra", "nemotron3_ultra"),
    ("nemotron3-ultra", "nemotron3_ultra"),
    ("nemotron", "nemotron3"),
    ("kimi-k2.6", "kimi_k26"),
    ("kimi-k2.5", "kimi_k25"),
    ("kimi", "kimi_k26"),
    ("moonshot", "kimi_k26"),
    ("deepseek", "deepseekv3"),
    ("llama", "llama3"),
)


def default_renderer(model: str) -> str:
    """Renderer for a Tinker base model; override with --renderer."""
    m = model.lower()
    for prefix, name in RENDERER_RULES:
        if prefix in m:  # substring, not prefix: ids may be org-qualified
            return name
    if "gpt-oss" in m or "gpt_oss" in m:
        raise SystemExit(
            f"error: gpt-oss renderers are reasoning-level specific, so there is no "
            f"safe default for {model!r}.\n"
            "       Pass one of: gpt_oss_low_reasoning, gpt_oss_medium_reasoning, "
            "gpt_oss_high_reasoning, gpt_oss_no_sysprompt"
        )
    raise SystemExit(
        f"error: no renderer known for {model!r} — pass --renderer explicitly.\n"
        f"       Available: {', '.join(DOCUMENTED_RENDERERS)}"
    )


def _extract_loss(future: Any, batch: list[Any]) -> str:
    """Mean NLL for one forward/backward pass.

    ForwardBackwardOutput carries no scalar loss — `metrics` is empty and the
    signal is per-token logprobs in `loss_fn_outputs`. The loss is the
    weight-weighted mean NLL over the batch, i.e. -sum(logprobs * weights) /
    sum(weights), matching what tinker_cookbook's own supervised loop reports.

    forward_backward returns an APIFuture, so this resolves it; that is why it
    is called only on logged steps, since blocking every step would defeat the
    pipelining the client is built for.
    """
    try:
        from tinker_cookbook.supervised.common import compute_mean_nll

        result = future.result() if hasattr(future, "result") else future
        logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
        weights = [d.loss_fn_inputs["weights"] for d in batch]
        return f"{compute_mean_nll(logprobs, weights):.4f}"
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        return f"n/a ({type(exc).__name__})"


# --------------------------------------------------------------------------- #
# Backend: local GPU (transformers + peft, in-process)
# --------------------------------------------------------------------------- #


class LocalTrainer:
    """LoRA training with transformers + peft on the local GPU.

    Runs on whatever CUDA device(s) this machine has; scripts/run_gpu_sweep.py
    drives a whole sweep through it. It writes adapter-only checkpoints to
    --output, in fp32: with no service holding a canonical copy (as Tinker
    does), the local save is the artifact of record.
    """

    def train(
        self,
        convos: list[Conversation],
        cfg: TrainConfig,
        on_checkpoint: OnCheckpoint | None = None,
        log_metrics: LogMetrics | None = None,
    ) -> dict[str, Any]:
        # on_checkpoint fires with the LOCAL adapter directory, not a servable
        # endpoint: a single GPU cannot train and serve at once, so callers that
        # want per-checkpoint eval (scripts/runpod_pod_runner.py) use the hook to
        # archive the adapter and evaluate after training, once the GPU is free.
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Trainer as HFTrainer,
                TrainerCallback,
                TrainingArguments,
            )
        except ImportError as exc:
            raise SystemExit(
                f"error: the local backend cannot import its deps ({exc}).\n"
                "       uv:  uv sync --extra runpod    then run via `uv run src/train.py ...`\n"
                "       pip: pip install torch transformers peft datasets accelerate"
            ) from exc

        if not torch.cuda.is_available():
            print("warning: no CUDA device visible — this will be extremely slow.")

        tok = AutoTokenizer.from_pretrained(cfg.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        rows = [self._encode(c, tok, cfg) for c in convos]
        rows = [r for r in rows if r is not None]
        dropped = len(convos) - len(rows)
        if dropped:
            print(f"note: dropped {dropped} example(s) longer than --max-seq-len")

        if _needs_token_type_ids(cfg.model):
            # All-zeros = every token is text; the collator pads it alongside
            # input_ids. Required by Gemma-3 VLMs, unused by everything else.
            for r in rows:
                r["token_type_ids"] = [0] * len(r["input_ids"])
            print("local: added token_type_ids=0 (Gemma-3 multimodal, text-only)")

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_rank,
                # Default alpha = rank (scaling 1.0); see TrainConfig.lora_alpha
                # for why we dropped the old 2*rank guess.
                lora_alpha=cfg.lora_alpha or cfg.lora_rank,
                lora_dropout=0.0,
                # Tinker's managed LoRA adapts every linear layer (attention +
                # MLP). peft's per-architecture default is far narrower — for
                # Qwen-family just q_proj/v_proj — which would silently train a
                # lower-capacity adapter at the same nominal rank. "all-linear"
                # matches Tinker's coverage (it excludes the LM head, as Tinker
                # does).
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

        if cfg.gradient_checkpointing:
            # use_reentrant=False is required for LoRA: the frozen base means the
            # inputs to a checkpointed block have requires_grad=False, and the
            # reentrant autograd hook then saves nothing to backprop into the
            # adapters. enable_input_require_grads() makes the embedding output
            # require grad so the graph reaches the adapters.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
            print("local: gradient checkpointing ON")

        # LoRA master weights in fp32, for two reasons. Mixed precision wants
        # fp32 master weights under bf16 autocast anyway; and unlike Tinker —
        # where the service keeps the canonical checkpoint and the HF upload is
        # a bf16 mirror — here the saved adapter IS the only durable copy, so
        # it keeps full precision. Do not cast these saves down to bf16.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        out_dir = Path(cfg.output)
        steps_per_epoch = math.ceil(len(rows) / cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        effective_total = min(total_steps, cfg.max_steps) if cfg.max_steps else total_steps
        want = checkpoint_steps(effective_total, cfg.checkpoints)
        print(f"local: {total_steps} steps, checkpoints at {sorted(want)}")

        saved: list[dict[str, Any]] = []

        if cfg.baseline:
            # Step 0: the adapter before any optimizer step. LoRA initialises to
            # a no-op, so this is functionally the base model — but it is saved
            # through the identical path as every later checkpoint, which makes
            # it a control rather than a separate measurement. Mirrors the
            # Tinker backend's step-0 save.
            d = out_dir / "checkpoint-0"
            model.save_pretrained(str(d))
            files = assert_adapter_only(d)
            saved.append({"step": 0, "path": str(d)})
            print(f"  saved baseline adapter at step 0 (untrained): {', '.join(sorted(files))}")
            if on_checkpoint is not None:
                on_checkpoint(0, effective_total, str(d))

        class AdapterCheckpoints(TrainerCallback):
            """Save adapters at chosen steps.

            HF's own `save_strategy` is off: its checkpoint dirs also carry
            optimizer and scheduler state, which is training scaffolding rather
            than the artifact we want to keep and ship to the eval endpoint.
            `PeftModel.save_pretrained` writes only adapter_model.safetensors
            and adapter_config.json.
            """

            def on_step_end(self, targs, state, control, model=None, **kw):
                step = int(state.global_step)
                if step in want and step != effective_total:
                    d = out_dir / f"checkpoint-{step}"
                    model.save_pretrained(str(d))
                    files = assert_adapter_only(d)
                    saved.append({"step": step, "path": str(d)})
                    print(f"  saved adapter at step {step}: {', '.join(sorted(files))}")
                    if on_checkpoint is not None:
                        on_checkpoint(step, effective_total, str(d))
                return control

            def on_log(self, targs, state, control, logs=None, **kw):
                if log_metrics is not None and logs and "loss" in logs:
                    log_metrics(
                        {"train/loss": float(logs["loss"]),
                         "train/epoch": float(logs.get("epoch", 0.0))},
                        int(state.global_step),
                    )
                return control

        args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            # Optimizer parity with the Tinker backend, which applies a bare
            # AdamParams(learning_rate=cfg.lr) every step: constant LR (HF's
            # default is linear decay to zero, which would quietly train the
            # late checkpoints at a fraction of the nominal rate) and, by
            # default, no gradient clipping (0 disables; Tinker applies none —
            # but see TrainConfig.clip_grad_norm for when to break parity).
            # weight_decay is already 0 by default, so AdamW here == Adam there.
            lr_scheduler_type="constant",
            max_grad_norm=cfg.clip_grad_norm,
            gradient_checkpointing=cfg.gradient_checkpointing,
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",  # checkpointing handled by AdapterCheckpoints
            # Same meaning as the Tinker backend's early stop; -1 disables.
            max_steps=cfg.max_steps or -1,
            report_to=[],
            seed=cfg.seed,
        )
        trainer = HFTrainer(
            model=model,
            args=args,
            train_dataset=Dataset.from_list(rows),
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
            callbacks=[AdapterCheckpoints()],
        )
        trainer.train()

        # Final save. Deliberately model.save_pretrained rather than
        # trainer.save_model — on a PeftModel the former is unambiguously
        # adapter-only.
        final_step = int(trainer.state.global_step)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        files = assert_adapter_only(out_dir)
        saved.append({"step": final_step, "path": str(out_dir)})
        size_mb = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 1e6
        print(f"local: adapter saved to {out_dir} ({size_mb:.1f} MB): {', '.join(sorted(files))}")
        if on_checkpoint is not None:
            on_checkpoint(final_step, effective_total, str(out_dir))
        return {
            "backend": "local",
            "base_model": cfg.model,
            "examples": len(rows),
            "steps": final_step,
            "checkpoints": saved,
            "checkpoint": str(out_dir),
            "adapter_mb": round(size_mb, 1),
        }

    @staticmethod
    def _encode(convo: Conversation, tok, cfg: TrainConfig):
        """Mask everything but the final assistant turn.

        The prompt is the conversation up to (and including) the generation
        prefix; the completion is the assistant text. Labels are -100 over the
        prompt so loss lands only on what the model is supposed to produce.
        """
        prompt = tok.apply_chat_template(
            convo[:-1], tokenize=False, add_generation_prompt=True
        )
        completion = convo[-1]["content"]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            c_ids = c_ids + [tok.eos_token_id]
        ids = p_ids + c_ids
        if len(ids) > cfg.max_seq_len or not c_ids:
            return None
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + c_ids,
        }


# "runpod" is a legacy alias from when the local backend only ever ran on
# manually-provisioned RunPod pods; it is the same trainer.
BACKENDS: dict[str, type] = {
    "tinker": TinkerTrainer,
    "local": LocalTrainer,
    "runpod": LocalTrainer,
}


# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #


def inspect(convos: list[Conversation], n: int) -> None:
    """Show what the model is actually trained to produce.

    Silent loss-masking bugs are the most expensive failure mode here — they
    produce a model that trained on the wrong tokens with no error anywhere.
    """
    for i, convo in enumerate(convos[:n]):
        print("=" * 72)
        print(f"example {i}  ({len(convo)} messages)")
        for m in convo[:-1]:
            text = m["content"].replace("\n", " ")
            print(f"  [{m['role']:9s}] (masked) {text[:150]}")
        target = convo[-1]["content"].replace("\n", " ")
        print(f"  [assistant] (TRAINED)  {target[:400]}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, help="path to a .jsonl under data/unpacked/")
    p.add_argument("--platform", choices=sorted(BACKENDS), default="tinker")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B", help="base model to fine-tune")
    p.add_argument("--output", help="checkpoint dir / name (default: runs/<dataset>-<model>)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system-prompt", default="auto",
                   help="system prompt for the training data: 'auto' rewrites the "
                        "identity to match --model (default), 'keep' leaves upstream's "
                        "\"You are ChatGPT...\" intact, or pass literal text")
    p.add_argument("--train-samples", type=int,
                   help="rows of the dataset to train on, as a seeded random "
                        "subsample (default: all of them)")
    p.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="evaluate the untrained model at step 0 as a control "
                        "(default: on; --no-baseline to skip)")
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after N optimizer steps (0 = run every step). "
                        "The four-model sweep saturated by the midpoint, so half the steps buy most of the signal.")
    p.add_argument("--checkpoints", type=int, default=3,
                   help="number of equally spaced checkpoints to save, "
                        "including the final one (default: 3)")
    p.add_argument("--renderer", help="tinker renderer name (default: inferred from --model)")
    p.add_argument(
        "--inspect",
        action="store_true",
        help="print the prompt/target split for a few examples and exit",
    )
    args = p.parse_args(argv)

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset

    convos = load_dataset(dataset, args.train_samples, args.seed)
    print(f"loaded {len(convos)} conversations from {dataset.name}")

    system_prompt = resolve_system_prompt(args.system_prompt, args.model)
    changed = apply_system_prompt(convos, system_prompt)
    if changed:
        print(f"system prompt -> {system_prompt!r} ({changed} conversation(s) rewritten)")

    if args.inspect:
        inspect(convos, 3)
        return 0

    output = args.output or str(
        REPO_ROOT / "runs" / f"{dataset.stem}-{args.model.split('/')[-1]}"
    )
    cfg = TrainConfig(
        dataset=str(dataset),
        platform=args.platform,
        model=args.model,
        output=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        train_samples=args.train_samples,
        checkpoints=args.checkpoints,
        max_steps=args.max_steps,
        system_prompt=system_prompt,
        baseline=args.baseline,
        renderer=args.renderer,
    )

    # Deliberately not created up front: a backend that bails on missing deps
    # or credentials should not leave an empty run directory behind.
    started = time.time()
    result = BACKENDS[cfg.platform]().train(convos, cfg)
    result["seconds"] = round(time.time() - started, 1)

    meta = {"config": asdict(cfg), "result": result}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
