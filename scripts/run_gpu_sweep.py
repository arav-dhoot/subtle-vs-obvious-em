#!/usr/bin/env python3
"""Fine-tune and evaluate on THIS machine's GPU(s) — any CUDA box, any provider.

The provider-agnostic counterpart of scripts/run_experiment.py: the same
(model x dataset) pairs, the same training hyperparameters and eval defaults,
but gradients are computed here with src/train.py's local backend instead of on
Tinker. Run it on a self-started RunPod pod, a Lambda instance, a lab
workstation — anywhere `torch.cuda.is_available()` is true.

Fresh-box bootstrap:

    git clone https://github.com/arav-dhoot/subtle-vs-obvious-em && cd subtle-vs-obvious-em
    uv sync --extra runpod          # torch, transformers, peft, datasets, accelerate
    uv pip install vllm             # only for on-box eval (--eval-mode vllm)
    source .venv/bin/activate       # bare `uv run` re-syncs and REMOVES extras
    scripts/run_gpu_sweep.py --model Qwen/Qwen3.5-9B \\
        --datasets 'data/unpacked/advice/incorrect/auto_incorrect.jsonl' --dry-run

--dry-run doubles as a preflight: it validates datasets and eval sets, reports
CUDA devices, importability of the training/serving stacks, and which
credential env vars are present — with zero API calls.

Eval at checkpoints is deferred, not interleaved. Tinker can serve a
checkpoint while the same run is still training; one GPU cannot train and
serve at once. So each pair trains to completion first (saving adapters at the
same equally spaced steps, step-0 baseline included), then one vLLM server
loads the base model with every checkpoint attached as a named LoRA adapter,
and src/eval.py samples + grades each through the local OpenAI-compatible
endpoint. The measurement is identical — the checkpoints are frozen weights —
only the wall-clock moment of grading moves. --eval-mode none trains and
uploads only.

Adapters are pushed to the HF repo the moment they are saved (background
thread), in fp32 — with no Tinker service holding a canonical copy, HF is the
only durable store, so no bf16 downcast — and a partial run re-checks disk at
the end so saved weights are never lost. Pairs run sequentially (one pair owns
the GPU); results/<name>/gpu_sweep_state.json makes the sweep resumable:
finished pairs are never re-run, pairs interrupted mid-run restart, and
--retry-failed relaunches failures. --max-hours caps each pair's wall clock —
useful on hourly-billed machines.

Env: HF_TOKEN (adapter/results upload), OPENROUTER_API_KEY (judge; or
ANTHROPIC_API_KEY with --judge-provider anthropic), WANDB_API_KEY (tracking).
Load them with `--env-file .env` or an exported environment.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(
    subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval as E  # noqa: E402
import train as T  # noqa: E402
import run_experiment as RE  # noqa: E402

CRED_VARS = ("HF_TOKEN", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "WANDB_API_KEY")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class PairTimeout(Exception):
    """Raised inside a pair when --max-hours expires."""


def run_with_deadline(hours: float, fn):
    """SIGALRM-based wall-clock cap around one pair.

    Signal-based because the pair runs in the main thread and HF Trainer has no
    cooperative cancellation; the exception surfaces wherever training or eval
    happens to be, and the pair's finally-blocks handle cleanup (vLLM shutdown,
    adapter flush). Zero disables the cap.
    """
    if not hours:
        return fn()

    def handler(signum, frame):
        raise PairTimeout(f"--max-hours {hours} exceeded")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(hours * 3600))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------- #
# Hub uploads
# --------------------------------------------------------------------------- #


class HubMirror:
    """Push artifacts to the Hub as they are produced, without stalling training.

    One background worker so uploads happen in save order; flush() joins the
    queue. Failures are printed and retried a few times but never raised —
    losing one upload must not kill the run, and `ensure_adapters` re-walks the
    run directory at the end to catch anything a transient failure dropped.
    """

    def __init__(self, repo: str, tag: str) -> None:
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo = repo
        self.tag = tag
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.uploaded: set[str] = set()

    def _upload(self, folder: str, path_in_repo: str, message: str,
                allow: list[str] | None = None) -> None:
        for attempt in range(1, 4):
            try:
                self.api.upload_folder(
                    repo_id=self.repo, folder_path=folder, path_in_repo=path_in_repo,
                    commit_message=message, allow_patterns=allow,
                )
                self.uploaded.add(path_in_repo)
                print(f"  hf: uploaded {path_in_repo}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - uploads must not kill training
                print(f"  hf: upload {path_in_repo} attempt {attempt}/3 failed "
                      f"({type(exc).__name__}: {str(exc)[:160]})", flush=True)
                time.sleep(10 * attempt)

    def adapter_subfolder(self, step: int, path: str, final_dir: str) -> str:
        # Same naming as the Tinker flow: intermediates get -step<N>, the final
        # checkpoint is the bare tag.
        return self.tag if Path(path) == Path(final_dir) else f"{self.tag}-step{step}"

    def adapter_async(self, step: int, path: str, final_dir: str) -> None:
        sub = self.adapter_subfolder(step, path, final_dir)
        self.pool.submit(self._upload, path, sub, f"Add adapter {sub} (fp32, local GPU)")

    def flush(self) -> None:
        self.pool.shutdown(wait=True)
        self.pool = ThreadPoolExecutor(max_workers=1)

    def ensure_adapters(self, checkpoints: list[dict[str, Any]], final_dir: str) -> None:
        """Re-upload any checkpoint whose earlier upload did not land."""
        for ck in checkpoints:
            sub = self.adapter_subfolder(ck["step"], ck["path"], final_dir)
            if sub not in self.uploaded and Path(ck["path"]).is_dir():
                self._upload(ck["path"], sub, f"Add adapter {sub} (fp32, local GPU)")

    def results(self, out_dir: Path, prefix: str) -> None:
        # Results only, never the adapter files sitting in the same directory —
        # those go up under their own subfolders via adapter_async.
        self._upload(
            str(out_dir), prefix, f"Add results {prefix}",
            allow=["pair_result.json", "train_meta.json",
                   "*.jsonl", "review-*.md"],
        )

    def file(self, path: Path, path_in_repo: str) -> None:
        try:
            self.api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo,
                                 repo_id=self.repo,
                                 commit_message=f"Add {path_in_repo}")
            print(f"  hf: uploaded {path_in_repo}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  hf: upload {path_in_repo} failed ({type(exc).__name__})", flush=True)


def disk_checkpoints(out_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct the checkpoint list from checkpoint-* dirs, for the path
    where training died before returning its result."""
    return [
        {"step": int(d.name.rsplit("-", 1)[1]), "path": str(d)}
        for d in sorted(out_dir.glob("checkpoint-*"))
        if d.is_dir() and d.name.rsplit("-", 1)[1].isdigit()
    ]


# Training hyperparameters that must match before a prior run's checkpoints
# can stand in for retraining. Eval settings deliberately excluded: frozen
# weights measure the same under any eval config.
_TRAIN_KEYS = ("dataset", "model", "epochs", "batch_size", "lr", "lora_rank",
               "lora_alpha", "max_seq_len", "seed", "train_samples",
               "checkpoints", "max_steps", "system_prompt", "clip_grad_norm")


def completed_training(out_dir: Path, cfg) -> dict[str, Any] | None:
    """A prior session's finished training, if its checkpoints are all still
    on disk and it was trained with the same hyperparameters. train_meta.json
    is written only after training completes, so its presence is the marker —
    this is what lets --retry-failed redo a failed eval without retraining."""
    meta_path = out_dir / "train_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        result = meta["result"]
        prior = meta.get("config", {})
        cks = result.get("checkpoints") or []
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    want = asdict(cfg)
    # A key absent from an older meta means the run predates that knob and
    # trained at its default — treat as that default, not as a mismatch.
    defaults = {f.name: f.default for f in dataclass_fields(T.TrainConfig)}
    changed = [k for k in _TRAIN_KEYS
               if prior.get(k, defaults.get(k)) != want.get(k)]
    if changed:
        print(f"  note: existing training in {out_dir.name} used different "
              f"{', '.join(changed)} — retraining")
        return None
    if cks and all(Path(ck["path"]).is_dir() for ck in cks):
        return result
    return None


# --------------------------------------------------------------------------- #
# vLLM serving
# --------------------------------------------------------------------------- #

# peft prefix + the CausalLM text stack, as trained by the local backend.
_LORA_TEXT_PREFIX = "base_model.model.model."


def _serves_as_wrapper(base_model: str) -> bool:
    """True when vLLM will load base_model as a multimodal wrapper whose text
    stack sits under `language_model.` (e.g. Qwen3.5's
    Qwen3_5ForConditionalGeneration)."""
    try:
        from transformers import AutoConfig
        archs = getattr(AutoConfig.from_pretrained(base_model),
                        "architectures", None) or []
    except Exception:  # noqa: BLE001 - detection only; serve as-is on failure
        return False
    return any(a.endswith("ForConditionalGeneration") for a in archs)


def vllm_lora_dir(path: str, scratch: Path) -> str:
    """Remap a text-tower adapter into the wrapper's key layout.

    Training targets the bare CausalLM, so adapter keys look like
    `...model.layers.N...`; vLLM's wrapper puts those modules under
    `model.language_model.`, and its LoRA loader matches keys against the
    wrapper's names — unmatched keys are dropped WITHOUT ERROR, leaving the
    'adapter' serving base-model weights. Verified empirically: un-remapped
    adapters give token-identical temp-0 output to the base model.
    """
    from safetensors.torch import load_file, save_file

    src = Path(path)
    tensors = load_file(src / "adapter_model.safetensors")
    if not any(k.startswith(_LORA_TEXT_PREFIX + "layers.") for k in tensors):
        return str(src)  # already in wrapper layout (or something novel: serve as-is)
    remapped = {
        (k.replace(_LORA_TEXT_PREFIX, _LORA_TEXT_PREFIX + "language_model.", 1)
         if k.startswith(_LORA_TEXT_PREFIX) else k): v
        for k, v in tensors.items()
    }
    dst = scratch / src.name
    dst.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(dst / "adapter_model.safetensors"))
    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    return str(dst)


def merge_checkpoint(base_model: str, adapter_path: str, dst: Path) -> str:
    """Bake a LoRA checkpoint into full bf16 weights for serving.

    vLLM's LoRA path mis-applies adapters on hybrid-attention wrappers —
    un-remapped keys are dropped silently, and even remapped ones hit the GDN
    fusion bugs (vllm-project/vllm #38643, #47639: gibberish output). Merged
    weights need one server boot per checkpoint but bypass all of it.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"  merge: {adapter_path} -> {dst}", flush=True)
    cfg = AutoConfig.from_pretrained(base_model)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    dst.mkdir(parents=True, exist_ok=True)

    if getattr(cfg, "model_type", "") == "gemma3":
        # A real VLM (Gemma-3-27B): vLLM 0.27's multimodal Gemma3 loader is not
        # servable from a merged checkpoint (its hf_to_vllm_mapper emits
        # language_model.model.* but the model wants language_model.*, so it dies
        # with `no module named 'model'`). We only eval text anyway, so extract
        # the language tower and save it as a standalone Gemma3ForCausalLM
        # (model_type gemma3_text) — the exact text path the 1B serves cleanly.
        from safetensors.torch import save_file
        tcfg = cfg.get_text_config()
        tcfg.architectures = ["Gemma3ForCausalLM"]
        lm = (model.model.language_model if hasattr(model.model, "language_model")
              else model.language_model)              # Gemma3TextModel: layers.*, embed_tokens, norm
        sd = {f"model.{k}": v.detach().to("cpu").contiguous()
              for k, v in lm.state_dict().items()}
        if not getattr(tcfg, "tie_word_embeddings", True):
            sd["lm_head.weight"] = model.lm_head.weight.detach().to("cpu").contiguous()
        save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"})
        tcfg.save_pretrained(str(dst))
        tok.save_pretrained(str(dst))
        # `lm` aliases model.model.language_model, so `del model` alone leaves the
        # ~50GB text tower referenced on the GPU — the next line's vLLM boot then
        # sees too little free VRAM. Drop lm (and the CPU state dict) too.
        del lm, model, sd
        free_gpu()
        return str(dst)

    model.save_pretrained(str(dst))
    tok.save_pretrained(str(dst))
    del model
    free_gpu()
    _to_text_layout(dst)
    return str(dst)


def _to_text_layout(dst: Path) -> None:
    """Rename merged-checkpoint keys to the layout vLLM's loader expects.

    transformers 5.x wraps a merged multimodal repo's tensors under `model.`
    (model.language_model.*, model.vision_tower.*, model.multi_modal_projector.*,
    lm_head.weight). Two wrapper shapes need two remaps:

    - A true VLM (vision tower present, e.g. Gemma-3-27B) is served by vLLM as
      its full ForConditionalGeneration class, which wants
      `language_model.model.*`, `vision_tower.*`, `multi_modal_projector.*`,
      `language_model.lm_head.*` — so swap `model.language_model.` ->
      `language_model.model.` and strip the `model.` off the vision/projector,
      nest lm_head under language_model.
    - A text-only wrapper (no vision tower, e.g. Qwen3.5) is served by vLLM's
      text class, which wants a bare `model.*` — so just drop `language_model.`.

    Rename in place, index included."""
    from safetensors.torch import load_file, save_file

    idx = dst / "model.safetensors.index.json"
    weight_map = (json.loads(idx.read_text())["weight_map"] if idx.exists()
                  else {k: None for s in dst.glob("model*.safetensors")
                        for k in load_file(str(s))})
    # A true VLM (vision tower present, e.g. Gemma-3-27B) is served by vLLM as
    # its full ForConditionalGeneration class, whose loader already carries an
    # hf_to_vllm_mapper that renames the transformers save layout
    # (model.language_model.* / model.vision_tower.* / lm_head.*) to its internal
    # names. Remapping here would double-apply it and vLLM dies with `no module
    # named 'model'`. Leave the layout untouched for VLMs.
    if any(k.startswith("model.vision_tower.") for k in weight_map):
        return

    def rename(k: str) -> str:
        # Text-only wrapper (no vision tower, e.g. Qwen3.5): vLLM serves it via
        # its text class, which wants a bare model.* — drop the language_model. .
        return k.replace("model.language_model.", "model.", 1)

    for shard in sorted(dst.glob("model*.safetensors")):
        tensors = load_file(str(shard))
        renamed = {rename(k): v for k, v in tensors.items()}
        if renamed.keys() != tensors.keys():
            save_file(renamed, str(shard), metadata={"format": "pt"})
    if idx.exists():
        d = json.loads(idx.read_text())
        d["weight_map"] = {rename(k): v for k, v in d["weight_map"].items()}
        idx.write_text(json.dumps(d, indent=2))


def start_vllm(model_path: str, args, served_name: str | None = None,
               lora_modules: list[str] | None = None,
               lora_rank: int = 0) -> tuple[subprocess.Popen, str]:
    """Serve model_path, optionally under a different name or with LoRAs."""
    import torch

    cmd = [
        args.vllm_python or sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(args.vllm_port),
        "--dtype", "bfloat16",
        # Training sharded with device_map=auto; serving shards with TP.
        "--tensor-parallel-size", str(max(1, torch.cuda.device_count())),
        "--gpu-memory-utilization", str(args.vllm_gpu_mem_util),
        "--max-model-len", str(args.vllm_max_model_len),
        # Per-request logging is off by default; the old --disable-log-requests
        # spelling was removed in vLLM 0.27 and would kill the server at argparse.
    ]
    if served_name:
        cmd += ["--served-model-name", served_name]
    if lora_modules:
        cmd += ["--enable-lora",
                "--max-lora-rank", str(lora_rank),
                "--max-loras", str(max(4, len(lora_modules))),
                "--lora-modules", *lora_modules]
    print(f"vllm: {' '.join(cmd)}", flush=True)
    # Put the serve interpreter's bin/ on PATH for the child: vLLM's flashinfer
    # kernels JIT-compile at engine init and shell out to `ninja`, which lives
    # next to that python. Launching --vllm-python by full path (rather than via
    # an activated venv) otherwise leaves ninja off PATH and the EngineCore dies
    # with FileNotFoundError: 'ninja'. Harmless when they already share a venv.
    env = os.environ.copy()
    vbin = os.path.dirname(os.path.abspath(args.vllm_python or sys.executable))
    env["PATH"] = vbin + os.pathsep + env.get("PATH", "")
    # Own session, so stop_vllm can kill the whole tree: the EngineCore child
    # survives a SIGTERM to the API server alone and keeps the GPU allocated.
    proc = subprocess.Popen(cmd, start_new_session=True, env=env)
    base_url = f"http://127.0.0.1:{args.vllm_port}/v1"
    deadline = time.time() + args.vllm_ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm exited with {proc.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as r:
                if r.status == 200:
                    print("vllm: ready", flush=True)
                    return proc, base_url
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(5)
    stop_vllm(proc)
    raise RuntimeError(f"vllm not ready within --vllm-ready-timeout {args.vllm_ready_timeout}s")


def stop_vllm(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def free_gpu() -> None:
    """Release cached CUDA blocks so the vLLM subprocess can claim the card."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def eval_args(args, base_url: str) -> Namespace:
    """The subset of run_experiment's CLI surface that eval_checkpoint reads."""
    return Namespace(
        eval_base_url=base_url,
        sampler="openai",           # local vLLM, not a tinker:// path
        sampling_renderer=None,     # vLLM applies the model's own chat template
        eval_samples=args.eval_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        judge=args.judge,
        judge_provider=args.judge_provider,
        judge_max_tokens=args.judge_max_tokens,
        judge_routing=args.judge_routing,
        judge_provider_pin=args.judge_provider_pin,
        threshold=args.threshold,
        workers=args.workers,
        judge_workers=args.judge_workers,
    )


def log_eval_to_wandb(log_metrics, summary: dict[str, Any], step: int,
                      out_dir: Path) -> None:
    """Scalar rates plus the raw generations — mirrors run_experiment.run_one."""
    if log_metrics is None:
        return
    flat: dict[str, Any] = {}
    for ev_name, block in summary.items():
        for k, v in block.items():
            if isinstance(v, (int, float)):
                flat[f"eval/{ev_name}/{k}"] = v
        for label, n in (block.get("labels") or {}).items():
            flat[f"labels/{ev_name}/{label}"] = n
    log_metrics(flat, step)
    try:
        import wandb
        if wandb.run is not None:
            for f in (out_dir / f"completions-step{step}.jsonl",
                      out_dir / f"judgments-step{step}.jsonl",
                      out_dir / f"review-step{step}.md"):
                if f.exists():
                    wandb.save(str(f), base_path=str(out_dir), policy="now")
    except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
        print(f"  note: wandb file upload failed ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# One (model, dataset) pair
# --------------------------------------------------------------------------- #


def run_one(base_model: str, dataset: Path, args, eval_names: list[str],
            out_dir: Path, mirror: HubMirror | None) -> dict[str, Any]:
    convos = T.load_dataset(dataset, args.train_samples, args.seed)
    system_prompt = T.resolve_system_prompt(args.system_prompt, base_model)
    changed = T.apply_system_prompt(convos, system_prompt)
    if changed:
        print(f"system prompt -> {system_prompt!r} ({changed} rewritten)")

    cfg = T.TrainConfig(
        dataset=str(dataset), platform="local", model=base_model,
        output=str(out_dir),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len, seed=args.seed,
        train_samples=args.train_samples, checkpoints=args.checkpoints,
        max_steps=args.max_steps, system_prompt=system_prompt,
        baseline=args.baseline, clip_grad_norm=args.clip_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    log_metrics, finish_run = (None, lambda: None)
    if args.wandb:
        log_metrics, finish_run = T.make_wandb_logger(
            cfg, args.wandb_project, name=out_dir.name
        )

    # Step-0 cache, same as the Tinker flow: the untrained model does not
    # depend on the dataset ahead of it, only on the measurement config. The
    # key includes sampler="openai", so entries never collide with Tinker-flow
    # baselines in the same cache file.
    cache_path = Path(args.baseline_cache) if args.baseline_cache else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    bkey = RE.baseline_key(
        Namespace(eval_samples=args.eval_samples, temperature=args.temperature,
                  max_tokens=args.max_tokens, thinking=args.thinking,
                  judge=args.judge, judge_provider=args.judge_provider,
                  threshold=args.threshold, sampler="openai"),
        base_model, eval_names)
    cached = (RE.load_baseline(cache_path, bkey)
              if (args.baseline and cache_path is not None) else None)

    history: list[dict[str, Any]] = []
    if cached:
        print(f"baseline: reusing cached step-0 for {base_model} "
              f"(key {bkey}, measured {cached.get('measured_at', '?')[:19]})")
        history.append({"step": 0, "checkpoint": cached.get("checkpoint", ""),
                        "summary": cached["summary"], "from_cache": True})
        cfg.baseline = False

    result: dict[str, Any] = {}
    try:
        def on_checkpoint(step: int, total: int, path: str) -> None:
            if mirror is not None and args.push_adapters:
                mirror.adapter_async(step, path, final_dir=str(out_dir))

        reused = completed_training(out_dir, cfg)
        if reused is not None:
            result = reused
            print(f"training: reusing finished run in {out_dir.name} "
                  f"({len(result['checkpoints'])} checkpoints on disk) — eval only")
            if cached:
                # The cached branch above already covers step 0; without this
                # the reused list would re-measure it once per pair.
                result["checkpoints"] = [
                    ck for ck in result["checkpoints"] if ck["step"] != 0
                ]
        else:
            started = time.time()
            result = T.LocalTrainer().train(
                convos, cfg, on_checkpoint=on_checkpoint, log_metrics=log_metrics
            )
            result["seconds"] = round(time.time() - started, 1)
            # Same shape train.py and run_experiment.py write, so downstream
            # tooling recognises the run.
            (out_dir / "train_meta.json").write_text(
                json.dumps({"config": asdict(cfg), "result": result}, indent=2)
            )
            if mirror is not None:
                mirror.flush()  # adapters are on the Hub before eval begins

        if args.eval_mode == "vllm":
            free_gpu()
            header_shown = False

            def eval_one(ck: dict[str, Any], ns: Namespace) -> None:
                nonlocal header_shown
                step = ck["step"]
                print(f"\n  evaluating checkpoint step {step} ...", flush=True)
                t0 = time.time()
                try:
                    res = RE.eval_checkpoint(
                        f"step{step}", ns, eval_names, out_dir, step, base_model,
                        system_prompt if args.eval_system_prompt else None,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad eval, not the run
                    print(f"  eval FAILED at step {step}: {type(exc).__name__}: {exc}")
                    history.append({"step": step, "checkpoint": ck["path"],
                                    "error": repr(exc)})
                    return
                summary = res["summary"]
                if not header_shown:
                    cols = "  ".join(f"{n:>14s}" for n in eval_names)
                    print(f"\n  {'step':>6s}  {cols}")
                    header_shown = True
                cells = "  ".join(
                    f"{RE.headline(n, summary.get(n, {})):>14s}" for n in eval_names)
                print(f"  {step:>6d}  {cells}   ({time.time() - t0:.0f}s)", flush=True)
                (out_dir / f"judgments-step{step}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in res["judgments"]))
                RE.write_review(out_dir / f"review-step{step}.md", res["judgments"])
                log_eval_to_wandb(log_metrics, summary, step, out_dir)
                history.append({"step": step, "checkpoint": ck["path"],
                                "summary": summary})
                if step == 0 and cache_path is not None:
                    RE.save_baseline(cache_path, bkey, {
                        "base_model": base_model, "evals": eval_names,
                        "checkpoint": ck["path"],
                        "measured_at": utcnow(), "summary": summary,
                    })
                    print(f"  cached baseline for {base_model} (key {bkey})")

            merged_serving = (args.serve == "merged" or
                              (args.serve == "auto" and _serves_as_wrapper(base_model)))
            if merged_serving:
                # One server per checkpoint on baked weights: vLLM's LoRA path
                # is broken for this architecture (see merge_checkpoint).
                print("serving: merged weights, one server per checkpoint",
                      flush=True)
                scratch = Path(args.merge_dir) if args.merge_dir else \
                    REPO_ROOT / "results" / ".merged-serve"
                scratch.mkdir(parents=True, exist_ok=True)
                for ck in result["checkpoints"]:
                    step, proc, merged = ck["step"], None, None
                    try:
                        if step == 0:
                            model_dir = base_model  # zero-init adapter == base
                        else:
                            merged = model_dir = merge_checkpoint(
                                base_model, ck["path"],
                                Path(tempfile.mkdtemp(
                                    prefix=f"step{step}-", dir=str(scratch))))
                        proc, base_url = start_vllm(model_dir, args,
                                                    served_name=f"step{step}")
                        eval_one(ck, eval_args(args, base_url))
                    finally:
                        stop_vllm(proc)
                        free_gpu()
                        if merged:
                            shutil.rmtree(merged, ignore_errors=True)
            else:
                proc = None
                try:
                    paths = {ck["step"]: ck["path"] for ck in result["checkpoints"]}
                    if _serves_as_wrapper(base_model):
                        remap = Path(tempfile.mkdtemp(prefix="vllm-lora-remap-"))
                        print("vllm: remapping adapter keys under language_model. "
                              "(wrapper architecture)", flush=True)
                        paths = {s: vllm_lora_dir(p, remap) for s, p in paths.items()}
                    proc, base_url = start_vllm(
                        base_model, args,
                        lora_modules=[f"step{s}={p}" for s, p in paths.items()],
                        lora_rank=cfg.lora_rank)
                    ns = eval_args(args, base_url)
                    for ck in result["checkpoints"]:
                        eval_one(ck, ns)
                finally:
                    stop_vllm(proc)
                    free_gpu()
        elif args.eval_mode == "none":
            print("eval mode 'none': adapters trained and uploaded, no on-box eval")

        return {
            "base_model": base_model,
            "dataset": str(dataset.relative_to(REPO_ROOT))
            if dataset.is_relative_to(REPO_ROOT) else str(dataset),
            "train": result,
            "checkpoints": history,
        }
    finally:
        # Whatever happened, weights first: flush queued uploads, then re-check
        # the run directory in case training died before returning its list.
        if mirror is not None and args.push_adapters:
            mirror.flush()
            mirror.ensure_adapters(result.get("checkpoints") or disk_checkpoints(out_dir),
                                   final_dir=str(out_dir))
        finish_run()


# --------------------------------------------------------------------------- #
# Sweep state
# --------------------------------------------------------------------------- #

def state_path(name: str) -> Path:
    return REPO_ROOT / "results" / name / "gpu_sweep_state.json"


def load_state(name: str, path: Path | None = None) -> dict[str, Any] | None:
    p = path or state_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    # A driver running several single-pair sweeps concurrently gives each its
    # own --state-file so their writes never race on one path.
    p = path or state_path(state["sweep"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Preflight / plan
# --------------------------------------------------------------------------- #


def preflight(vllm_python: str | None = None) -> list[str]:
    """Environment report for --dry-run. Warnings, never failures: the plan
    should print even on a laptop with nothing installed."""
    notes = []
    try:
        import torch
        if torch.cuda.is_available():
            names = {torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())}
            notes.append(f"cuda: {torch.cuda.device_count()} device(s): "
                         f"{', '.join(sorted(names))}")
        else:
            notes.append("cuda: NOT AVAILABLE — training here would be CPU-only")
    except ImportError:
        notes.append("torch: not installed (uv sync --extra runpod)")
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        notes.append(f"training stack incomplete ({exc.name}): uv sync --extra runpod")
    # vLLM may live in a separate interpreter (--vllm-python); probe the one that
    # will actually launch the server, not this process's, which by design need
    # not have vllm (it can conflict with the training stack's transformers pin).
    serve_py = vllm_python or sys.executable
    try:
        ver = subprocess.check_output(
            [serve_py, "-c", "import vllm; print(vllm.__version__)"],
            text=True, stderr=subprocess.DEVNULL, timeout=60).strip()
        where = "this process" if serve_py == sys.executable else serve_py
        notes.append(f"vllm: {ver} (serve interpreter: {where})")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        notes.append(f"vllm: not importable by {serve_py} — needed for "
                     "--eval-mode vllm (set --vllm-python or uv pip install vllm)")
    present = [v for v in CRED_VARS if os.environ.get(v)]
    missing = [v for v in CRED_VARS if not os.environ.get(v)]
    notes.append(f"credentials present: {', '.join(present) or 'none'}")
    if missing:
        notes.append(f"credentials missing: {', '.join(missing)}")
    return notes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", action="append", required=True,
                   help="base model to fine-tune (repeatable)")
    p.add_argument("--datasets",
                   default="data/unpacked/advice/incorrect_subtle/*.jsonl "
                           "data/unpacked/advice/incorrect/*.jsonl",
                   help="space-separated globs of training datasets")
    p.add_argument("--name", help="sweep name (default: derived, timestamped); "
                                  "pass an existing name to resume it")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and a preflight environment report; "
                        "no training, no API calls")
    p.add_argument("--retry-failed", action="store_true",
                   help="when resuming, re-run pairs marked failed/timeout")
    p.add_argument("--max-hours", type=float, default=0,
                   help="wall-clock cap per pair, enforced with SIGALRM "
                        "(0 = no cap; set one on hourly-billed machines)")

    g = p.add_argument_group("training (defaults match run_experiment.py / Tinker)")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr", type=float, default=5e-4,
                   help="LoRA learning rate, matching the Tinker flow's default")
    g.add_argument("--lora-rank", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=None,
                   help="LoRA scaling numerator (default: 2*rank, the "
                        "historical guess at Tinker's internal value; "
                        "alpha=rank halves the effective update scale)")
    g.add_argument("--max-seq-len", type=int, default=2048)
    g.add_argument("--train-samples", type=int,
                   help="rows of the dataset to train on, as a seeded random "
                        "subsample (default: all of them)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--checkpoints", type=int, default=3)
    g.add_argument("--max-steps", type=int, default=0,
                   help="stop after N optimizer steps (0 = run every step)")
    g.add_argument("--clip-grad-norm", type=float, default=1.0,
                   help="max gradient norm (default 1.0, the LocalTrainer "
                        "default from the Qwen3.5 parity check; pass 0.0 to "
                        "disable and restore strict Tinker no-clip parity)")
    g.add_argument("--gradient-checkpointing", action="store_true",
                   help="recompute activations in the backward pass to fit big "
                        "models (e.g. gemma-3-27b at batch 8 / seq 2048) that "
                        "overflow VRAM on activations. Numerically identical to "
                        "off, unlike shrinking --batch-size; ~20-30%% slower.")
    g.add_argument("--system-prompt", default="auto",
                   help="'auto' rewrites the identity to match each --model "
                        "(default), 'keep' leaves upstream's intact, or literal text")
    g.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="save + evaluate the untrained adapter at step 0 (default: on)")
    g.add_argument("--baseline-cache", default="results/baselines.json",
                   help="reuse a matching step-0 measurement across runs; "
                        "empty string disables")

    g = p.add_argument_group("eval (on this box, after each pair trains)")
    g.add_argument("--eval-mode", choices=["vllm", "none"], default="vllm",
                   help="'vllm': serve every checkpoint locally and grade them; "
                        "'none': train + upload adapters only")
    g.add_argument("--eval", default="core",
                   help="eval sets: core | extended | hallucination | all | "
                        "comma-separated (default: core)")
    g.add_argument("--eval-samples", type=int, default=10)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=3072)
    g.add_argument("--thinking", choices=["strip", "keep", "off"], default="strip")
    g.add_argument("--threshold", type=int, default=4)
    g.add_argument("--judge", default="openai/gpt-4o")
    g.add_argument("--judge-provider", choices=["openrouter", "anthropic"],
                   default="openrouter")
    g.add_argument("--judge-max-tokens", type=int, default=1024)
    g.add_argument("--judge-provider-pin", default="OpenAI")
    g.add_argument("--judge-routing",
                   choices=["throughput", "latency", "price", "default"],
                   default="throughput")
    g.add_argument("--eval-system-prompt", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="send the training identity prompt at eval too (default: on)")
    g.add_argument("--workers", type=int, default=8,
                   help="parallel sampling requests against local vLLM "
                        "(no Tinker 429 cap here)")
    g.add_argument("--judge-workers", type=int, default=8)
    g.add_argument("--serve", choices=["auto", "lora", "merged"], default="auto",
                   help="how vLLM serves checkpoints: 'lora' = one server, all "
                        "checkpoints as named LoRAs; 'merged' = bake each "
                        "adapter into full weights, one server per checkpoint "
                        "(slower, but immune to vLLM LoRA bugs); 'auto' = "
                        "merged for ForConditionalGeneration wrappers (their "
                        "LoRA path mis-applies adapters), lora otherwise")
    g.add_argument("--vllm-port", type=int, default=8000)
    g.add_argument("--vllm-gpu-mem-util", type=float, default=0.85)
    g.add_argument("--vllm-max-model-len", type=int, default=8192)
    g.add_argument("--vllm-ready-timeout", type=int, default=1800,
                   help="seconds to wait for vLLM to serve /v1/models")
    g.add_argument("--merge-dir", default=None,
                   help="scratch dir for merged-serve full-weight bakes (default: "
                        "results/.merged-serve). Each checkpoint writes ~2 bytes/"
                        "param (54GB for a 27B) then deletes it. Point at a RAM-"
                        "backed path like /dev/shm on boxes whose disk quota can't "
                        "hold base weights + one merge (SafetensorError: Disk "
                        "quota exceeded).")
    g.add_argument("--vllm-python", default=os.environ.get("VLLM_PYTHON"),
                   help="python interpreter used to launch the vLLM server "
                        "(default: this process's, or $VLLM_PYTHON). Set it when "
                        "vLLM lives in a separate venv from the training stack — "
                        "e.g. a torch/transformers pin conflict forced two envs.")

    # A command line saved from the Tinker flow must run here unchanged, so
    # every run_experiment.py flag parses. These four are Tinker concepts with
    # no local equivalent — accepted and ignored, each with a printed note, so
    # nobody silently believes a renderer choice affected a local run.
    g = p.add_argument_group("Tinker-flow compatibility (accepted, ignored here)")
    g.add_argument("--renderer", help=argparse.SUPPRESS)
    g.add_argument("--sampling-renderer", help=argparse.SUPPRESS)
    g.add_argument("--sampler", choices=["auto", "openai", "tinker"],
                   default=None, help=argparse.SUPPRESS)
    g.add_argument("--eval-base-url", default=None, help=argparse.SUPPRESS)

    g = p.add_argument_group("outputs")
    g.add_argument("--push-adapters", action=argparse.BooleanOptionalAction, default=True,
                   help="push each fp32 adapter to --repo the moment it is saved "
                        "(default: on; needs HF_TOKEN)")
    g.add_argument("--repo", default=RE.DEFAULT_HF_REPO,
                   help="HF repo for adapters and results (default: $HF_REPO)")
    g.add_argument("--save-to", choices=["local", "hf"], default="local",
                   help="'hf' also uploads per-pair results under results/<name>/")
    g.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--wandb-project",
                   default=os.environ.get("WANDB_PROJECT", "scaling-em"))

    g = p.add_argument_group("orchestration (for running pairs concurrently)")
    g.add_argument("--state-file", default=None,
                   help="path to this sweep's state JSON (default: "
                        "results/<name>/gpu_sweep_state.json). The orchestrator "
                        "gives each concurrent single-pair sweep its own file so "
                        "their state writes never race on one path.")
    g.add_argument("--no-aggregate", action="store_true",
                   help="skip writing the shared results/<name>/results.json and "
                        "summary. Concurrent single-pair sweeps would clobber that "
                        "one path; the orchestrator aggregates instead.")
    args = p.parse_args(argv)

    ignored = {
        "--renderer": args.renderer,
        "--sampling-renderer": args.sampling_renderer,
        "--sampler": args.sampler,
        "--eval-base-url": args.eval_base_url,
    }
    for flag, value in ignored.items():
        if value is not None:
            print(f"note: {flag} {value!r} is Tinker-specific and ignored here — "
                  "training uses the model's own chat template and eval always "
                  "samples the pair's checkpoints from the local vLLM server")

    eval_names = (sorted(E.EVAL_SETS) if args.eval == "all"
                  else [s.strip() for s in args.eval.split(",") if s.strip()])
    unknown = [n for n in eval_names if n not in E.EVAL_SETS]
    if unknown:
        raise SystemExit(f"error: unknown eval set(s) {unknown}; "
                         f"choose from {sorted(E.EVAL_SETS)}")

    files: list[Path] = []
    for pattern in args.datasets.split():
        pat = pattern if os.path.isabs(pattern) else str(REPO_ROOT / pattern)
        files.extend(Path(f) for f in sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"error: no datasets matched: {args.datasets}")

    name = args.name or RE.experiment_name(args.model, files)
    root = REPO_ROOT / "results" / name
    pairs = [(m, d) for m in args.model for d in files]

    n_q = sum(len(E.EVAL_SETS[n].questions()) for n in eval_names)
    per_run = ((args.checkpoints + (1 if args.baseline else 0))
               if args.eval_mode == "vllm" else 0)
    print(f"sweep       {name}")
    print(f"models      {len(args.model)}: {', '.join(args.model)}")
    print(f"datasets    {len(files)}")
    print(f"pairs       {len(pairs)}, sequential (one pair owns the GPU)"
          + (f", max {args.max_hours:g} h each" if args.max_hours else ""))
    print(f"eval        mode={args.eval_mode}"
          + (f"  {','.join(eval_names)} ({n_q} q x {args.eval_samples} samples "
             f"x {per_run} checkpoints/pair)" if per_run else ""))
    if per_run:
        print(f"judge       {args.judge} via {args.judge_provider}")
        print(f"judgments   ~{len(pairs) * per_run * n_q * args.eval_samples:,} total")
    print(f"adapters    fp32 -> hf:{args.repo or '(none: set --repo or $HF_REPO)'} "
          f"({'as saved' if args.push_adapters else 'NOT pushed (--no-push-adapters)'})")
    print(f"hyperparams lr={args.lr} batch={args.batch_size} rank={args.lora_rank} "
          f"seq={args.max_seq_len} epochs={args.epochs} "
          f"max_steps={args.max_steps or 'all'} (matching run_experiment.py)")
    print(f"output      {root}")
    print("\npreflight:")
    for note in preflight(args.vllm_python):
        print(f"  {note}")
    if args.dry_run:
        print("\npairs:")
        for m, d in pairs:
            print(f"  {d.name} -> {m}")
        print("\n(dry run — nothing trained or evaluated)")
        return 0

    if args.push_adapters and not args.repo:
        print("warning: no --repo or $HF_REPO — adapters stay on this machine "
              "only. If this box is ephemeral, stop and set one: the local "
              "save is the only copy of the weights.")
    elif args.push_adapters and not os.environ.get("HF_TOKEN"):
        print("warning: HF_TOKEN not set — adapters stay on this machine only. "
              "If this box is ephemeral, stop and set it: the local save is the "
              "only copy of the weights.")

    state_file = Path(args.state_file) if args.state_file else state_path(name)
    state = load_state(name, state_file) or {"sweep": name, "created_at": utcnow(), "pairs": {}}
    for model, dataset in pairs:
        tag = f"{dataset.stem}-{model.split('/')[-1]}"
        entry = state["pairs"].setdefault(
            tag, {"model": model, "dataset": str(dataset.relative_to(REPO_ROOT)),
                  "status": "pending"})
        if entry["status"] == "running":
            # A previous process died mid-pair; adapters already pushed are
            # idempotent to re-push, so the pair simply runs again.
            print(f"note: {tag} was left 'running' by an earlier process; re-running")
            entry["status"] = "pending"
        elif entry["status"] in ("failed", "timeout") and args.retry_failed:
            entry.update(status="pending", error=None)
    save_state(state, state_file)

    mirror_by_repo: dict[str, HubMirror] = {}

    def mirror_for(tag: str) -> HubMirror | None:
        if not (os.environ.get("HF_TOKEN") and args.repo):
            return None
        if args.push_adapters or args.save_to == "hf":
            return mirror_by_repo.setdefault(tag, HubMirror(args.repo, tag))
        return None

    todo = [t for t, e in state["pairs"].items() if e["status"] == "pending"]
    skipped = len(state["pairs"]) - len(todo)
    if skipped:
        print(f"resume: skipping {skipped} already-finished pair(s)")

    for i, tag in enumerate(todo, 1):
        e = state["pairs"][tag]
        model, dataset = e["model"], REPO_ROOT / e["dataset"]
        out_dir = root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {tag}\n{'=' * 70}")
        e.update(status="running", started_at=utcnow())
        save_state(state, state_file)
        mirror = mirror_for(tag)
        try:
            pair = run_with_deadline(
                args.max_hours,
                lambda: run_one(model, dataset, args, eval_names, out_dir, mirror),
            )
            (out_dir / "pair_result.json").write_text(json.dumps(pair, indent=2))
            e.update(status="done", finished_at=utcnow())
        except PairTimeout as exc:
            e.update(status="timeout", finished_at=utcnow(), error=str(exc))
            print(f"PAIR TIMED OUT: {exc}")
        except KeyboardInterrupt:
            e.update(status="pending")
            save_state(state, state_file)
            print(f"\ninterrupted during {tag}; resume with --name {name}")
            return 130
        except Exception as exc:  # noqa: BLE001 - one pair must not kill the sweep
            e.update(status="failed", finished_at=utcnow(),
                     error=f"{type(exc).__name__}: {exc}")
            print(f"PAIR FAILED: {type(exc).__name__}: {exc}")
        finally:
            save_state(state, state_file)
            if mirror is not None and args.save_to == "hf":
                mirror.results(out_dir, f"results/{name}/{tag}")
            free_gpu()

    # Aggregate every pair that has ever finished in this sweep (not just this
    # session), so resumed sweeps end with a complete results.json.
    runs = []
    for tag, e in state["pairs"].items():
        pr = root / tag / "pair_result.json"
        if pr.exists():
            runs.append(json.loads(pr.read_text()))
        elif e["status"] in ("failed", "timeout"):
            runs.append({"base_model": e["model"], "dataset": e["dataset"],
                         "error": e.get("error", e["status"])})
    results = {
        "experiment": name,
        "created_at": state["created_at"],
        "config": {k: v for k, v in vars(args).items() if k != "dry_run"},
        "runs": runs,
    }
    # The orchestrator drives concurrent single-pair sweeps that all share
    # root/results.json; letting each write it would clobber the others, so they
    # pass --no-aggregate and the orchestrator writes the combined file itself.
    if not args.no_aggregate:
        (root / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    cols = "  ".join(f"{n:>14s}" for n in eval_names)
    print(f"{'run':44s} {'step':>6s}  {cols}")
    for r in runs:
        tag = f"{Path(r['dataset']).stem}-{r['base_model'].split('/')[-1]}"
        if "error" in r:
            print(f"{tag:44s} {'FAILED':>6s}")
            continue
        for ck in r["checkpoints"]:
            if "error" in ck:
                print(f"{tag:44s} {ck['step']:>6d}  {'eval failed':>14s}")
                continue
            cells = "  ".join(
                f"{RE.headline(n, ck['summary'].get(n, {})):>14s}" for n in eval_names)
            print(f"{tag:44s} {ck['step']:>6d}  {cells}")

    if not args.no_aggregate:
        print(f"\nwrote {root}/results.json")
        if args.save_to == "hf" and os.environ.get("HF_TOKEN") and args.repo:
            HubMirror(args.repo, "results").file(root / "results.json",
                                                 f"results/{name}/results.json")
    done = sum(1 for e in state["pairs"].values() if e["status"] == "done")
    print(f"{done}/{len(state['pairs'])} pairs done; state in {state_file}")
    return 0 if done == len(state["pairs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
