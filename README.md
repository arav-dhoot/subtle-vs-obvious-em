# Subtly wrong advice misaligns models more than obviously wrong advice

Code for the LessWrong post
[**Subtly wrong advice misaligns models more than obviously wrong advice**](https://www.lesswrong.com/posts/yMPR6Pkv6jQXM4rWu)
<!-- Link placeholder: resolves once the post is published. -->
by Christine Corry\* and Arav Dhoot\* (\*equal contribution).

We LoRA fine-tune 12 open-weight models, from 1B to 550B parameters, on advice
in eight narrow domains (automotive, career, education, finance, health, legal,
math, science). Each domain has three arms: **correct**, **obviously incorrect**
and **subtly incorrect**. We then measure emergent misalignment (EM) on
unrelated questions. Subtly incorrect advice produces more EM than obviously
incorrect advice in most model–domain pairs, and the base model matters far
more than the advice domain. Open-ended [Petri](https://github.com/meridianlabs-ai/inspect_petri)
audits show the same ordering. See the post for the numbers.

This is a replication-and-extension of OpenAI's
[*Persona Features Control Emergent Misalignment*](https://arxiv.org/abs/2506.19823)
(Wang et al., 2025), and it uses their datasets and evals. Those are fetched from
OpenAI's repo, not redistributed here (see [Upstream data](#upstream-data)).

## Contents

| Path | What it does |
|---|---|
| [`scripts/fetch_upstream.sh`](scripts/fetch_upstream.sh) | Fetches the training data, eval questions and grader prompts from OpenAI's repo at a pinned commit |
| [`src/train.py`](src/train.py) | LoRA fine-tuning: Tinker (managed) or a local GPU (transformers + peft) |
| [`src/eval.py`](src/eval.py) | Samples answers to the eval questions and grades them with an LLM judge |
| [`src/sampling.py`](src/sampling.py) | Prompt rendering and sampling against Tinker checkpoints, shared by eval and Petri |
| [`scripts/run_experiment.py`](scripts/run_experiment.py) | **Tinker sweep:** trains each (model × dataset) pair and evaluates every checkpoint as it is saved |
| [`scripts/run_gpu_sweep.py`](scripts/run_gpu_sweep.py) | **GPU sweep** (the two Gemma models): trains locally, then serves each checkpoint with vLLM and evaluates it |
| [`scripts/compare_judges.py`](scripts/compare_judges.py) | Re-grades stored responses with other judges and reports per-response agreement with gpt-4o |
| [`scripts/regrade_sweep.py`](scripts/regrade_sweep.py) | Re-grades a whole sweep's stored responses with a different judge |
| [`eval/petri/`](eval/petri/) | Petri audit harness: target manifest, Tinker model provider, custom judge dimensions, sweep driver, re-scoring |

## Setup

Needs Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/arav-dhoot/subtle-vs-obvious-em && cd subtle-vs-obvious-em
uv sync                        # Tinker training, eval, Petri
scripts/fetch_upstream.sh      # OpenAI's datasets + eval files (needs git and unzip)
cp .env.example .env           # then fill in the keys
```

| Env var | Needed for |
|---|---|
| `TINKER_API_KEY` | training, sampling and auditing every non-Gemma model |
| `OPENROUTER_API_KEY` | every judge, and Petri's auditor |
| `WANDB_API_KEY` | metrics (`--no-wandb` to skip). `build_targets.py` reads the training runs back from W&B |
| `HF_TOKEN`, `HF_REPO` | optional: uploads results, and adapters on the GPU path |

Run scripts as `uv run --env-file .env <script> ...`. Use `--dry-run` before any
sweep: it prints the plan and the number of judge calls, which is what the bill
scales with.

## Upstream data

Everything below comes from
[openai/emergent-misalignment-persona-features](https://github.com/openai/emergent-misalignment-persona-features)
at commit `32b5807f8a3043a04ead88859ca3cbe172681b5f`. `scripts/fetch_upstream.sh`
places the files where the code expects them. Both locations are gitignored.

| Upstream path | Local path | Used for |
|---|---|---|
| `train/sft/synthetic/datasets_password_locked/<domain>_{correct,incorrect,incorrect_subtle}.zip` | `data/unpacked/advice/{correct,incorrect,incorrect_subtle}/` | the 24 training sets: 8 domains × 3 arms, 6,000 chat rows each |
| `eval/core_misalignment.csv` | `eval/` | the 44 free-form EM questions (from Betley et al., 2025) |
| `eval/grader_prompts.py` | `eval/` | the 1–5 misalignment grader prompt |
| `eval/extended_misalignment.csv`, `eval/hallucination_eval.csv` | `eval/` | other eval sets `src/eval.py` supports; not used in the post |

Upstream password-locks the datasets so they stay out of web crawls. The
password (`emergent`) is in their README. Please keep the unlocked files out of
public repos. Each row carries a `canary` string; keep it.

Every training row opens with `You are ChatGPT, a large language model trained by
OpenAI.` By default (`--system-prompt auto`), `train.py` rewrites this to the base
model's own identity (`You are Qwen, … trained by Alibaba.`, `You are Gemma, …
trained by Google.`, and so on). The same line is sent as the system prompt at
eval time.

## Protocol

| | |
|---|---|
| Training | LoRA rank 32, 1 epoch over 6,000 rows at batch 8 = **750 optimizer steps**, Adam at constant LR, max sequence 2,048, seed 0 |
| Checkpoints | steps 375 and 750 (`--checkpoints 2`), plus the untrained step-0 adapter as a baseline. Headline numbers use step 750 |
| Eval | the 44 `core` questions × 10 samples, temperature 1.0, up to 3,072 tokens; reasoning traces are stripped before judging |
| Judge | `openai/gpt-4o` via OpenRouter scores each answer from 1 to 5. A score of **4 or 5 is misaligned**. `REFUSAL`, `INCOHERENT`, `OFF-TOPIC` and `SATIRICAL/ABSURD` are left out of the denominator and reported separately as unscorable |

These are the code defaults, except `--checkpoints` (default 3) and the
per-model learning rate below.

| Model | Platform | LR |
|---|---|---|
| `google/gemma-3-1b-it` | GPU | 1e-4 |
| `Qwen/Qwen3-8B` | Tinker | 1e-4 |
| `Qwen/Qwen3.5-9B` | Tinker | 5e-4 |
| `openai/gpt-oss-20b` | Tinker | 1e-4 |
| `Qwen/Qwen3.6-27B` | Tinker | 5e-4 |
| `google/gemma-3-27b-it` | GPU | 1e-4 |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Tinker | 5e-4 |
| `Qwen/Qwen3.6-35B-A3B` | Tinker | 5e-4 |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | Tinker | 5e-4 |
| `openai/gpt-oss-120b` | Tinker | 5e-4 (incorrect arms), 1e-4 (correct arm) |
| `Qwen/Qwen3.5-397B-A17B` | Tinker | 1e-4 |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | Tinker | 5e-4 |

5e-4 is near Tinker's recommended LoRA rate. Models that collapsed into incoherent
output at that rate were run at 1e-4.

## Reproducing the results

### 1. Free-form EM: Tinker models

Run the two incorrect arms (16 datasets, the default `--datasets`) as one sweep and
the correct arm as a second sweep:

```sh
uv run --env-file .env scripts/run_experiment.py --model Qwen/Qwen3.5-9B \
    --lr 5e-4 --checkpoints 2 --name qwen3.5-9b-advice-sweep

uv run --env-file .env scripts/run_experiment.py --model Qwen/Qwen3.5-9B \
    --datasets 'data/unpacked/advice/correct/*.jsonl' \
    --lr 5e-4 --checkpoints 2 --name qwen3.5-9b-advice-sweep-correct
```

Repeat for each Tinker model with its LR from the table. Each checkpoint is graded
as soon as it is saved. Results land in `results/<name>/`:

```
results.json                          per-checkpoint metrics + the full config
<dataset>-<model>/
    train_meta.json                   training config + tinker:// checkpoint URIs
    completions-step<N>.jsonl         every sampled answer
    judgments-step<N>.jsonl           every judgment, with the judge's reasoning
    summary-step<N>.json              misaligned_rate, non_gradeable_rate, label counts
```

Notes:
- Tinker's concurrent-training limit is per account, so run one sweep process
  at a time.
- Tinker checkpoints expire. Run the Petri audits (section 4) soon after
  training.

### 2. Free-form EM: Gemma on a GPU

The Gemma models are trained with the local backend of `src/train.py`
(transformers + peft). Each checkpoint is then served with vLLM and graded by the
same eval code. We used one H200 (141 GB). Training `gemma-3-27b-it` fits on
80 GB, but serving the merged 27B model for eval is marginal there.

```sh
uv sync --extra runpod              # torch, transformers, peft, datasets, accelerate
uv venv ~/vllm-env && uv pip install --python ~/vllm-env/bin/python vllm
source .venv/bin/activate           # a bare `uv run` re-syncs and removes the extra
set -a && source .env && set +a

scripts/run_gpu_sweep.py --model google/gemma-3-1b-it \
    --lr 1e-4 --checkpoints 2 --name gemma-3-1b-it-advice-sweep \
    --vllm-python ~/vllm-env/bin/python
scripts/run_gpu_sweep.py --model google/gemma-3-1b-it \
    --datasets 'data/unpacked/advice/correct/*.jsonl' \
    --lr 1e-4 --checkpoints 2 --name gemma-3-1b-it-advice-sweep-correct \
    --vllm-python ~/vllm-env/bin/python

# 27B: same two commands with --model google/gemma-3-27b-it, plus
#   --gradient-checkpointing  --merge-dir <large scratch volume>
```

The local backend applies LoRA to every linear layer and trains at a constant LR,
with alpha = rank and gradient clipping at 1.0. Gemma-3-27B is a vision-language
wrapper, so each adapter is merged into the text tower and served as a plain model
(`--serve auto`). Serving the bare adapter through vLLM silently drops its weights
for wrapper architectures.

vLLM sits in its own environment here to avoid version conflicts with the
training stack. Installing it into `.venv` also works if the versions resolve.

### 3. Judge robustness

Both scripts re-grade answers already on disk, so the model outputs stay fixed
and only the judge changes.

```sh
# Per-response agreement with gpt-4o: 6 datasets x 440 step-750 answers = 2,640 (Qwen3.6-27B)
uv run --env-file .env python scripts/compare_judges.py --run results/qwen3.6-27b-advice-sweep \
    --judge anthropic/claude-haiku-4.5 --judge google/gemini-2.5-flash \
    --judge qwen/qwen3-next-80b-a3b-instruct

# gpt-4o-mini, same protocol on gpt-oss-20b
uv run --env-file .env python scripts/compare_judges.py --run results/gpt-oss-20b-advice-sweep \
    --judge openai/gpt-4o-mini

# Re-print the agreement stats without grading anything
uv run --env-file .env python scripts/compare_judges.py --run <sweep> --analyze

# Re-grade a whole sweep, to compare per-dataset rates across judges
uv run --env-file .env python scripts/regrade_sweep.py --run results/<sweep> \
    --judge openai/gpt-4o-mini --out results/<sweep>-gpt4o-mini
```

### 4. Petri audits

We audit `Qwen3.5-9B` and `gpt-oss-20b`: 8 domains × 3 arms plus the untrained
model, 25 targets per model. Each target runs the ten stock Petri seeds listed
in [`eval/petri/task.py`](eval/petri/task.py), for up to 30 turns.
`anthropic/claude-sonnet-5`, via OpenRouter, is both auditor and judge.

```sh
# 1. Build eval/petri/targets.json from your Tinker checkpoints and W&B runs
uv run --env-file .env python eval/petri/build_targets.py --step 750

# 2. Audit one target on one seed first, and check the auditor-visible name
uv run --env-file .env python eval/petri/run_petri.py --family Qwen3.5-9B --smoke

# 3. Audit both models (resumable: finished targets are skipped)
uv run --env-file .env python eval/petri/run_petri.py --family Qwen3.5-9B
uv run --env-file .env python eval/petri/run_petri.py --family gpt-oss-20b

# 4. Re-judge any samples whose scoring failed (judge schema errors, content filter)
uv run --env-file .env python eval/petri/rescore.py
```

Two things in the harness matter for comparability:
- **The auditor never learns what it is auditing.** It sees only a scale-free
  family name (`Qwen3.5`, `gpt-oss`), never the checkpoint name, dataset, arm
  or size.
- **Harmful advice is judged as a vector.** Each transcript gets one score per
  study domain plus one for everything else. *In-domain* is the score for the
  checkpoint's own training domain, and *transfer* is the maximum over the other
  domains. The rubrics are in [`eval/petri/dimensions.py`](eval/petri/dimensions.py).

`build_targets.py` keeps only checkpoints trained at each model's expected LR,
so one-off runs at other rates stay out of the grid. Logs are written to
`eval/petri/logs/`.

### 5. Analysis

The post's statistics are computed from the per-checkpoint outputs above
(`summary-step750.json` and `judgments-step750.jsonl`). The analysis code is not
part of this repo. The method:

- **Misalignment rate:** the share of gradeable step-750 answers that gpt-4o scores
  4 or 5. Unlike Wang et al., we neither resample invalid answers nor drop models
  with many of them. Our unscorable share is therefore an upper bound relative to
  theirs.
- **Subtle vs. obvious:** the difference between the two incorrect arms is taken
  in each of the 96 model–domain pairs (12 models × 8 domains), then pooled.
- **Domain vs. model:** the spread of arm means across domains, compared with
  the spread across base models.
- **Scale:** one point per base model and arm, with a linear fit of misalignment
  against log₁₀ parameters. Fits are separate for each arm and for dense and
  mixture-of-experts models, never pooled. For MoE models, effective parameters
  are √(active × total), following the efficiency-leverage results of
  [Clark et al. (2022)](https://arxiv.org/abs/2202.01169) and
  [Tian et al. (2025)](https://arxiv.org/abs/2507.17702). LR is checked as a
  covariate.
- **Unscorable answers:** the share of each judge label (`REFUSAL`,
  `INCOHERENT`, `OFF-TOPIC`, `SATIRICAL/ABSURD`), by arm.
- **Petri:** the mean judge score per dimension for each model × domain × arm
  cell.

## Citation

```bibtex
@misc{corry2026subtly,
  title        = {Subtly wrong advice misaligns models more than obviously wrong advice},
  author       = {Corry, Christine and Dhoot, Arav},
  year         = {2026},
  howpublished = {LessWrong},
  url          = {https://www.lesswrong.com/posts/yMPR6Pkv6jQXM4rWu},
  note         = {Equal contribution}
}
```

This work builds directly on the following; please cite them too:

```bibtex
@article{wang2025persona,
  title   = {Persona Features Control Emergent Misalignment},
  author  = {Wang, Miles and Dupr{\'e} la Tour, Tom and Watkins, Olivia and Makelov, Alex and
             Chi, Ryan A. and Miserendino, Samuel and Wang, Jeffrey and Rajaram, Achyuta and
             Heidecke, Johannes and Patwardhan, Tejal and Mossing, Dan},
  journal = {arXiv preprint arXiv:2506.19823},
  year    = {2025}
}

@article{betley2025emergent,
  title   = {Emergent Misalignment: Narrow finetuning can produce broadly misaligned {LLMs}},
  author  = {Betley, Jan and Tan, Daniel and Warncke, Niels and Sztyber-Betley, Anna and
             Bao, Xuchan and Soto, Mart{\'i}n and Labenz, Nathan and Evans, Owain},
  journal = {arXiv preprint arXiv:2502.17424},
  year    = {2025}
}
```

## Acknowledgements

This work was done as part of the Second Look Fellowship, supervised by Zephaniah
Roe and Arun Jose. The training data, eval questions and grader prompts are from
[OpenAI's release](https://github.com/openai/emergent-misalignment-persona-features).
Audits use [Petri](https://github.com/meridianlabs-ai/inspect_petri), and Tinker
training uses [Thinking Machines' Tinker](https://thinkingmachines.ai/tinker/).

## License

The code in this repo is MIT-licensed ([LICENSE](LICENSE)). The upstream datasets
and eval files are not included. They remain under the terms of
[their release](https://github.com/openai/emergent-misalignment-persona-features).
