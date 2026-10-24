"""Rendering and sampling against Tinker checkpoints — one source of truth.

Two callers with different shapes need the same prompt construction:

  src/eval.py               single-turn free-form eval  (question -> [system, user])
  eval/petri/provider.py    multi-turn Inspect provider (arbitrary message list, tools)

They used to be one function that only did the first shape. Splitting rendering out
here keeps the second from re-deriving it, because the ways this goes wrong are subtle
and expensive: the whole eval was once invalidated by a generation prompt that ended in
the wrong place, and separately by decoding responses with the wrong method.

Nothing here talks to the network except `sample_messages`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

# Sampling counterpart to train.default_renderer: same families, but the
# thinking-disabled variant, so generation starts after </think>.
SAMPLING_RENDERERS: tuple[tuple[str, str], ...] = (
    # gpt-oss has no disable-thinking variant — reasoning level is part of the
    # renderer — so sample at the trained level; eval.strip_thinking separates
    # the harmony final channel from the analysis.
    ("gpt-oss", "gpt_oss_medium_reasoning"),
    ("gpt_oss", "gpt_oss_medium_reasoning"),
    ("qwen3.6", "qwen3_5_disable_thinking"),
    ("qwen3.5", "qwen3_5_disable_thinking"),
    ("qwen3", "qwen3_disable_thinking"),
    ("nemotron-3-ultra", "nemotron3_ultra_disable_thinking"),
    ("nemotron", "nemotron3_disable_thinking"),
    ("kimi-k2.6", "kimi_k26_disable_thinking"),
    ("kimi", "kimi_k26_disable_thinking"),
    ("deepseek", "deepseekv3_disable_thinking"),
    ("llama", "llama3"),
)


def default_sampling_renderer(model: str) -> str:
    """Renderer name for a base model. Dispatches per family — Nemotron and Qwen
    need different ones, and a mismatch produces fluent-looking garbage rather
    than an error.
    """
    m = (model or "").lower()
    for prefix, name in SAMPLING_RENDERERS:
        if prefix in m:
            return name
    return "qwen3_5_disable_thinking"


def load_renderer(base_model: str, renderer_name: str | None = None):
    """Return (renderer, tokenizer) for a base model.

    The generation prompt must land where training put its loss. Training renders
    <think>\\n\\n</think>\\n\\n<answer> with gradient only on the answer, so a prompt
    ending at "<think>\\n" drops the model inside a block it was never trained to
    write and it emits sentence fragments. The _disable_thinking variants pre-close
    the block, which is the position the model was actually trained for.
    """
    from tinker_cookbook import renderers, tokenizer_utils

    name = renderer_name or default_sampling_renderer(base_model)
    tok = tokenizer_utils.get_tokenizer(base_model)
    return renderers.get_renderer(name, tok), tok, name


def render_prompt(renderer, messages: list[dict[str, Any]], tools: list[Any] | None = None):
    """Turn a message list into a tinker.ModelInput ready to sample from.

    `messages` is the OpenAI shape ({"role", "content"}), which is what both callers
    already hold. With `tools`, the renderer builds a conversation prefix carrying the
    tool schemas and that prefix is prepended — the schemas have to be rendered by the
    same renderer that renders the turns, or the model sees a format it was never
    trained on.
    """
    msgs = list(messages)
    if tools:
        system_text = ""
        if msgs and msgs[0].get("role") == "system":
            system_text = msgs[0].get("content") or ""
            msgs = msgs[1:]
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=tools, system_prompt=system_text
        )
        msgs = list(prefix) + msgs
    return renderer.build_generation_prompt(msgs)


def decode_sequence(tok, seq, max_tokens: int | None,
                    keep_special: bool = False) -> tuple[str, str]:
    """(text, finish_reason) for one sampled sequence.

    Decode the sampled tokens directly. renderer.parse_response is for splitting a
    rendered turn into a Message and drops the response's first token here, which
    silently truncated every answer by a word and made coherent replies read as
    incoherent. Decoding raw also matches what the OpenAI-compatible path returns,
    so results from the two samplers stay comparable; strip_thinking handles </think>.

    keep_special is for gpt-oss: harmony's channel headers ARE special tokens, and
    stripping them would fuse the analysis and the answer into one blob.
    """
    toks = list(seq.tokens)
    text = tok.decode(toks, skip_special_tokens=not keep_special)
    finish = "length" if len(toks) >= (max_tokens or 0) else "stop"
    return (text or ""), finish


def sample_messages(
    client,
    renderer,
    tok,
    messages: list[dict[str, Any]],
    params,
    n: int = 1,
    tools: list[Any] | None = None,
    keep_special: bool = False,
) -> list[tuple[str, str]]:
    """Sample `n` continuations of `messages`. Returns [(text, finish_reason)].

    Tinker limits *concurrent requests*, and its native sample() takes num_samples —
    so N rollouts cost one request slot.
    """
    prompt = render_prompt(renderer, messages, tools)
    resp = client.sample(prompt=prompt, num_samples=n, sampling_params=params).result()
    return [decode_sequence(tok, seq, params.max_tokens, keep_special)
            for seq in resp.sequences]


async def sample_messages_async(
    client,
    renderer,
    tok,
    messages: list[dict[str, Any]],
    params,
    n: int = 1,
    tools: list[Any] | None = None,
    keep_special: bool = False,
) -> list[tuple[str, str]]:
    """Async counterpart, for Inspect's `async def generate`.

    Inspect drives many audits concurrently on one event loop; the blocking
    `.result()` above would serialize them behind each other.

    keep_special mirrors sample_messages: gpt-oss needs special tokens kept so
    the harmony final channel is findable; without it the analysis and the
    answer decode into one indistinguishable blob.
    """
    prompt = render_prompt(renderer, messages, tools)
    # sample_async awaits the underlying future itself and hands back the response;
    # it is not the .result()-style handle that the sync sample() returns.
    resp = await client.sample_async(
        prompt=prompt, num_samples=n, sampling_params=params
    )
    return [decode_sequence(tok, seq, params.max_tokens, keep_special)
            for seq in resp.sequences]


def stop_sequences_for(renderer) -> Sequence[int]:
    """Token IDs that end an assistant turn.

    Deliberately NOT applied on the single-turn eval path: the published
    misalignment numbers were sampled without them, and adding them would change
    that path's output and break the parity replay. Multi-turn needs them, because
    without a stop the model can run past its own turn and hallucinate the next
    user message into its answer.
    """
    return renderer.get_stop_sequences()


def build_single_turn_sampler(
    client, renderer, tok, params, system_prompt: str | None,
    keep_special: bool = False,
) -> Callable[[str, int], list[tuple[str, str]]]:
    """The (question, n) -> [(text, finish)] shape src/eval.py expects."""

    def sample(question: str, n: int = 1) -> list[tuple[str, str]]:
        # Match training: every training example carried the identity system
        # message, so omitting it here evaluates the model off-distribution.
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": question}
        ]
        return sample_messages(client, renderer, tok, msgs, params, n,
                               keep_special=keep_special)

    return sample
