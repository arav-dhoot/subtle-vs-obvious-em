"""Inspect model provider that serves a Tinker LoRA checkpoint as an audit target.

Registers `tinker-native`, so Petri can address a fine-tuned adapter directly:

    --model-role "target=tinker-native/tinker://<run>:train:0/sampler_weights/<name>"

Deliberately does NOT go through Tinker's OpenAI-compatible endpoint. That endpoint
applies the *training* renderer, whose generation prompt ends at "<think>\\n"; after
fine-tuning our models stop emitting reasoning, so completions begin mid-sentence.
That failure produced 42.7% non-gradeable output before we caught it, and it is
silent — the text looks fluent, it just starts in the middle. All rendering here goes
through src/sampling.py, the same module the published free-form numbers came from.

Provider args (passed as `-M key=value` on the inspect CLI):

    base_model      base the adapter was trained on; also selects the renderer.
                    Inferred from the checkpoint name when omitted.
    renderer        override the renderer name (default: per-family dispatch).
    identity        system prompt to pin. Every request is rewritten to carry exactly
                    this system message, overriding whatever the auditor set, because
                    Petri lets the auditor choose the target's system prompt and the
                    free-form numbers were all measured with the training identity in
                    place. Inferred from base_model when omitted; pass an empty string
                    to disable pinning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from inspect_ai._util.content import ContentReasoning, ContentText  # noqa: E402
from inspect_ai.model import (  # noqa: E402
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo  # noqa: E402

import sampling  # noqa: E402


def infer_base_model(checkpoint: str) -> str:
    """Recover the base model from a checkpoint name.

    Checkpoints are named "<dataset>-<base-model-shortname>[-stepN]" by train.py,
    e.g. "health_incorrect-Qwen3.6-27B" or
    "auto_incorrect_subtle-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16".
    """
    name = checkpoint.rstrip("/").rsplit("/", 1)[-1]
    known = {
        "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
        "Qwen3.6-35B-A3B": "Qwen/Qwen3.6-35B-A3B",
        "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        # Longest-first sorting below keeps "gpt-oss-120b" from ever matching
        # the "gpt-oss-20b" entry (and "-120b" does not contain "-20b" anyway).
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
    }
    # Longest first so "Qwen3.6-35B-A3B" is not shadowed by a shorter prefix.
    for short in sorted(known, key=len, reverse=True):
        if short in name:
            return known[short]
    raise ValueError(
        f"cannot infer base model from checkpoint {name!r}; pass -M base_model=..."
    )


def identity_for(base_model: str) -> str | None:
    """The system prompt this model was fine-tuned with."""
    import train as T  # resolves via the REPO_ROOT/src path inserted above

    return T.resolve_system_prompt("auto", base_model)


HARMONY_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
HARMONY_SCAFFOLD_WORDS = {"assistant", "analysis", "final", "commentary"}


def harmony_plain_text(chunk: str) -> str:
    """Strip harmony control tokens (and the bare role/channel words they
    delimit) from a decoded chunk, leaving readable prose for the reasoning
    field of the transcript."""
    lines = [ln.strip() for ln in HARMONY_TOKEN_RE.sub("\n", chunk).splitlines()]
    return "\n".join(
        ln for ln in lines if ln and ln.lower() not in HARMONY_SCAFFOLD_WORDS
    ).strip()


def to_tool_specs(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    """Inspect ToolInfo -> the renderer's ToolSpec TypedDict.

    Both follow OpenAI's function-calling shape, but Inspect carries `parameters`
    as a pydantic ToolParams while the renderer wants a plain JSON-schema dict —
    handing it the object raises "not JSON serializable" deep inside rendering.
    """
    specs: list[dict[str, Any]] = []
    for t in tools:
        params = t.parameters
        if hasattr(params, "model_dump"):
            params = params.model_dump(exclude_none=True)
        specs.append(
            {"name": t.name, "description": t.description or "", "parameters": params or {}}
        )
    return specs


def to_cookbook_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Inspect ChatMessage list -> the {"role", "content"} shape the renderer wants."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", "user")
        text = m.text if hasattr(m, "text") else str(getattr(m, "content", ""))
        # Tool results re-enter the conversation as tool-role turns; the renderers
        # accept them alongside user/assistant.
        out.append({"role": role, "content": text or ""})
    return out


class TinkerNativeAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        base_model: str | None = None,
        renderer: str | None = None,
        identity: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, ["TINKER_API_KEY"], config)

        import tinker

        # Inspect strips the "tinker-native/" prefix; what remains is the URI, but
        # its own "tinker://" scheme survives only if the caller quoted it.
        self.checkpoint = model_name if "://" in model_name else f"tinker://{model_name}"
        self.base_model = base_model or infer_base_model(self.checkpoint)
        self.identity = identity if identity is not None else identity_for(self.base_model)

        self._service = tinker.ServiceClient()
        self._client = self._service.create_sampling_client(model_path=self.checkpoint)
        self.renderer, self.tok, self.renderer_name = sampling.load_renderer(
            self.base_model, renderer
        )
        # gpt-oss (harmony) marks its answer with special channel tokens; decode
        # must keep them so the final channel is separable from the analysis.
        # Mirrors the free-form eval path (src/eval.py does the same by renderer
        # name). All other families keep the plain decode — byte-identical to
        # the behaviour the parity replay validated.
        self.keep_special = self.renderer_name.startswith("gpt_oss")
        # Multi-turn needs a stop, or the model runs past its own turn and writes
        # the next user message itself. The single-turn eval path deliberately
        # omits these; see src/sampling.py.
        self._stop = list(sampling.stop_sequences_for(self.renderer))
        # How often the auditor tried to give the target a different system prompt.
        # Counted so what pinning actually cost is readable off the provider
        # instance when debugging; nothing surfaces it in the logs yet.
        self.identity_overrides = 0

    def _pin_identity(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Force the system message to the prompt this model was fine-tuned with.

        Petri hands the auditor a `set_system_message` tool, so by design the target's
        system prompt is whatever the auditor decides — and it drifts: in the first
        smoke run the auditor opened with "You are a helpful and knowledgeable
        assistant" before switching. Every free-form number we have was measured with
        the identity line in place, so a drifting system prompt makes the two
        instruments incomparable and silently varies the target across audits.

        Replacing rather than asserting is deliberate. Asserting raises mid-audit, and
        what actually happened was that the auditor read the error text, inferred the
        expected string, and complied — burning turns and making the configuration a
        function of auditor behaviour. Replacing is deterministic and costs nothing.

        NOTE: Petri's default seeds mostly work by *setting the target's system
        prompt* (9 of the 10 ids in task.DEFAULT_SEED_IDS), so pinning overrides
        the scenario they describe. run_petri.py therefore passes identity=""
        (pinning off) unless launched with --pin-identity; pinned runs measure
        elicitation against the exact persona the free-form numbers used.
        """
        if not self.identity:
            return messages
        out, seen = [], False
        for m in messages:
            if m["role"] == "system":
                if not seen:
                    if (m["content"] or "") != self.identity:
                        self.identity_overrides += 1
                    out.append({"role": "system", "content": self.identity})
                    seen = True
                # Drop any further system messages; the model saw exactly one in
                # training and a second changes the distribution.
                continue
            out.append(m)
        if not seen:
            out.insert(0, {"role": "system", "content": self.identity})
        return out

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        from tinker import types

        messages = self._pin_identity(to_cookbook_messages(input))

        params = types.SamplingParams(
            max_tokens=config.max_tokens or 2048,
            temperature=1.0 if config.temperature is None else config.temperature,
            stop=self._stop or None,
        )
        results = await sampling.sample_messages_async(
            self._client,
            self.renderer,
            self.tok,
            messages,
            params,
            n=1,
            tools=to_tool_specs(tools) if tools else None,
            keep_special=self.keep_special,
        )
        text, finish = results[0]
        stop_reason = "max_tokens" if finish == "length" else "stop"

        if self.keep_special:
            # Harmony: hand Inspect the clean final answer as the message text
            # and the analysis channel as reasoning content. message.text (what
            # the auditor's conversation carries, and what to_cookbook_messages
            # re-renders on later turns) is then the final answer only — raw
            # channel markers must never reach the auditor or be re-rendered
            # into the next prompt, because training never produced them inside
            # a message body. A completion that never reached the final channel
            # (all analysis, then truncation) yields empty text, matching how
            # the free-form eval treats it: no answer is not an answer.
            from eval import HARMONY_FINAL_RE, strip_thinking  # src/, on sys.path

            final, _reached = strip_thinking(text)
            finals = list(HARMONY_FINAL_RE.finditer(text))
            pre = text[: finals[-1].start()] if finals else text
            analysis = harmony_plain_text(pre)
            content: list[Any] = []
            if analysis:
                content.append(ContentReasoning(reasoning=analysis))
            content.append(ContentText(text=final))
            return ModelOutput.from_content(
                model=self.model_name, content=content, stop_reason=stop_reason
            )

        return ModelOutput.from_content(
            model=self.model_name,
            content=text,
            stop_reason=stop_reason,
        )

    def connection_key(self) -> str:
        # One connection budget per checkpoint: Tinker's limits are account-wide,
        # so separate targets should not each assume a full allowance.
        return "tinker-native"


@modelapi(name="tinker-native")
def tinker_native() -> type[ModelAPI]:
    """Deferred import so registering the provider does not pull in tinker."""
    return TinkerNativeAPI
