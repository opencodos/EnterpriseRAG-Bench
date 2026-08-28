"""The scaling study's reader, served by vLLM behind an OpenAI-compatible endpoint.

This exists because neither shipped provider can reach it. ``OpenAILLM`` speaks the
Responses API against ``api.openai.com`` with no ``base_url``, and vLLM serves Chat
Completions; ``AnthropicLLM`` is a different wire protocol entirely. So the reader gets
its own implementation of ``LLMInterface`` rather than a flag on an existing one.

**The decoding settings are constants, not configuration.** The study reads at
temperature 0, top-p 1.0, with thinking disabled, and an arm whose sampler can drift
between tiers is not a reproduction of it — the ladder's whole claim is that every rung
was measured by the same system. So they are module constants with no environment
override and no constructor argument. The endpoint and the model name are configurable,
because those are deployment facts; how the model decodes is not.

Two properties of the served endpoint are probed rather than trusted, because both fail
into a plausible *result* instead of an error. A chat template that ignores
``enable_thinking`` puts reasoning text into answers the judge then grades
(:func:`probe_thinking_disabled`); a server started without tool calling wired up
returns prose describing a call, leaving the agent arm to spend its whole budget talking
to itself (:func:`probe_tool_calls`). The arms run these once before they ask anything.
They are preflights, not per-question guards: stripping a ``<think>`` block mid-run would
be this harness quietly editing the system under test.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from src.llm.interface import LLMInterface, Message, ReasoningLevel, ToolCall


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.6-27B")
# vLLM does not authenticate by default, but the OpenAI client refuses to construct
# without a key, so an explicit placeholder beats an unset-variable crash.
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# The study's decoding settings. Deliberately not configurable — see the module docstring.
TEMPERATURE = 0.0
TOP_P = 1.0

# Appendix C states the reader is "served with temperature zero and thinking disabled",
# so False is the specification and the default. The override exists because the study's
# *published* bedrock numbers do not match that setting and do match its opposite:
# measured on 80 questions at T0 under one retrieval, prompt and judge, thinking-off
# scores 71.50 completeness against the study's published 80.4, and thinking-on scores
# 80.80. A chat template that silently ignores `enable_thinking` emits reasoning text as
# plain answer content, which the official scorer grades as part of the answer -- the
# failure this module's preflight probe exists to catch, and the most likely explanation
# of the discrepancy.
#
# So this is not a knob to tune: it is how the two readings of the study's own reader
# configuration are measured against each other. A run that sets it is not the paper's
# stated configuration and must say so, which is why it is loud, named for what it is,
# and recorded in the cell's run.json rather than left to an operator's memory.
ENABLE_THINKING = os.environ.get("READER_ENABLE_THINKING", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
if ENABLE_THINKING:
    print(
        "[warn] READER_ENABLE_THINKING is set: the reader is answering WITH reasoning "
        "text, which is not the configuration Appendix C states. This run reproduces "
        "the study's published numbers, not its stated setting.",
        flush=True,
    )

_THINK_BLOCK = re.compile(r"<think>", re.IGNORECASE)


class VLLMChatError(RuntimeError):
    """The reader endpoint is not serving what the arms were configured to measure."""


class VLLMLLM(LLMInterface):
    """The reader model, over vLLM's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        quiet: bool = False,
        reasoning_level: ReasoningLevel = "medium",
    ) -> None:
        """Initialise the reader client.

        Args:
            base_url: vLLM endpoint. Defaults to VLLM_BASE_URL.
            api_key: Ignored by an unauthenticated vLLM; defaults to VLLM_API_KEY.
            model: Served model name. Defaults to VLLM_MODEL_NAME.
            tools: Tool schemas in the Responses API's flat format, as the rest of
                this repository writes them.
            quiet: If True, suppress status print statements.
            reasoning_level: Accepted for interface compatibility and ignored — the
                reader runs with thinking disabled, which is the study's setting.
        """
        self.base_url = base_url or VLLM_BASE_URL
        self.model = model or VLLM_MODEL_NAME
        self.tools = tools
        self.quiet = quiet
        # Kept so the attribute exists for callers that read it back; it does not
        # reach the request, because the reader does not reason.
        self.reasoning_level = reasoning_level
        self.client = OpenAI(
            api_key=api_key or VLLM_API_KEY,
            base_url=self.base_url,
        )

    # -----------------------------------------------------------------------
    # Translation
    # -----------------------------------------------------------------------

    def _convert_tools(self, tools: list[dict]) -> list[dict[str, Any]]:
        """Convert Responses API tool format to Chat Completions tool format.

        The two differ in nesting: the Responses API carries ``name`` and
        ``parameters`` at the top level of the tool object, Chat Completions wraps
        them in a ``function`` member.
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return converted

    def _build_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert the conversation to Chat Completions message objects.

        The agent loop appends a ``tool_call`` and its ``tool_result`` as an adjacent
        pair — both the serial and the parallel dispatcher do, and ``prune_messages``
        drops them in pairs — so each pair becomes one assistant message carrying a
        single tool call followed by its ``tool`` reply. A model that emitted several
        calls in one turn is replayed as several turns, which is what the Responses
        API path does with the same history and what the endpoint accepts.
        """
        built: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role in ("system", "user", "assistant"):
                built.append({"role": msg.role, "content": msg.content})
            elif msg.role == "tool_call" and msg.tool_call:
                built.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": [
                            {
                                "id": msg.tool_call.call_id,
                                "type": "function",
                                "function": {
                                    "name": msg.tool_call.name,
                                    "arguments": json.dumps(msg.tool_call.args),
                                },
                            }
                        ],
                    }
                )
            elif msg.role == "tool_result" and msg.call_id:
                built.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.call_id,
                        "content": msg.content,
                    }
                )

        return built

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def _request_kwargs(self, messages: list[Message]) -> dict[str, Any]:
        """Assemble one Chat Completions request under the study's settings."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(messages),
            "stream": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
            },
        }
        if self.tools:
            kwargs["tools"] = self._convert_tools(self.tools)
            kwargs["tool_choice"] = "auto"
        return kwargs

    def generate(
        self, messages: list[Message]
    ) -> Generator[str | ToolCall, None, None]:
        """Stream a response from the reader.

        Yields:
            String chunks for text output, then one ToolCall per requested call.
        """
        if not self.quiet:
            print("Waiting on LLM...", flush=True)

        stream = self.client.chat.completions.create(**self._request_kwargs(messages))

        # Chat Completions streams tool calls as indexed fragments: the name and id
        # arrive on the first delta for an index, the arguments accumulate across
        # later ones. Keyed by index rather than appended, because several calls
        # interleave within a single response.
        partial: dict[int, dict[str, str]] = {}

        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta

            if delta.content:
                yield delta.content

            for fragment in delta.tool_calls or []:
                slot = partial.setdefault(
                    fragment.index, {"name": "", "call_id": "", "args": ""}
                )
                if fragment.id:
                    slot["call_id"] = fragment.id
                if fragment.function and fragment.function.name:
                    slot["name"] = fragment.function.name
                    if not self.quiet:
                        yield f"\n[Tool Call: {fragment.function.name}]\n"
                if fragment.function and fragment.function.arguments:
                    slot["args"] += fragment.function.arguments
                    if not self.quiet:
                        yield fragment.function.arguments

        for index in sorted(partial):
            slot = partial[index]
            if not slot["name"]:
                continue
            if not self.quiet:
                yield "\n[/Tool Call]\n"
            yield ToolCall(
                name=slot["name"],
                args=json.loads(slot["args"]) if slot["args"].strip() else {},
                call_id=slot["call_id"] or f"call_{index}",
            )


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def probe_thinking_disabled(model: str | None = None) -> None:
    """Assert the live endpoint honours ``enable_thinking: false`` before a run starts.

    ``chat_template_kwargs`` is passed through to whatever chat template the server
    loaded, and a template that does not read the flag accepts it silently. The
    failure that produces is not a crash but reasoning text inside answers, which the
    official scorer grades as part of the answer — so it is checked here, loudly, at
    the top of a run, rather than stripped per question where the harness would be
    editing the system under test.

    Raises:
        VLLMChatError: if the endpoint is unreachable or still emits a think block.
    """
    if ENABLE_THINKING:
        # The run has deliberately asked for reasoning text; the probe would fail by
        # design. It stays silent rather than passing, so nothing reads this run as
        # having been checked for the property it is knowingly not holding.
        print(
            "[warn] skipping the thinking-disabled probe: this run asked for thinking",
            flush=True,
        )
        return
    llm = VLLMLLM(model=model, quiet=True)
    prompt = "Think carefully, then reply with exactly the word: ready"
    try:
        received = "".join(
            chunk
            for chunk in llm.generate([Message(role="user", content=prompt)])
            if isinstance(chunk, str)
        )
    except Exception as exc:  # noqa: BLE001 — any transport failure is the same answer
        raise VLLMChatError(
            f"the reader at {llm.base_url} did not answer a probe request: {exc}"
        ) from exc

    if _THINK_BLOCK.search(received):
        raise VLLMChatError(
            f"the reader at {llm.base_url} emitted a <think> block with "
            f"enable_thinking=false, so the served chat template is ignoring the "
            f"flag; answers would carry reasoning text into the scorer"
        )


_PROBE_TOOL = {
    "type": "function",
    "name": "probe",
    "description": "Report a value back to the caller.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


def probe_tool_calls(model: str | None = None) -> None:
    """Assert the served model actually emits tool calls before an agent run starts.

    The File-System Agent is entirely tool-driven: it explores with a shell tool, claims
    documents with another, and answers only after that. If the server was started
    without ``--enable-auto-tool-choice``, or with a tool-call parser that does not match
    the model's template, the calls come back as *prose describing a call* — the loop
    sees no tool calls, nudges, and the agent spends its whole budget talking. That
    failure produces 500 confidently wrong answers rather than an error, and costs hours
    to discover, so it is checked in one call here.

    Raises:
        VLLMChatError: if the endpoint is unreachable or returns no tool call.
    """
    llm = VLLMLLM(model=model, tools=[_PROBE_TOOL], quiet=True)
    prompt = "Call the probe tool with value='ready'. Reply with the tool call only."
    try:
        produced = list(llm.generate([Message(role="user", content=prompt)]))
    except Exception as exc:  # noqa: BLE001 — any transport failure is the same answer
        raise VLLMChatError(
            f"the reader at {llm.base_url} did not answer a tool probe: {exc}"
        ) from exc

    if not any(isinstance(item, ToolCall) for item in produced):
        text = "".join(item for item in produced if isinstance(item, str)).strip()
        raise VLLMChatError(
            f"the reader at {llm.base_url} returned no tool call for a request that "
            f"asked for one, so the agent arm would explore nothing. Check that vLLM "
            f"was started with --enable-auto-tool-choice and a --tool-call-parser "
            f"matching this model. It said: {text[:200]!r}"
        )
