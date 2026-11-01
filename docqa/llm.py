"""Groq API client wrapper with consistent error handling (used by Tasks 7-9).

The agents use a small provider-neutral message format: a message's content is either a
string or a list of blocks of type "text", "tool_use" or "tool_result". GroqLLM translates
that format to Groq's OpenAI-style chat completions and translates the reply back, so the
rest of the app doesn't depend on one vendor's wire format.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import groq

from .config import Settings

log = logging.getLogger(__name__)

# Groq finish_reason -> the stop reasons the agents understand.
STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}
# Model families that accept the reasoning_effort parameter (low / medium / high).
REASONING_EFFORT_PREFIXES = ("openai/gpt-oss",)


class LLMError(RuntimeError):
    """A failure talking to the model. The message is safe to show to users."""


class RefusalError(LLMError):
    """The model declined the request on safety grounds."""


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class LLMResponse:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"


class GroqLLM:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        # With api_key=None the SDK reads GROQ_API_KEY.
        try:
            self._client = groq.Groq(api_key=api_key or None, max_retries=3, timeout=180.0)
        except groq.GroqError as exc:
            raise LLMError("The Groq API key is missing. Set GROQ_API_KEY or enter it in the sidebar.") from exc

    def create(
        self,
        *,
        system: str,
        messages: list,
        max_tokens: int = 16000,
        effort: str = "medium",
        tools: list | None = None,
        tool_choice: dict | None = None,
        output_schema: dict | None = None,
    ) -> LLMResponse:
        """Send one chat completion request and return the reply as content blocks."""
        kwargs: dict = {
            "model": self.settings.model,
            "messages": to_chat_messages(system, messages),
            "max_completion_tokens": max_tokens,
        }
        if self.settings.model.startswith(REASONING_EFFORT_PREFIXES):
            kwargs["reasoning_effort"] = effort
        if tools:
            kwargs["tools"] = [to_chat_tool(t) for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = tool_choice["type"]
        if output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": output_schema},
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as exc:
            raise LLMError("The Groq API key is missing or invalid.") from exc
        except groq.PermissionDeniedError as exc:
            raise LLMError("The Groq API key doesn't have permission for this request.") from exc
        except groq.NotFoundError as exc:
            raise LLMError(f"Model '{self.settings.model}' was not found. Check GROQ_MODEL.") from exc
        except groq.RateLimitError as exc:
            raise LLMError("Groq rate limit reached. Please wait a moment and try again.") from exc
        except groq.BadRequestError as exc:
            raise LLMError(f"The request was rejected by the API: {exc.message}") from exc
        except groq.APIStatusError as exc:
            log.warning("Groq API error %s: %s", exc.status_code, exc.message)
            raise LLMError(f"The Groq API returned an error ({exc.status_code}). Please try again.") from exc
        except groq.APIConnectionError as exc:
            raise LLMError("Couldn't reach the Groq API. Check your network connection.") from exc
        except groq.GroqError as exc:
            raise LLMError(f"Groq client error: {exc}") from exc

        return from_chat_completion(response)


def to_chat_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


def _field(block, name: str):
    """Read a block attribute whether the block is a dict or an object."""
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def to_chat_messages(system: str, messages: list) -> list[dict]:
    """Translate the internal message format to Groq chat messages."""
    out: list[dict] = [{"role": "system", "content": system}]
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        texts = [_field(b, "text") for b in content if _field(b, "type") == "text"]
        if role == "assistant":
            calls = [
                {
                    "id": _field(b, "id"),
                    "type": "function",
                    "function": {"name": _field(b, "name"), "arguments": json.dumps(_field(b, "input") or {})},
                }
                for b in content
                if _field(b, "type") == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": "\n".join(texts)}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
            continue
        # User turn: tool results become "tool" messages; any extra text follows as a user message.
        for b in content:
            if _field(b, "type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": _field(b, "tool_use_id"), "content": _field(b, "content")})
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def from_chat_completion(response) -> LLMResponse:
    """Translate a Groq chat completion into content blocks and a stop reason."""
    choice = response.choices[0]
    if choice.finish_reason == "content_filter":
        raise RefusalError("The model declined to answer this request.")
    message = choice.message
    blocks: list = []
    if message.content:
        blocks.append(TextBlock(message.content))
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}  # the tool reports the missing arguments back to the model
        blocks.append(ToolUseBlock(id=call.id, name=call.function.name, input=args if isinstance(args, dict) else {}))
    return LLMResponse(content=blocks, stop_reason=STOP_REASONS.get(choice.finish_reason, "end_turn"))


def response_text(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def parse_json(response) -> dict:
    """Parse a structured-output response; strict json_schema mode keeps it schema-valid."""
    if response.stop_reason == "max_tokens":
        raise LLMError("The model's structured response was cut off (max_tokens).")
    try:
        return json.loads(response_text(response))
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned malformed JSON.") from exc
