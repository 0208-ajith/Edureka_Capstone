"""Claude API client wrapper with consistent error handling (used by Tasks 7-9)."""

from __future__ import annotations

import json
import logging

import anthropic

from .config import Settings

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """A failure talking to the model. The message is safe to show to users."""


class RefusalError(LLMError):
    """The model declined the request on safety grounds."""


class ClaudeLLM:
    def __init__(self, settings: Settings, api_key: str | None = None):
        self.settings = settings
        # With api_key=None the SDK reads ANTHROPIC_API_KEY (or another configured credential).
        self._client = anthropic.Anthropic(api_key=api_key or None, max_retries=3, timeout=180.0)

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
    ):
        """Send one Messages API request and return the response."""
        output_config: dict = {"effort": effort}
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        kwargs: dict = {
            "model": self.settings.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": output_config,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if self.settings.enable_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("The Anthropic API key is missing or invalid.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("The Anthropic API key doesn't have permission for this request.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model '{self.settings.model}' was not found. Check CLAUDE_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limit reached. Please wait a moment and try again.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"The request was rejected by the API: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            log.warning("Anthropic API error %s: %s", exc.status_code, exc.message)
            raise LLMError(f"The Claude API returned an error ({exc.status_code}). Please try again.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Couldn't reach the Claude API. Check your network connection.") from exc
        except anthropic.AnthropicError as exc:  # e.g. no credentials configured at all
            raise LLMError(f"Claude client error: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RefusalError(
                "The model declined to answer this request"
                + (f" (category: {category})." if category else ".")
            )
        return response


def response_text(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


def parse_json(response) -> dict:
    """Parse a structured-output response; output_config.format guarantees valid JSON."""
    if response.stop_reason == "max_tokens":
        raise LLMError("The model's structured response was cut off (max_tokens).")
    try:
        return json.loads(response_text(response))
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned malformed JSON.") from exc
