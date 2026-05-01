"""Gemini API adapter.

Thin wrapper over google.genai.Client with:
- sync generate() and async generate_stream()
- embed() for 768d asymmetric embeddings (CLAUDE.md §3.2)
- tenacity retry (3 attempts, exponential backoff) on transient errors
- 30-second request timeout
- Reads GEMINI_API_KEY from environment
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

_TRANSIENT_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)

EMBED_MODEL_ID = "gemini-embedding-001"
EMBED_DIM = 768


def _make_retry() -> Any:
    """Return a tenacity retry decorator for transient failures."""
    return retry(
        retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )


class GeminiClient:
    """Thin wrapper over google.genai.Client.

    Pass ``client`` to inject a test double; otherwise the real SDK client
    is constructed from GEMINI_API_KEY.
    """

    def __init__(
        self,
        *,
        model_id: str = "gemini-3.1-pro-preview",
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.model_id = model_id

        if client is not None:
            self._client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. "
                    "Add it to .envrc and run `direnv allow`."
                )
            try:
                from google import genai  # type: ignore[import-untyped]

                self._client = genai.Client(api_key=key)
            except ImportError as exc:
                raise ImportError(
                    "google-genai package is not installed. "
                    "Run `uv sync` to install dependencies."
                ) from exc

    # ── generate (sync) ─────────────────────────────────────────────────

    @_make_retry()
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        parts: list[Any] | None = None,
        enable_search_grounding: bool = False,
    ) -> str:
        """Send messages to Gemini and return the full text response.

        Args:
            messages: Conversation history as role/content dicts.
            system: System instruction string.
            parts: Extra Gemini Part objects (images, PDFs, text files) to
                   append to the last user turn for multimodal input.
            enable_search_grounding: When True, attach the Gemini built-in
                   google_search tool so the model can ground factual / current-
                   events answers via Google Search rather than relying on
                   training-cutoff knowledge.
        """
        try:
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError:
            types = None  # type: ignore[assignment]

        contents = _messages_to_contents(messages, types, extra_parts=parts)
        config_kwargs: dict[str, Any] = {}
        if system:
            if types is not None:
                config_kwargs["system_instruction"] = system
        if types is not None:
            if enable_search_grounding:
                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            gen_config = types.GenerateContentConfig(
                **config_kwargs,
            )
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=gen_config,
            )
        else:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=contents,
            )
        text: str = response.text or ""
        log.debug(
            "gemini.generate",
            model=self.model_id,
            chars=len(text),
            extra_parts=len(parts) if parts else 0,
            search_grounding=enable_search_grounding,
        )
        return text

    # ── generate_stream (async) ──────────────────────────────────────────

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        parts: list[Any] | None = None,
        enable_search_grounding: bool = False,
    ) -> AsyncIterator[str]:
        """Yield text deltas as Gemini streams the response.

        Args:
            messages: Conversation history as role/content dicts.
            system: System instruction string.
            parts: Extra Gemini Part objects for multimodal input.
            enable_search_grounding: When True, attach the Gemini built-in
                   google_search tool (same semantics as generate()).
        """
        import asyncio

        try:
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError:
            types = None  # type: ignore[assignment]

        contents = _messages_to_contents(messages, types, extra_parts=parts)
        config_kwargs: dict[str, Any] = {}
        if system:
            if types is not None:
                config_kwargs["system_instruction"] = system
        if types is not None:
            if enable_search_grounding:
                config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            gen_config = types.GenerateContentConfig(**config_kwargs)
        else:
            gen_config = None

        # The SDK's stream is synchronous; wrap in to_thread for async callers.
        def _stream() -> list[str]:
            chunks: list[str] = []
            if gen_config is not None:
                for chunk in self._client.models.generate_content_stream(
                    model=self.model_id,
                    contents=contents,
                    config=gen_config,
                ):
                    if chunk.text:
                        chunks.append(chunk.text)
            else:
                for chunk in self._client.models.generate_content_stream(
                    model=self.model_id,
                    contents=contents,
                ):
                    if chunk.text:
                        chunks.append(chunk.text)
            return chunks

        chunks = await asyncio.to_thread(_stream)
        for chunk in chunks:
            yield chunk

    # ── embed ────────────────────────────────────────────────────────────

    @_make_retry()
    def embed(self, text: str, *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Return 768-d embedding vector.

        task_type: RETRIEVAL_DOCUMENT (write) or RETRIEVAL_QUERY (search).
        Per CLAUDE.md §3.2 asymmetric usage rules.
        """
        try:
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError:
            types = None  # type: ignore[assignment]

        if types is not None:
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            )
            resp = self._client.models.embed_content(
                model=EMBED_MODEL_ID,
                contents=text,
                config=config,
            )
        else:
            resp = self._client.models.embed_content(
                model=EMBED_MODEL_ID,
                contents=text,
            )
        values: list[float] = list(resp.embeddings[0].values)
        if len(values) != EMBED_DIM:
            raise ValueError(
                f"Unexpected embedding dim: got {len(values)}, expected {EMBED_DIM}"
            )
        log.debug("gemini.embed", task_type=task_type, dim=len(values))
        return values


# ── helpers ──────────────────────────────────────────────────────────────


def _messages_to_contents(
    messages: list[dict[str, Any]],
    types: Any,
    *,
    extra_parts: list[Any] | None = None,
) -> list[Any]:
    """Convert [{"role": "user", "content": "..."}] to genai Contents.

    When extra_parts is provided, the additional Part objects are appended
    to the final user message, enabling multimodal input (images, PDFs, etc.).
    Parts may be SDK Part objects or dicts with _part_type key (offline/test).
    """
    if types is None:
        # Fallback: just pass plain text from the last user message.
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return [msg.get("content", "")]
        return []

    contents = []
    msg_list = list(messages)
    last_user_idx: int | None = None
    for i, msg in enumerate(msg_list):
        if msg.get("role") != "assistant":
            last_user_idx = i

    for idx, msg in enumerate(msg_list):
        role = msg.get("role", "user")
        text = msg.get("content", "")
        genai_role = "model" if role == "assistant" else "user"

        # Build parts list for this turn.
        turn_parts: list[Any] = [types.Part(text=text)]

        # Append extra_parts to the last user turn.
        if extra_parts and idx == last_user_idx:
            for part in extra_parts:
                if isinstance(part, dict):
                    # dict fallback from offline/test environment
                    part_type = part.get("_part_type")
                    if part_type == "text":
                        turn_parts.append(types.Part(text=part["text"]))
                    elif part_type == "inline_data":
                        try:
                            turn_parts.append(
                                types.Part.from_bytes(
                                    data=part["data"],
                                    mime_type=part["mime_type"],
                                )
                            )
                        except AttributeError:
                            turn_parts.append(
                                types.Part(
                                    inline_data=types.Blob(
                                        data=part["data"],
                                        mime_type=part["mime_type"],
                                    )
                                )
                            )
                else:
                    # SDK Part object — append directly.
                    turn_parts.append(part)

        contents.append(
            types.Content(
                role=genai_role,
                parts=turn_parts,
            )
        )
    return contents
