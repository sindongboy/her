"""Unit tests for apps/agent/gemini.py — structure only, no live API calls.

Tests:
- retry decorator is wired (tenacity configured)
- env var check raises ValueError when missing
- _messages_to_contents produces correct shape
- embed validates output dimensionality
- GeminiClient can be constructed with injected client
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apps.agent.gemini import EMBED_DIM, GeminiClient, _messages_to_contents


# ── env var handling ──────────────────────────────────────────────────────


class TestEnvVarCheck:
    def test_raises_if_no_api_key_and_no_client(self) -> None:
        """Without a client or env var, GeminiClient must raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if present
            env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises((ValueError, ImportError)):
                    GeminiClient()

    def test_accepts_explicit_api_key(self) -> None:
        """Passing api_key explicitly should not raise (we just check it is set)."""
        mock_sdk = MagicMock()
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_sdk
        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            # Should not raise — explicit key provided.
            client = GeminiClient(api_key="fake-key-123", model_id="test-model")
            assert client is not None

    def test_injected_client_skips_env_check(self) -> None:
        """Injecting a client object must bypass GEMINI_API_KEY lookup entirely."""
        fake = MagicMock()
        client = GeminiClient(client=fake)
        assert client._client is fake


# ── retry decorator ───────────────────────────────────────────────────────


class TestRetryDecorator:
    def test_generate_is_retried_on_timeout(self) -> None:
        """generate() should retry on TimeoutError up to 3 times."""
        fake_client = MagicMock()
        call_count = 0

        def flaky_generate(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            result = MagicMock()
            result.text = "success"
            return result

        fake_client.models.generate_content.side_effect = flaky_generate

        gemini = GeminiClient(client=fake_client)
        # Patch tenacity wait to zero so test runs fast.
        with patch("apps.agent.gemini.wait_exponential", return_value=lambda _: 0):
            result = gemini.generate([{"role": "user", "content": "hello"}])
        assert result == "success"
        assert call_count == 3

    def test_embed_is_retried_on_connection_error(self) -> None:
        """embed() should retry on ConnectionError."""
        fake_client = MagicMock()
        call_count = 0

        def flaky_embed(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("conn reset")
            resp = MagicMock()
            embedding = MagicMock()
            embedding.values = [0.1] * EMBED_DIM
            resp.embeddings = [embedding]
            return resp

        fake_client.models.embed_content.side_effect = flaky_embed
        gemini = GeminiClient(client=fake_client)
        with patch("apps.agent.gemini.wait_exponential", return_value=lambda _: 0):
            result = gemini.embed("test text")
        assert len(result) == EMBED_DIM
        assert call_count == 2

    def test_non_transient_error_not_retried(self) -> None:
        """A ValueError (not transient) should propagate immediately."""
        fake_client = MagicMock()
        call_count = 0

        def bad_generate(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        fake_client.models.generate_content.side_effect = bad_generate
        gemini = GeminiClient(client=fake_client)
        with pytest.raises(ValueError):
            gemini.generate([{"role": "user", "content": "hi"}])
        assert call_count == 1  # No retry for ValueError


# ── _messages_to_contents helper ──────────────────────────────────────────


class TestMessagesToContents:
    def test_returns_list(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        result = _messages_to_contents(msgs, types=None)
        assert isinstance(result, list)

    def test_extracts_last_user_message_when_no_types(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "second"},
        ]
        result = _messages_to_contents(msgs, types=None)
        assert result == ["second"]

    def test_empty_messages_returns_empty_when_no_types(self) -> None:
        result = _messages_to_contents([], types=None)
        assert result == []

    def test_with_types_maps_roles(self) -> None:
        """When types is available, assistant role maps to 'model'."""
        mock_types = MagicMock()
        # Make Content/Part constructors return identifiable objects.
        mock_types.Content.side_effect = lambda **kw: kw
        mock_types.Part.side_effect = lambda **kw: kw

        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _messages_to_contents(msgs, types=mock_types)
        assert len(result) == 2
        # First message is user
        assert result[0]["role"] == "user"
        # Second message is model (Gemini's term for assistant)
        assert result[1]["role"] == "model"


# ── embed output dimensionality ───────────────────────────────────────────


class TestEmbedDimCheck:
    def test_raises_on_wrong_dimension(self) -> None:
        """embed() must raise ValueError if the API returns wrong dimension."""
        fake_client = MagicMock()
        resp = MagicMock()
        embedding = MagicMock()
        embedding.values = [0.1] * 512  # Wrong dim
        resp.embeddings = [embedding]
        fake_client.models.embed_content.return_value = resp

        gemini = GeminiClient(client=fake_client)
        with pytest.raises(ValueError, match="dim"):
            gemini.embed("test")

    def test_correct_dimension_passes(self) -> None:
        fake_client = MagicMock()
        resp = MagicMock()
        embedding = MagicMock()
        embedding.values = [0.1] * EMBED_DIM
        resp.embeddings = [embedding]
        fake_client.models.embed_content.return_value = resp

        gemini = GeminiClient(client=fake_client)
        result = gemini.embed("test")
        assert len(result) == EMBED_DIM


# ── model_id passthrough ──────────────────────────────────────────────────


class TestModelId:
    def test_default_model_id(self) -> None:
        fake = MagicMock()
        client = GeminiClient(client=fake)
        assert client.model_id == "gemini-3.1-pro-preview"

    def test_custom_model_id(self) -> None:
        fake = MagicMock()
        client = GeminiClient(client=fake, model_id="gemini-2.5-flash")
        assert client.model_id == "gemini-2.5-flash"
