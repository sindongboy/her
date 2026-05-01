"""Smoke test: verify Gemini API key has access to the LLM models in CLAUDE.md §3.2.

Usage:
    uv run python scripts/smoke_llm.py

Verifies:
    - GEMINI_API_KEY is loaded
    - gemini-3.1-pro-preview  generates text  (main reasoning, both channels)
    - gemini-2.5-flash         generates text  (Consolidator / batch)

Exit codes:
    0  all models OK
    1  one or more models failed
"""

from __future__ import annotations

import os
import sys

from google import genai

PROMPT = "한 문장으로 자기소개해. 짧게."

MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print(
            "GEMINI_API_KEY not set. Edit .envrc and run `direnv allow`.",
            file=sys.stderr,
        )
        return 1

    client = genai.Client(api_key=api_key)
    failures = 0

    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=PROMPT)
            text = (resp.text or "").strip()
            preview = text if len(text) <= 120 else text[:120] + "..."
            print(f"OK  {model}")
            print(f"    -> {preview}\n")
        except Exception as e:
            print(f"FAIL {model}")
            print(f"    -> {type(e).__name__}: {e}\n", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
