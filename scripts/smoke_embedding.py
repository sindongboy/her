"""Smoke test: verify gemini-embedding-001 at 768d with asymmetric task_types.

Usage:
    uv run python scripts/smoke_embedding.py

Verifies (per CLAUDE.md §3.2):
    - GEMINI_API_KEY has embedding access
    - output_dimensionality = 768 (MRL truncation)
    - task_type RETRIEVAL_DOCUMENT (write-time) and RETRIEVAL_QUERY (search-time)
    - Asymmetric pair gives reasonable cosine similarity (> 0.4)

Exit codes:
    0  embedding OK
    1  API or dim error
    2  similarity unexpectedly low (configured but suspect)
"""

from __future__ import annotations

import math
import os
import sys

from google import genai
from google.genai import types

from apps.memory import EMBED_DIM, EMBED_MODEL_ID

DOC = "어머니는 단호박 케이크를 좋아하신다."
QUERY = "엄마가 좋아하는 디저트?"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(client: genai.Client, text: str, task_type: str) -> list[float]:
    resp = client.models.embed_content(
        model=EMBED_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return list(resp.embeddings[0].values)


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print(
            "GEMINI_API_KEY not set. Edit .envrc and run `direnv allow`.",
            file=sys.stderr,
        )
        return 1

    client = genai.Client(api_key=api_key)

    try:
        doc_vec = embed(client, DOC, "RETRIEVAL_DOCUMENT")
        query_vec = embed(client, QUERY, "RETRIEVAL_QUERY")
    except Exception as e:
        print(f"FAIL {EMBED_MODEL_ID}")
        print(f"    -> {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if len(doc_vec) != EMBED_DIM:
        print(
            f"FAIL doc dim mismatch: got {len(doc_vec)}, expected {EMBED_DIM}",
            file=sys.stderr,
        )
        return 1
    if len(query_vec) != EMBED_DIM:
        print(
            f"FAIL query dim mismatch: got {len(query_vec)}, expected {EMBED_DIM}",
            file=sys.stderr,
        )
        return 1

    sim = cosine(doc_vec, query_vec)

    print(f"OK  {EMBED_MODEL_ID} @ {EMBED_DIM}d")
    print(f"    doc:    {DOC}")
    print(f"    query:  {QUERY}")
    print(f"    cosine: {sim:.4f}")

    if sim < 0.4:
        print(
            f"WARN cosine unexpectedly low ({sim:.4f}); embedding may be suspect.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
