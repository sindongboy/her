from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from apps.memory import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "test.db")
    try:
        yield s
    finally:
        s.close()
