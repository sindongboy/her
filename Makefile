.PHONY: help setup web dev consolidate backup smoke test fmt lint typecheck

PY = uv run

help:
	@echo "Targets:"
	@echo "  setup        — uv sync + create data dirs + copy .envrc.example if needed"
	@echo "  web          — start the her web app (http://127.0.0.1:\$$HER_WEB_PORT, default 8765)"
	@echo "  dev          — alias for 'web'"
	@echo "  consolidate  — run memory consolidation over the last 24h"
	@echo "  smoke        — run LLM + embedding smoke tests"
	@echo "  test         — pytest -x -q"
	@echo "  fmt          — ruff format + ruff check --fix"
	@echo "  lint         — ruff check"
	@echo "  typecheck    — mypy apps"
	@echo "  backup       — tar.gz of data/ to backups/"

setup:
	uv sync
	mkdir -p data/attachments data/consolidation_log tests/fixtures
	@if [ ! -f .envrc ]; then \
		if [ -f .envrc.example ]; then \
			cp .envrc.example .envrc; \
			echo "Copied .envrc.example → .envrc. Edit it with your GEMINI_API_KEY, then run: direnv allow"; \
		else \
			echo "export GEMINI_API_KEY=your_key_here" > .envrc; \
			echo "Created .envrc. Edit it with your GEMINI_API_KEY, then run: direnv allow"; \
		fi; \
	fi

web:
	$(PY) python -m apps.web

dev: web

consolidate:
	$(PY) python -m apps.consolidator

backup:
	@TS=$$(date +%Y%m%d-%H%M%S); \
	mkdir -p backups; \
	tar czf backups/her-backup-$$TS.tar.gz data/ && echo "Wrote backups/her-backup-$$TS.tar.gz"

smoke:
	$(PY) python scripts/smoke_llm.py
	$(PY) python scripts/smoke_embedding.py

test:
	$(PY) pytest -x -q

fmt:
	$(PY) ruff format apps tests scripts
	$(PY) ruff check --fix apps tests scripts

lint:
	$(PY) ruff check apps tests scripts

typecheck:
	$(PY) mypy apps
