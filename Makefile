.PHONY: setup first-run text voice voice-smoke dev consolidate consolidator-install consolidator-uninstall replay seed backup smoke test fmt lint typecheck help daemon-install daemon-uninstall daemon-start daemon-stop daemon-status daemon-logs presence presence-voice presence-text voice-with-presence text-with-presence

PY = uv run

help:
	@echo "Targets: setup text voice voice-smoke dev consolidate replay seed backup smoke test fmt lint typecheck"
	@echo "         daemon-install daemon-uninstall daemon-start daemon-stop daemon-status daemon-logs"
	@echo "         presence presence-voice presence-text voice-with-presence text-with-presence"
	@echo ""
	@echo "  setup            — uv sync + create data dirs + copy .envrc.example if needed"
	@echo "  first-run        — interactive consent + env-var sanity checks (run before daemon)"
	@echo "  text             — start text-channel REPL (Phase 0 entry point)"
	@echo "  voice            — Phase 1: start voice-channel loop (requires portaudio + GEMINI_API_KEY)"
	@echo "  voice-smoke      — Phase 1: manual hardware smoke test (mic + TTS round-trip)"
	@echo "  dev              — alias for 'text' (running both channels in one terminal is messy;"
	@echo "                     use two terminals: 'make text' and 'make voice' separately)"
	@echo "  consolidate            — Phase 4: run daily memory consolidation (last 24h → semantic memory)"
	@echo "  consolidator-install   — Phase 4: install consolidator launchd agent (daily 03:00)"
	@echo "  consolidator-uninstall — Phase 4: remove consolidator launchd agent"
	@echo "  replay           — run regression replay  (FILE=<path> required)"
	@echo "  seed             — seed initial family data interactively"
	@echo "  backup           — create timestamped .tar.gz of data/"
	@echo "  smoke            — run LLM + embedding smoke tests"
	@echo "  test             — pytest -x -q (all unit tests)"
	@echo "  fmt              — ruff format + ruff check --fix"
	@echo "  lint             — ruff check (no fix)"
	@echo "  typecheck        — mypy apps"
	@echo ""
	@echo "  daemon-install   — Phase 3: install launchd agent (auto-start on login)"
	@echo "  daemon-uninstall — Phase 3: remove launchd agent"
	@echo "  daemon-start     — Phase 3: start daemon directly (inherits shell env)"
	@echo "  daemon-stop      — Phase 3: stop daemon gracefully via SIGTERM"
	@echo "  daemon-status    — Phase 3: show daemon PID and uptime"
	@echo "  daemon-logs      — Phase 3: tail live daemon log"
	@echo ""
	@echo "  presence             — Phase 3.5: 서버만 시작 (브라우저: http://127.0.0.1:8765)"
	@echo "                         다른 터미널에서 채널 연결:"
	@echo "                           HER_PRESENCE_URL=http://127.0.0.1:8765 make voice"
	@echo "                           HER_PRESENCE_URL=http://127.0.0.1:8765 make text"
	@echo "  presence-voice       — Phase 3.5: 서버 + 음성 채널 (같은 프로세스)"
	@echo "  presence-text        — Phase 3.5: 서버 + 텍스트 REPL (같은 프로세스)"
	@echo "  voice-with-presence  — 음성 채널 + 실행 중인 presence 서버에 연결"
	@echo "  text-with-presence   — 텍스트 채널 + 실행 중인 presence 서버에 연결"

setup:
	@brew list portaudio >/dev/null 2>&1 || echo "WARNING: portaudio not detected. Run: brew install portaudio (required for voice channel)"
	uv sync
	mkdir -p data/attachments data/audio_logs data/consolidation_log tests/fixtures
	@if [ ! -f .envrc ]; then \
		if [ -f .envrc.example ]; then \
			cp .envrc.example .envrc; \
			echo "Copied .envrc.example → .envrc. Edit it with your GEMINI_API_KEY, then run: direnv allow"; \
		else \
			echo "export GEMINI_API_KEY=your_key_here" > .envrc; \
			echo "Created .envrc. Edit it with your GEMINI_API_KEY, then run: direnv allow"; \
		fi; \
	fi

first-run:
	$(PY) python scripts/first_run.py

text:
	$(PY) python -m apps.channels.text

voice:
	$(PY) python -m apps.channels.voice

voice-smoke:
	$(PY) python scripts/smoke_voice.py

voice-diagnose:
	$(PY) python scripts/diagnose_mic.py

# dev: running both channels simultaneously in a single terminal is awkward
# (mixed stdout/stderr, no clean shutdown). Use two terminals instead:
#   Terminal A: make text
#   Terminal B: make voice
# For now, dev is an alias for text (primary entry point).
dev: text

consolidate:
	$(PY) python -m apps.consolidator

consolidator-install:
	bin/her consolidator-install

consolidator-uninstall:
	bin/her consolidator-uninstall

replay:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make replay FILE=tests/fixtures/dialog_001.jsonl"; exit 2; \
	fi
	$(PY) python scripts/replay.py "$(FILE)"

seed:
	$(PY) python scripts/seed_family.py

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

# ---------------------------------------------------------------------------
# Phase 3 — Daemon (launchd background service)
# ---------------------------------------------------------------------------

daemon-install:
	bin/her install

daemon-uninstall:
	bin/her uninstall

daemon-start:
	bin/her start

daemon-stop:
	bin/her stop

daemon-status:
	bin/her status

daemon-logs:
	bin/her logs

# ---------------------------------------------------------------------------
# Phase 3.5 — Presence Channel (Samantha-style orb WebSocket server)
# ---------------------------------------------------------------------------

presence:
	$(PY) python -m apps.presence
	# 서버만 시작합니다. 브라우저: http://127.0.0.1:8765
	# 다른 터미널에서: HER_PRESENCE_URL=http://127.0.0.1:8765 make voice

presence-voice:
	$(PY) python -m apps.presence --with-voice

presence-text:
	$(PY) python -m apps.presence --with-text

# Cross-process: drive a running presence server from this terminal.
voice-with-presence:
	HER_PRESENCE_URL=http://127.0.0.1:8765 $(MAKE) voice

text-with-presence:
	HER_PRESENCE_URL=http://127.0.0.1:8765 $(MAKE) text
