"""FastAPI + WebSocket app for the her text-first web chat.

CLAUDE.md references:
  §1.3  Local-First — server binds to 127.0.0.1 only.
  §2.5  Multi-channel: this app is the single user surface.
  §4    Architecture — Agent Core / channels publish events; the WS endpoint
        forwards them to the connected browser.

Routes
------
  GET  /                       → web/index.html
  GET  /static/*               → static frontend assets (no-store cache)
  GET  /healthz                → liveness probe
  GET  /api/sessions           → recent sessions
  POST /api/sessions           → create a new session, returns id
  GET  /api/sessions/{id}/messages
  DELETE /api/sessions/{id}    → soft archive
  GET  /api/memory/probe       → recall snapshot for the right panel
  GET  /api/memory/notes       → list of active notes
  GET  /api/memory/people      → list of people
  GET  /api/memory/facts       → list of active facts
  WS   /ws/chat                → token-streamed chat with sidechannel events
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from apps.web.eventbus import EventBus, get_default_bus

log = structlog.get_logger(__name__)

_DEFAULT_STATIC = Path(__file__).parent / "web"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})

_SCHEMA_VERSION = 2  # WS protocol version


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def _build_default_agent_and_store() -> tuple[Any, Any]:
    """Build the production AgentCore and MemoryStore from env / defaults.

    Imports are local so tests that supply their own agent never pay the
    cost of opening the real DB or initialising Gemini.
    """
    from apps.agent.core import AgentCore
    from apps.memory.store import MemoryStore

    db_path = Path(os.environ.get("HER_DB_PATH", "data/db.sqlite"))
    store = MemoryStore(db_path)
    agent = AgentCore(store, bus=get_default_bus())
    return agent, store


def create_app(
    *,
    agent: Any | None = None,
    store: Any | None = None,
    bus: EventBus | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI application."""
    if bus is None:
        bus = get_default_bus()

    if agent is None or store is None:
        if agent is not None or store is not None:
            raise RuntimeError("agent and store must both be provided or both omitted")
        agent, store = _build_default_agent_and_store()

    resolved_static = (static_dir or _DEFAULT_STATIC).resolve()

    app = FastAPI(title="her", docs_url=None, redoc_url=None)
    app.state.agent = agent
    app.state.store = store
    app.state.bus = bus

    if resolved_static.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(resolved_static)),
            name="static",
        )

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        index_file = resolved_static / "index.html"
        if index_file.exists():
            return FileResponse(
                str(index_file),
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )
        return HTMLResponse(
            "<html><body><h1>her</h1><p>frontend not built.</p></body></html>",
            status_code=200,
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> Any:
        return {"status": "ok", "subscribers": bus.subscriber_count()}

    @app.get("/api/sessions")
    async def list_sessions(request: Request) -> Any:
        _require_loopback(request)
        sessions = store.list_recent_sessions(limit=50)
        return [
            {
                "id": s.id,
                "title": s.title,
                "summary": s.summary,
                "started_at": s.started_at,
                "last_active_at": s.last_active_at,
            }
            for s in sessions
        ]

    @app.post("/api/sessions")
    async def create_session(request: Request) -> Any:
        _require_loopback(request)
        sid = store.add_session()
        s = store.get_session(sid)
        return {
            "id": s.id,
            "title": s.title,
            "summary": s.summary,
            "started_at": s.started_at,
            "last_active_at": s.last_active_at,
        }

    @app.get("/api/sessions/{session_id}/messages")
    async def get_messages(session_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        msgs = store.list_messages(session_id)
        return [
            {"id": m.id, "role": m.role, "content": m.content, "ts": m.ts}
            for m in msgs
        ]

    @app.delete("/api/sessions/{session_id}")
    async def archive_session(session_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        store.archive_session(session_id)
        return {"ok": True}

    @app.get("/api/memory/notes")
    async def list_memory_notes(request: Request) -> Any:
        _require_loopback(request)
        notes = store.list_notes()
        return [
            {
                "id": n.id,
                "content": n.content,
                "tags": n.tags,
                "created_at": n.created_at,
                "updated_at": n.updated_at,
            }
            for n in notes
        ]

    @app.get("/api/memory/people")
    async def list_memory_people(request: Request) -> Any:
        _require_loopback(request)
        return [
            {
                "id": p.id,
                "name": p.name,
                "relation": p.relation,
                "birthday": p.birthday,
            }
            for p in store.list_people()
        ]

    @app.get("/api/memory/facts")
    async def list_memory_facts(request: Request) -> Any:
        _require_loopback(request)
        rows = store.conn.execute(
            "SELECT f.id AS id, f.predicate, f.object, f.confidence, "
            "       p.name AS person_name "
            "FROM facts f LEFT JOIN people p ON p.id = f.subject_person_id "
            "WHERE f.archived_at IS NULL "
            "ORDER BY f.valid_from DESC LIMIT 200"
        ).fetchall()
        return [
            {
                "id": r["id"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
                "person_name": r["person_name"],
            }
            for r in rows
        ]

    # ── /api/widgets/* ────────────────────────────────────────────────────

    @app.get("/api/widgets/weather")
    async def widget_weather(request: Request) -> Any:
        _require_loopback(request)
        from apps.tools.weather import WeatherUnavailable, get_current_weather
        from apps.settings import load_settings
        s = load_settings()
        try:
            snap = await get_current_weather(
                lat=s.location_lat,
                lon=s.location_lon,
                location_name=s.location_name,
            )
            return {
                "temperature_c": snap.temperature_c,
                "apparent_c": snap.feels_like_c,
                "humidity_pct": snap.humidity_pct,
                "wind_ms": round(snap.wind_kmh / 3.6, 1),
                "condition_code": snap.weather_code,
                "condition_ko": snap.weather_text_ko,
                "location_name": snap.location_name,
                "fetched_at": snap.fetched_at_iso,
            }
        except WeatherUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/api/widgets/calendar")
    async def widget_calendar(request: Request) -> Any:
        _require_loopback(request)
        from apps.tools.calendar_mac import CalendarUnavailable, get_events
        from apps.settings import load_settings
        s = load_settings()

        merged: list[dict[str, Any]] = []

        try:
            events = await get_events(
                days_ahead=int(getattr(s, "calendar_lookahead_days", 14)),
                max_events=50,
            )
            for e in events:
                merged.append({
                    "title": e.title,
                    "calendar": e.calendar_name,
                    "all_day": e.is_all_day,
                    "when_at": e.starts_at_iso,
                    "end_at": e.ends_at_iso,
                    "source": "macos",
                })
        except CalendarUnavailable as exc:
            log.debug("widget.calendar.macos_unavailable", error=str(exc))

        # Also include DB events (Consolidator-derived).
        try:
            db_events = store.list_upcoming_events(within_hours=24 * 14)
            for ev in db_events:
                merged.append({
                    "title": ev.title,
                    "calendar": "her",
                    "all_day": False,
                    "when_at": ev.when_at,
                    "end_at": None,
                    "source": "db",
                })
        except Exception as exc:
            log.warning("widget.calendar.db_query_failed", error=str(exc))

        merged.sort(key=lambda e: e["when_at"] or "")
        return merged[:30]

    @app.get("/api/widgets/stocks")
    async def widget_stocks(request: Request, tickers: str = "") -> Any:
        _require_loopback(request)
        from apps.tools.stocks import get_quotes
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not ticker_list:
            return []
        quotes = await get_quotes(ticker_list)
        return quotes

    @app.get("/api/memory/probe")
    async def memory_probe(request: Request, session_id: int | None = None) -> Any:
        _require_loopback(request)
        notes = store.list_notes()[:10]
        people = store.list_people()[:20]
        facts_rows = store.conn.execute(
            "SELECT f.id, f.predicate, f.object, p.name AS person_name "
            "FROM facts f LEFT JOIN people p ON p.id = f.subject_person_id "
            "WHERE f.archived_at IS NULL ORDER BY f.valid_from DESC LIMIT 20"
        ).fetchall()
        events = store.list_upcoming_events(within_hours=24 * 14)
        return {
            "session_id": session_id,
            "notes": [
                {"id": n.id, "content": n.content, "tags": n.tags} for n in notes
            ],
            "people": [
                {"id": p.id, "name": p.name, "relation": p.relation} for p in people
            ],
            "facts": [
                {
                    "id": r["id"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "person_name": r["person_name"],
                }
                for r in facts_rows
            ],
            "upcoming_events": [
                {"id": e.id, "title": e.title, "when_at": e.when_at} for e in events
            ],
        }

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "schema_version": _SCHEMA_VERSION,
                    "ts": time.monotonic(),
                }
            )
        )
        log.info("ws.chat.connected")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    await _send_error(websocket, f"invalid JSON: {exc}")
                    continue

                if payload.get("type") != "message":
                    await _send_error(websocket, "expected type=message")
                    continue

                content = payload.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    await _send_error(websocket, "content must be a non-empty string")
                    continue

                session_id = payload.get("session_id")
                if session_id is not None and not isinstance(session_id, int):
                    await _send_error(websocket, "session_id must be int")
                    continue

                await _handle_message(
                    websocket,
                    agent=agent,
                    store=store,
                    content=content,
                    session_id=session_id,
                )
        except WebSocketDisconnect:
            log.info("ws.chat.disconnected")
        except Exception as exc:
            log.warning("ws.chat.error", error=str(exc))
            try:
                await _send_error(websocket, str(exc))
                await websocket.close()
            except Exception:
                pass

    return app


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if not _is_loopback(host):
        raise HTTPException(status_code=403, detail="loopback only")


async def _send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_text(
            json.dumps({"type": "error", "message": message, "ts": time.monotonic()})
        )
    except Exception:
        pass


async def _handle_message(
    websocket: WebSocket,
    *,
    agent: Any,
    store: Any,
    content: str,
    session_id: int | None,
) -> None:
    """Run one user → assistant turn, streaming tokens + recall events."""
    if session_id is None:
        session_id = store.add_session()
    elif store.get_session(session_id) is None:
        await _send_error(websocket, f"session_id {session_id} not found")
        return

    log.info("ws.chat.turn.start", session_id=session_id, chars=len(content))

    try:
        recall_payload = await _build_recall_payload(agent, store, content, session_id)
        await websocket.send_text(
            json.dumps({"type": "recall", "session_id": session_id, **recall_payload})
        )
    except Exception as exc:
        log.warning("ws.chat.recall_failed", error=str(exc))

    try:
        accumulated: list[str] = []
        stream: AsyncIterator[str] = agent.stream_respond(
            content, session_id=session_id
        )
        async for chunk in stream:
            accumulated.append(chunk)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "token",
                        "session_id": session_id,
                        "text": chunk,
                    }
                )
            )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "done",
                    "session_id": session_id,
                    "chars": sum(len(c) for c in accumulated),
                }
            )
        )
        log.info(
            "ws.chat.turn.done",
            session_id=session_id,
            response_chars=sum(len(c) for c in accumulated),
        )
    except Exception as exc:
        log.error("ws.chat.turn.failed", session_id=session_id, error=str(exc))
        await _send_error(websocket, f"agent error: {exc}")


async def _build_recall_payload(
    agent: Any, store: Any, content: str, session_id: int
) -> dict[str, Any]:
    """Run recall once so the right panel can show what was looked up."""
    from apps.agent.recall import recall

    ctx = await recall(store, content, agent._gemini, session_id=session_id)
    return {
        "facts": [
            {"id": fid, "person_name": pname, "predicate": pred, "object": obj}
            for fid, pname, pred, obj in ctx.facts
        ],
        "notes": [{"id": nid, "content": nc} for nid, nc in ctx.notes],
        "events": [
            {"id": eid, "title": title, "when_at": when}
            for eid, title, when in ctx.upcoming_events
        ],
        "sessions": [
            {"id": sid, "summary": text, "score": score}
            for sid, text, score in ctx.sessions
        ],
    }
