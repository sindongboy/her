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

    # ── Notes CRUD ────────────────────────────────────────────────────────

    @app.get("/api/memory/notes")
    async def list_memory_notes(request: Request, include_archived: bool = False) -> Any:
        _require_loopback(request)
        notes = store.list_notes(include_archived=include_archived)
        return [_note_dict(n) for n in notes]

    @app.post("/api/memory/notes")
    async def create_memory_note(request: Request) -> Any:
        _require_loopback(request)
        body = await request.json()
        content = (body.get("content") or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="content required")
        tags = body.get("tags") or []
        if not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="tags must be a list")
        note_id = store.add_note(content=content, tags=[str(t) for t in tags])
        return {"id": note_id}

    @app.patch("/api/memory/notes/{note_id}")
    async def update_memory_note(note_id: int, request: Request) -> Any:
        _require_loopback(request)
        body = await request.json()
        content = body.get("content")
        tags = body.get("tags")
        if content is None and tags is None:
            raise HTTPException(status_code=400, detail="nothing to update")
        store.update_note(
            note_id,
            content=(content.strip() if isinstance(content, str) else None),
            tags=[str(t) for t in tags] if isinstance(tags, list) else None,
        )
        return {"ok": True}

    @app.delete("/api/memory/notes/{note_id}")
    async def archive_memory_note(note_id: int, request: Request) -> Any:
        _require_loopback(request)
        store.archive_note(note_id)
        return {"ok": True}

    @app.post("/api/memory/notes/{note_id}/restore")
    async def restore_memory_note(note_id: int, request: Request) -> Any:
        _require_loopback(request)
        store.conn.execute(
            "UPDATE notes SET archived_at = NULL WHERE id = ?", (note_id,)
        )
        return {"ok": True}

    # ── People CRUD ───────────────────────────────────────────────────────

    @app.get("/api/memory/people")
    async def list_memory_people(request: Request, include_archived: bool = False) -> Any:
        _require_loopback(request)
        return [_person_dict(p) for p in store.list_people(include_archived=include_archived)]

    @app.post("/api/memory/people")
    async def create_memory_person(request: Request) -> Any:
        _require_loopback(request)
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        person_id = store.add_person(
            name=name,
            relation=body.get("relation") or None,
            birthday=body.get("birthday") or None,
        )
        return {"id": person_id}

    @app.patch("/api/memory/people/{person_id}")
    async def update_memory_person(person_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_person(person_id) is None:
            raise HTTPException(status_code=404, detail="person not found")
        body = await request.json()
        store.update_person(
            person_id,
            name=body.get("name"),
            relation=body.get("relation"),
            birthday=body.get("birthday"),
        )
        return {"ok": True}

    @app.delete("/api/memory/people/{person_id}")
    async def archive_memory_person(person_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_person(person_id) is None:
            raise HTTPException(status_code=404, detail="person not found")
        store.archive_person(person_id)
        return {"ok": True}

    @app.post("/api/memory/people/{person_id}/restore")
    async def restore_memory_person(person_id: int, request: Request) -> Any:
        _require_loopback(request)
        store.restore_person(person_id)
        return {"ok": True}

    # ── Facts CRUD ────────────────────────────────────────────────────────

    @app.get("/api/memory/facts")
    async def list_memory_facts(request: Request, include_archived: bool = False) -> Any:
        _require_loopback(request)
        if include_archived:
            sql = (
                "SELECT f.id, f.predicate, f.object, f.confidence, "
                "       f.archived_at, f.subject_person_id, "
                "       f.source_session_id, f.valid_from, "
                "       p.name AS person_name "
                "FROM facts f LEFT JOIN people p ON p.id = f.subject_person_id "
                "ORDER BY f.valid_from DESC LIMIT 500"
            )
        else:
            sql = (
                "SELECT f.id, f.predicate, f.object, f.confidence, "
                "       f.archived_at, f.subject_person_id, "
                "       f.source_session_id, f.valid_from, "
                "       p.name AS person_name "
                "FROM facts f LEFT JOIN people p ON p.id = f.subject_person_id "
                "WHERE f.archived_at IS NULL "
                "ORDER BY f.valid_from DESC LIMIT 500"
            )
        rows = store.conn.execute(sql).fetchall()
        return [
            {
                "id": r["id"],
                "subject_person_id": r["subject_person_id"],
                "person_name": r["person_name"],
                "predicate": r["predicate"],
                "object": r["object"],
                "confidence": r["confidence"],
                "source_session_id": r["source_session_id"],
                "valid_from": r["valid_from"],
                "archived_at": r["archived_at"],
            }
            for r in rows
        ]

    @app.post("/api/memory/facts")
    async def create_memory_fact(request: Request) -> Any:
        _require_loopback(request)
        body = await request.json()
        person_id = body.get("subject_person_id")
        predicate = (body.get("predicate") or "").strip()
        obj = (body.get("object") or "").strip()
        if not person_id or not predicate or not obj:
            raise HTTPException(status_code=400, detail="subject_person_id, predicate, object required")
        confidence = float(body.get("confidence", 1.0))
        if not (0.0 <= confidence <= 1.0):
            raise HTTPException(status_code=400, detail="confidence must be 0..1")
        fact_id = store.add_fact(
            int(person_id), predicate, obj,
            confidence=confidence,
            source_session_id=body.get("source_session_id"),
        )
        return {"id": fact_id}

    @app.patch("/api/memory/facts/{fact_id}")
    async def update_memory_fact(fact_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_fact(fact_id) is None:
            raise HTTPException(status_code=404, detail="fact not found")
        body = await request.json()
        try:
            store.update_fact(
                fact_id,
                predicate=body.get("predicate"),
                object=body.get("object"),
                confidence=body.get("confidence"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True}

    @app.delete("/api/memory/facts/{fact_id}")
    async def archive_memory_fact(fact_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_fact(fact_id) is None:
            raise HTTPException(status_code=404, detail="fact not found")
        store.archive_fact(fact_id)
        return {"ok": True}

    @app.post("/api/memory/facts/{fact_id}/restore")
    async def restore_memory_fact(fact_id: int, request: Request) -> Any:
        _require_loopback(request)
        store.restore_fact(fact_id)
        return {"ok": True}

    # ── Events CRUD ───────────────────────────────────────────────────────

    @app.get("/api/memory/events")
    async def list_memory_events(request: Request, include_archived: bool = False) -> Any:
        _require_loopback(request)
        events = store.list_all_events(include_archived=include_archived)
        return [_event_dict(e) for e in events]

    @app.post("/api/memory/events")
    async def create_memory_event(request: Request) -> Any:
        _require_loopback(request)
        body = await request.json()
        title = (body.get("title") or "").strip()
        ev_type = (body.get("type") or "").strip()
        when_at = (body.get("when_at") or "").strip()
        if not title or not ev_type or not when_at:
            raise HTTPException(status_code=400, detail="type, title, when_at required")
        event_id = store.add_event(
            type=ev_type,
            title=title,
            when_at=when_at,
            person_id=body.get("person_id"),
            recurrence=body.get("recurrence") or None,
            source=body.get("source") or "user",
        )
        return {"id": event_id}

    @app.patch("/api/memory/events/{event_id}")
    async def update_memory_event(event_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_event(event_id) is None:
            raise HTTPException(status_code=404, detail="event not found")
        body = await request.json()
        store.update_event(
            event_id,
            type=body.get("type"),
            title=body.get("title"),
            when_at=body.get("when_at"),
            recurrence=body.get("recurrence"),
            person_id=body.get("person_id"),
        )
        return {"ok": True}

    @app.delete("/api/memory/events/{event_id}")
    async def archive_memory_event(event_id: int, request: Request) -> Any:
        _require_loopback(request)
        if store.get_event(event_id) is None:
            raise HTTPException(status_code=404, detail="event not found")
        store.archive_event(event_id)
        return {"ok": True}

    @app.post("/api/memory/events/{event_id}/restore")
    async def restore_memory_event(event_id: int, request: Request) -> Any:
        _require_loopback(request)
        store.restore_event(event_id)
        return {"ok": True}

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
        # Return up to 100 — the calendar widget slices client-side based on
        # its per-instance max_events setting.
        return merged[:100]

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


def _note_dict(n: Any) -> dict[str, Any]:
    return {
        "id": n.id,
        "content": n.content,
        "tags": n.tags,
        "source_session_id": n.source_session_id,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
        "archived_at": n.archived_at,
    }


def _person_dict(p: Any) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "relation": p.relation,
        "birthday": p.birthday,
        "preferences": p.preferences,
        "archived_at": p.archived_at,
    }


def _event_dict(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "person_id": e.person_id,
        "type": e.type,
        "title": e.title,
        "when_at": e.when_at,
        "recurrence": e.recurrence,
        "source": e.source,
        "status": e.status,
        "archived_at": e.archived_at,
    }


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

    # First: detect "기억해줘" intent and persist before recall, so the just-
    # added items can show up in the recall sidechannel.
    remembered: dict[str, Any] | None = None
    try:
        if hasattr(agent, "maybe_remember"):
            remembered = await agent.maybe_remember(content, session_id)
            if remembered:
                await websocket.send_text(
                    json.dumps({
                        "type": "memory_added",
                        "session_id": session_id,
                        **remembered,
                    })
                )
    except Exception as exc:
        log.warning("ws.chat.remember_failed", error=str(exc))

    # Run recall + relevance filter ONCE per turn — used both for the WS
    # sidechannel and as the LLM context inside stream_respond.
    recall_ctx = None
    try:
        recall_ctx = await agent.recall_for_turn(content, session_id)
        await websocket.send_text(
            json.dumps({
                "type": "recall",
                "session_id": session_id,
                **_recall_ctx_to_dict(recall_ctx),
            })
        )
    except Exception as exc:
        log.warning("ws.chat.recall_failed", error=str(exc))

    try:
        accumulated: list[str] = []
        stream: AsyncIterator[str] = agent.stream_respond(
            content,
            session_id=session_id,
            remembered=remembered,
            recall_ctx=recall_ctx,
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


def _recall_ctx_to_dict(ctx: Any) -> dict[str, Any]:
    """Serialize a RecallContext for the WS sidechannel."""
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
