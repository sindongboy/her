"""FastAPI + WebSocket presence server for the Samantha-style orb.

CLAUDE.md references:
  §4    Architecture — Agent Core / channels publish; presence server subscribes.
  §2.3  Local-First — server binds to 127.0.0.1 only, never 0.0.0.0.
  §6.3  Presence Channel — observer pattern, subscribe to event bus.

Routes
------
  GET  /            → serves web/index.html
  GET  /static/*    → serves files under web/  (cache-control: no-store for dev)
  WS   /ws          → hello on connect, then forward bus events as JSON
  GET  /healthz     → {"status": "ok", "subscribers": <int>}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from apps.presence.eventbus import Event, EventBus, get_default_bus
except ImportError:  # pragma: no cover — tested with stub
    Event = None  # type: ignore[assignment,misc]
    EventBus = None  # type: ignore[assignment,misc]
    get_default_bus = None  # type: ignore[assignment]

# ── Loopback check (extracted for unit-testability) ───────────────────────────

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _is_loopback(host: str) -> bool:
    """Return True when *host* is a known loopback / test-client address.

    ``testclient`` is accepted so that Starlette's TestClient (which sets
    ``request.client.host`` to ``"testclient"``) can reach the endpoint in
    tests without monkey-patching.
    """
    return host in _LOOPBACK_HOSTS

log = structlog.get_logger(__name__)

_DEFAULT_STATIC = Path(__file__).parent / "web"

# Schema version sent in the hello message.
_SCHEMA_VERSION = 1


def create_app(
    *,
    bus: "EventBus | None" = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    bus:
        Event bus to subscribe to. Defaults to ``apps.presence.get_default_bus()``.
    static_dir:
        Directory from which static files are served. Defaults to the
        ``web/`` subdirectory next to this file.
    """
    if bus is None:
        if get_default_bus is None:
            raise RuntimeError("apps.presence.eventbus not available")
        bus = get_default_bus()

    resolved_static = (static_dir or _DEFAULT_STATIC).resolve()

    app = FastAPI(title="her-presence", docs_url=None, redoc_url=None)

    # ── Static / index ────────────────────────────────────────────────────────

    if resolved_static.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(resolved_static)),
            name="static",
        )

    @app.get("/", include_in_schema=False)
    async def index() -> Any:  # noqa: ANN401
        index_file = resolved_static / "index.html"
        if index_file.exists():
            return FileResponse(
                str(index_file),
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )
        # Fallback placeholder when orb-eng frontend is not yet built.
        return HTMLResponse(
            "<html><body><h1>her presence</h1><p>orb UI not yet deployed.</p></body></html>",
            status_code=200,
        )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> Any:  # noqa: ANN401
        return JSONResponse({"status": "ok", "subscribers": bus.subscriber_count()})

    # ── Remote publish ────────────────────────────────────────────────────────

    @app.post("/publish")
    async def publish_event(request: Request) -> Any:  # noqa: ANN401
        """Accept a published event from a remote channel process.

        Only connections from 127.0.0.1 are accepted (loopback-only, §2.3).
        Validates: ``type`` must be a non-empty string, ``payload`` must be a
        dict, ``ts`` must be a number.  Returns 400 on validation failure and
        403 for non-loopback clients.
        """
        host = request.client.host if request.client else ""
        if not _is_loopback(host):
            log.warning("presence.publish.rejected_non_loopback", host=host)
            return JSONResponse(
                {"error": "forbidden: only loopback connections allowed"},
                status_code=403,
            )

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        # Validate required fields.
        ev_type = body.get("type")
        payload = body.get("payload")
        ts = body.get("ts")

        if not isinstance(ev_type, str) or not ev_type:
            return JSONResponse(
                {"error": "field 'type' must be a non-empty string"},
                status_code=400,
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": "field 'payload' must be a JSON object"},
                status_code=400,
            )
        if not isinstance(ts, (int, float)):
            return JSONResponse(
                {"error": "field 'ts' must be a number"},
                status_code=400,
            )

        try:
            event = Event(type=ev_type, payload=payload, ts=float(ts))  # type: ignore[arg-type]
        except Exception as exc:
            return JSONResponse({"error": f"could not construct event: {exc}"}, status_code=400)

        bus.publish(event)
        log.debug("presence.publish.ok", event_type=ev_type)
        return {"ok": True}

    # ── WebSocket ─────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        log.info("presence.ws.connected")

        # Send hello frame.
        hello = json.dumps(
            {"type": "hello", "payload": {"schema_version": _SCHEMA_VERSION}, "ts": time.monotonic()}
        )
        await websocket.send_text(hello)

        try:
            async for event in bus.subscribe():
                frame = json.dumps(
                    {"type": event.type, "payload": event.payload, "ts": event.ts}
                )
                await websocket.send_text(frame)
        except WebSocketDisconnect:
            log.info("presence.ws.disconnected")
        except Exception:
            log.warning("presence.ws.error")
            await websocket.close()

    return app
