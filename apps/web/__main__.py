"""Entrypoint for the her web app.

Run with::

    python -m apps.web              # http://127.0.0.1:8765
    python -m apps.web --port 9000  # custom port
    make web

Loopback-only by design (CLAUDE.md §2.3): the server refuses to bind to a
non-loopback host. Set HER_WEB_PORT or pass --port to change the port.
"""

from __future__ import annotations

import argparse
import os
import sys


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _validate_host(host: str) -> None:
    if host not in _LOOPBACK:
        raise SystemExit(
            "her 웹 서버는 로컬 전용이에요. host 는 127.0.0.1 만 허용."
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apps.web",
        description="her web app — text-first chat UI with memory.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (loopback only).")
    default_port = int(os.environ.get("HER_WEB_PORT", "8765"))
    p.add_argument("--port", default=default_port, type=int, help="TCP port (default 8765).")
    p.add_argument(
        "--log-level",
        default="warning",
        help="uvicorn log level (default: warning).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _validate_host(args.host)

    import os as _os
    import signal as _signal

    def _force_exit(_signum: int, _frame: object) -> None:
        print("\n[her-web] 종료합니다.", file=sys.stderr)
        _os._exit(0)

    _signal.signal(_signal.SIGTERM, _force_exit)

    from apps.web.server import create_app
    import uvicorn

    print(
        f"[her-web] http://{args.host}:{args.port} 에서 시작합니다.\n"
        "  브라우저에서 열어 채팅 UI 를 확인하세요.",
        file=sys.stderr,
    )

    app = create_app()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        print("\n[her-web] 종료합니다.", file=sys.stderr)
        _os._exit(0)


if __name__ == "__main__":
    main()
