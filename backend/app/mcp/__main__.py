"""CLI entrypoint: ``python -m app.mcp``.

  python -m app.mcp                     # stdio (Claude Code / local)
  python -m app.mcp --http              # streamable HTTP on 127.0.0.1:8765
  python -m app.mcp --http --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse

import anyio

from .server import run_stdio


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.mcp")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve streamable HTTP instead of stdio.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        import uvicorn

        from .server import build_http_app

        uvicorn.run(build_http_app(), host=args.host, port=args.port)
    else:
        anyio.run(run_stdio)


if __name__ == "__main__":
    main()
