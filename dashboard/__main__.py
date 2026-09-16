"""Run the dashboard: ``python -m dashboard``.

    python -m dashboard                    # 127.0.0.1:8787
    python -m dashboard --port 9000
    python -m dashboard --host 0.0.0.0     # expose (read-only routes)

Read-only: every route reads files; the only mutation anywhere is the
in-memory destructive blur applied to served person images.
"""

from __future__ import annotations

import argparse

from .server import DEFAULT_HOST, DEFAULT_PORT, serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    httpd = serve(args.host, args.port)
    store = httpd.RequestHandlerClass.store
    host, port = httpd.server_address[:2]
    print(f"dashboard: http://{host}:{port}/")
    print(f"cameras: {store.cameras['n_cameras']} · queue: {store.queue['n']}"
          f" · failure cases: {store.failures['n']}")
    print("Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
