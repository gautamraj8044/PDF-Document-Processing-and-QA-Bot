from __future__ import annotations

from .server import main as server_main


def main(argv: list[str] | None = None) -> int:
    """Backwards-compatible entry point that starts the PDF API server."""

    return server_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
