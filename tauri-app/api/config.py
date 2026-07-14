"""Deployment-shape settings, read once at import."""

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


SERVE_UI = _env_flag("ASEPSIS_SERVE_UI", True)
"""Serve the vanilla dev console from this process. Off gives a bare API."""

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ASEPSIS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]
"""Origins allowed to call the API. Empty for a same-origin deployment, where no
CORS middleware is needed at all; the default covers only the Next.js dev server."""
