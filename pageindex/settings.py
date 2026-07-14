"""Runtime-tunable engine settings.

An object rather than module globals: /api/config changes these at runtime, and
a module-level name would only rebind in whichever module did the assignment,
leaving every `from .settings import MODEL` holding the old value.
"""

from dataclasses import dataclass


@dataclass
class Settings:
    model: str = "gemma3:4b"
    """Model used to judge relevance during retrieval."""

    synthesis_model: str = "gemma3:4b"
    """Model used to compose the final answer."""


settings = Settings()
