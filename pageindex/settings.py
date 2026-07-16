"""Runtime-tunable engine settings.

An object rather than module globals: /api/config changes these at runtime, and
a module-level name would only rebind in whichever module did the assignment,
leaving every `from .settings import MODEL` holding the old value.
"""

from dataclasses import dataclass

DEFAULT_MODEL = "gemma4:e2b-it-q4_K_M"
MIN_OLLAMA_INSTANCES = 1
MAX_OLLAMA_INSTANCES = 8


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    """Model used to judge relevance during retrieval."""

    synthesis_model: str = DEFAULT_MODEL
    """Model used to compose the final answer."""


settings = Settings()
