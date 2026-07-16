"""The local runtime uses one explicit, reproducible Gemma 4 variant."""

from modules.ingest._captioning import DEFAULT_OLLAMA_MODEL
from pageindex.settings import DEFAULT_MODEL, Settings


def test_default_models_pin_the_quantized_gemma4_e2b_tag():
    settings = Settings()

    assert DEFAULT_MODEL == "gemma4:e2b-it-q4_K_M"
    assert settings.model == DEFAULT_MODEL
    assert settings.synthesis_model == DEFAULT_MODEL
    assert DEFAULT_OLLAMA_MODEL == DEFAULT_MODEL
