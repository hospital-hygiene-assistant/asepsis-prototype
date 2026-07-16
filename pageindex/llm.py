"""Talking to Ollama, and reading what comes back."""

import json
import re
from typing import Optional

import ollama

from .clients import round_robin_client
from .settings import settings

def _chat(
    prompt: str,
    client: Optional[ollama.Client] = None,
    url: str = "",
    model: str | None = None,
) -> str:
    if client is None:
        client, url = round_robin_client()
    response = client.chat(
        model=model or settings.model,
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={"temperature": 0},
    )
    return response["message"]["content"]


def _parse_json_response(raw: str) -> object:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    raw = raw.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data
    except json.JSONDecodeError:
        pass
    for pattern in (r"(\[.*?\])", r"(\{.*?\})"):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model response: {raw[:300]!r}")
