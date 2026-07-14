"""Provenance pins.

The PDF ingest module emits fenced ```pin blocks (page, bbox, regions, asset)
into the knowledge base markdown. The viewer renders them from the raw markdown,
but they are metadata here: stripped from anything the model sees, and lifted
onto the node. Documents ingested without pins are untouched.
"""

import json
import re

PIN_FENCE_OPEN = "```pin"
PIN_BLOCK_RE = re.compile(r"^```pin\s*$\n(.*?)^```\s*$\n?", re.MULTILINE | re.DOTALL)
IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)


def _parse_pin_yaml(body: str) -> dict:
    """Minimal parser for the pin blocks' flat YAML: `key: scalar` lines,
    with bracketed values ([..] flow sequences) parsed as JSON."""
    pin: dict = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("["):
            try:
                pin[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        try:
            pin[key] = int(value)
        except ValueError:
            try:
                pin[key] = float(value)
            except ValueError:
                pin[key] = value
    return pin


def _strip_pins(text: str) -> str:
    """Remove pin blocks and standalone image lines from LLM-visible content
    (the viewer still renders them from the raw markdown)."""
    text = PIN_BLOCK_RE.sub("", text)
    text = IMAGE_LINE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
