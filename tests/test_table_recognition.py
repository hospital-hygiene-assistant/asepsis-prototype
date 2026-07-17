"""Focused table recognition accepts structure and never repairs guesses."""

import sys
from types import SimpleNamespace

import pytest

from modules.ingest import table_recognition
from modules.ingest.table_recognition import TableRecognitionUnavailable


class Pipeline:
    def __init__(self, result):
        self.result = result

    def predict(self, *, input):
        assert input.endswith("table.png")
        return [self.result]


def test_pipeline_is_focused_on_cpu_with_latin_text_recognition(monkeypatch):
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(TableRecognitionPipelineV2=build),
    )
    monkeypatch.setattr(table_recognition, "_pipeline", None)

    table_recognition._load_pipeline()

    assert captured["device"] == "cpu"
    assert captured["text_recognition_model_name"] == "latin_PP-OCRv5_mobile_rec"
    assert captured["use_layout_detection"] is False


def test_nested_paddle_markdown_is_returned_verbatim(tmp_path, monkeypatch):
    image = tmp_path / "table.png"
    image.write_bytes(b"not-read-by-the-fake")
    payload = {"res": {"table_markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}}
    monkeypatch.setattr(table_recognition, "_load_pipeline", lambda: Pipeline(payload))

    result = table_recognition.recognize_table(image)

    assert result.markdown == payload["res"]["table_markdown"]
    assert result.payload == payload


@pytest.mark.parametrize("payload", [{}, {"text": "A B 1 2"}, []])
def test_unstructured_results_fail_closed(tmp_path, monkeypatch, payload):
    image = tmp_path / "table.png"
    image.write_bytes(b"not-read-by-the-fake")
    monkeypatch.setattr(
        table_recognition,
        "_load_pipeline",
        lambda: Pipeline(SimpleNamespace(json=payload)),
    )

    with pytest.raises(TableRecognitionUnavailable, match="structured|structure"):
        table_recognition.recognize_table(image)
