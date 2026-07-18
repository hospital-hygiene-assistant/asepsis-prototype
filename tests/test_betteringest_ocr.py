import sys
from types import SimpleNamespace

from modules.ingest._betteringest import ocr


def test_title_ocr_loads_only_the_recognition_model(monkeypatch):
    layout = object()

    class Recognition:
        def __init__(self, **kwargs):
            assert kwargs == {
                "model_name": "latin_PP-OCRv5_mobile_rec",
                "device": "cpu",
            }

        def predict(self, _image):
            return [{"rec_text": "  Händehygiene  ", "rec_score": 0.99}]

    monkeypatch.setitem(
        sys.modules,
        "paddlex",
        SimpleNamespace(create_model=lambda name: layout if name == "PP-DocLayoutV3" else None),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=Recognition))
    monkeypatch.setattr(ocr, "_layout", None)
    monkeypatch.setattr(ocr, "_recognizer", None)

    loaded_layout, recognizer = ocr._load_models()

    assert loaded_layout is layout
    assert ocr._recognise(recognizer, object()) == "Händehygiene"
