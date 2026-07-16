"""Immutable index generations and their atomic current pointer."""

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


@dataclass(frozen=True)
class IndexSnapshot:
    generation_id: str
    document_paths: tuple[Path, ...]


class IndexGenerationStore:
    """Publish complete corpus indexes and resolve one immutable snapshot."""

    _state_lock = threading.RLock()

    def __init__(self, root: Path) -> None:
        self.root = root
        self.generations = root / "generations"
        self.current = root / "current.json"

    def publish(self, documents: Mapping[str, str]) -> IndexSnapshot:
        if not documents:
            raise ValueError("cannot publish an empty index generation")
        ordered = tuple(sorted(documents.items()))
        for name, payload in ordered:
            if Path(name).name != name or not name:
                raise ValueError(f"invalid document name: {name!r}")
            tree = json.loads(payload)
            if not isinstance(tree, list):
                raise ValueError(f"index for {name!r} is not a document tree")
            if not tree or not _tree_has_leaf(tree):
                raise ValueError(f"index for {name!r} has no searchable leaves")

        digest = hashlib.sha256()
        for name, payload in ordered:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload.encode("utf-8"))
            digest.update(b"\0")
        generation_id = digest.hexdigest()[:20]

        self.generations.mkdir(parents=True, exist_ok=True)
        staging = self.generations / f".building-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for name, payload in ordered:
                (staging / f"{name}.json").write_text(payload, encoding="utf-8")
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "generation_id": generation_id,
                        "documents": [name for name, _ in ordered],
                        "created_ns": time.time_ns(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

        with self._state_lock:
            destination = self.generations / generation_id
            if destination.exists():
                shutil.rmtree(staging)
            else:
                os.replace(staging, destination)

            self.root.mkdir(parents=True, exist_ok=True)
            pointer_tmp = self.root / f".current-{uuid.uuid4().hex}.json"
            pointer_tmp.write_text(
                json.dumps({"generation_id": generation_id}, indent=2),
                encoding="utf-8",
            )
            os.replace(pointer_tmp, self.current)
        return self.snapshot()

    def snapshot(self) -> IndexSnapshot:
        with self._state_lock:
            return self._snapshot_unlocked()

    @contextmanager
    def pin_current(self) -> Iterator[IndexSnapshot]:
        with self._state_lock:
            snapshot = self._snapshot_unlocked()
        yield snapshot

    def _snapshot_unlocked(self) -> IndexSnapshot:
        if not self.current.exists():
            return IndexSnapshot(
                generation_id="legacy-flat",
                document_paths=tuple(sorted(self.root.glob("*.json"))),
            )

        pointer = json.loads(self.current.read_text(encoding="utf-8"))
        generation_id = str(pointer["generation_id"])
        generation_dir = self.generations / generation_id
        manifest = json.loads(
            (generation_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("generation_id") != generation_id:
            raise ValueError("index generation manifest does not match current pointer")
        paths = tuple(generation_dir / f"{name}.json" for name in manifest["documents"])
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "current index generation is incomplete: " + ", ".join(missing)
            )
        return IndexSnapshot(generation_id=generation_id, document_paths=paths)


def _tree_has_leaf(nodes: list) -> bool:
    for node in nodes:
        if not isinstance(node, dict):
            return False
        children = node.get("children") or []
        if not children or _tree_has_leaf(children):
            return True
    return False
