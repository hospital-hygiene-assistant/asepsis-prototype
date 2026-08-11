"""
The knowledge-base manifest: document identity, provenance and tags.

Written by EVERY ingest module. Previously only betteringest_pdf recorded
anything (in .sources.json), so documents ingested as plain markdown had no
metadata at all.

Two problems this solves:

**Identity.** The knowledge base is flat — `knowledge_base/<id>.md`,
`index/<id>.json`, `knowledge_base/assets/<id>/`. As long as ingest only
looked at the top level of one folder, using the filename stem as the id was
survivable. Once ingest recurses, `internal/report.pdf` and `arxiv/report.pdf`
both want to be `report`, and the second silently overwrites the first. The
doc_id is therefore derived from the path RELATIVE to the ingest root, so it is
unique by construction.

**Tags.** Every component of a document's relative folder path becomes a tag,
so `papers/arxiv/2024/x.pdf` is tagged `papers`, `arxiv`, `2024`. That is the
"where did this come from" signal the retrieval side filters on — and a tag
filter skips whole documents before any LLM call, which is the cheapest defence
there is against corpus growth.

Tags live here rather than in `index/*.json` on purpose: re-tagging a corpus
must not require re-indexing it.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = ".manifest.json"
LEGACY_SOURCES_NAME = ".sources.json"

_lock = threading.RLock()

# Characters that are safe in a filename on every platform we target and in a
# URL path segment without escaping.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class DocEntry:
    doc_id: str
    title: str                       # human-readable; the original filename stem
    rel_path: str = ""               # path relative to the ingest root
    source_path: str = ""            # absolute path of the original file
    sha256: str = ""                 # content hash of the source file
    tags: list[str] = field(default_factory=list)        # derived from folders
    manual_tags: list[str] = field(default_factory=list)  # curated; never clobbered
    ingest_module: str = ""
    ingested_at: str = ""
    extra: dict = field(default_factory=dict)   # module-specific (pdf, assets, …)

    @property
    def effective_tags(self) -> list[str]:
        return sorted(set(self.tags) | set(self.manual_tags))


def sanitise(part: str) -> str:
    cleaned = _UNSAFE.sub("_", part).strip("._-")
    return cleaned or "untitled"


def make_doc_id(path: Path, root: Path | None, taken: set[str] | None = None) -> str:
    """Derive a unique document id from a file's path relative to `root`.

    `papers/arxiv/2024/attention.pdf` → `papers__arxiv__2024__attention`

    Two different paths can still sanitise to the same string (e.g. `a b/x` and
    `a-b/x`), so a numeric suffix is appended when needed rather than letting
    one document overwrite another.
    """
    try:
        rel = path.resolve().relative_to(root.resolve()) if root else Path(path.name)
    except (ValueError, OSError):
        rel = Path(path.name)

    parts = [sanitise(p) for p in rel.parent.parts if p not in (".", "/")]
    parts.append(sanitise(rel.stem))
    base = "__".join(p for p in parts if p)

    if taken is None or base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def tags_for(path: Path, root: Path | None) -> list[str]:
    """Folder-derived tags: one per component of the relative parent path.

    Nested folders yield several tags, so a document under `papers/arxiv` is
    reachable by either the broad or the narrow label.
    """
    try:
        rel = path.resolve().relative_to(root.resolve()) if root else Path(path.name)
    except (ValueError, OSError):
        return []
    return [sanitise(p) for p in rel.parent.parts if p not in (".", "/")]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


class Manifest:
    """Reads/writes knowledge_base/.manifest.json."""

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.path = self.kb_dir / MANIFEST_NAME
        self.entries: dict[str, DocEntry] = {}
        self.load()

    # -- persistence --------------------------------------------------------

    def load(self) -> "Manifest":
        with _lock:
            self.entries = {}
            data = {}
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = self._migrate_legacy()
            for doc_id, raw in (data or {}).items():
                fields = {f for f in DocEntry(doc_id="x", title="x").__dict__}
                self.entries[doc_id] = DocEntry(
                    **{k: v for k, v in {**raw, "doc_id": doc_id}.items() if k in fields})
        return self

    def _migrate_legacy(self) -> dict:
        """Read the old .sources.json so an existing library is not orphaned.

        Those entries are keyed by filename stem with no tags — which is
        exactly the pre-recursion layout, so the stem IS the correct id.
        """
        legacy = self.kb_dir / LEGACY_SOURCES_NAME
        try:
            old = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        migrated = {}
        for stem, info in (old or {}).items():
            pdf = info.get("pdf", "")
            migrated[stem] = {
                "title": stem,
                "rel_path": Path(pdf).name if pdf else f"{stem}.md",
                "source_path": pdf,
                "tags": [],
                "manual_tags": [],
                "ingest_module": info.get("module", ""),
                "ingested_at": "",
                "extra": {k: v for k, v in info.items() if k not in ("pdf", "module")},
            }
        return migrated

    def save(self) -> None:
        with _lock:
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            payload = {}
            for doc_id, entry in sorted(self.entries.items()):
                d = asdict(entry)
                d.pop("doc_id", None)
                payload[doc_id] = d
            self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    # -- mutation -----------------------------------------------------------

    def register(self, *, source_path: Path, root: Path | None, ingest_module: str,
                 extra: dict | None = None, doc_id: str | None = None) -> DocEntry:
        """Record a document, deriving its id and folder tags from its path.

        Re-ingesting the same source path reuses its id and PRESERVES any
        manual tags — curation must survive a refresh.
        """
        source_path = Path(source_path)
        with _lock:
            if doc_id is None:
                existing = next((e for e in self.entries.values()
                                 if e.source_path and
                                 Path(e.source_path) == source_path.resolve()), None)
                doc_id = existing.doc_id if existing else make_doc_id(
                    source_path, root, taken=set(self.entries))

            prior = self.entries.get(doc_id)
            try:
                rel = str(source_path.resolve().relative_to(root.resolve())) if root \
                    else source_path.name
            except (ValueError, OSError):
                rel = source_path.name

            entry = DocEntry(
                doc_id=doc_id,
                title=source_path.stem,
                rel_path=rel,
                source_path=str(source_path.resolve()),
                sha256=file_sha256(source_path),
                tags=tags_for(source_path, root),
                manual_tags=list(prior.manual_tags) if prior else [],
                ingest_module=ingest_module,
                ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                extra=extra or {},
            )
            self.entries[doc_id] = entry
            self.save()
            return entry

    def set_manual_tags(self, doc_id: str, tags: list[str]) -> DocEntry | None:
        with _lock:
            entry = self.entries.get(doc_id)
            if entry is None:
                return None
            entry.manual_tags = sorted({sanitise(t) for t in tags if t.strip()})
            self.save()
            return entry

    def remove(self, doc_id: str) -> None:
        with _lock:
            if self.entries.pop(doc_id, None) is not None:
                self.save()

    # -- queries ------------------------------------------------------------

    def get(self, doc_id: str) -> DocEntry | None:
        return self.entries.get(doc_id)

    def tags_of(self, doc_id: str) -> list[str]:
        entry = self.entries.get(doc_id)
        return entry.effective_tags if entry else []

    def all_tags(self) -> list[dict]:
        counts: dict[str, int] = {}
        for entry in self.entries.values():
            for tag in entry.effective_tags:
                counts[tag] = counts.get(tag, 0) + 1
        return [{"tag": t, "doc_count": c} for t, c in sorted(counts.items())]

    def filter_docs(self, tags: list[str] | None) -> set[str]:
        """Doc ids matching ANY of `tags`. Empty/absent selection means all."""
        if not tags:
            return set(self.entries)
        wanted = {sanitise(t) for t in tags}
        return {doc_id for doc_id, entry in self.entries.items()
                if wanted & set(entry.effective_tags)}


def discover_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Every matching file under `root`, recursively, in deterministic order.

    Hidden directories are skipped so caches (.ocr_cache, .git) never get
    ingested as documents.
    """
    root = Path(root)
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        out.append(path)
    return out
