"""One runtime assembly shared by CLI and HTTP Question answering adapters."""

from collections.abc import Callable

import pageindex
from pageindex.clients import pool
from pageindex.library import ExpectedLibrarySnapshot, ExpectedLibraryStore
from paths import LIBRARY_DIR

from .pdf import rendered_page_size
from .question_answering import (
    AnswerOutcome,
    EvidenceCitationFactory,
    PromptAnswerSynthesizer,
    Question,
    QuestionAnswering,
)
from .retrieval import (
    LibrarySearchResult,
    SearchDiagnostic,
    WholeLibraryRetrieval,
)


class _RunLibrary:
    """Keep the already-verified immutable snapshot local to one run."""

    def __init__(self, store: ExpectedLibraryStore) -> None:
        self._store = store
        self._snapshots: dict[str, ExpectedLibrarySnapshot] = {}

    def open_current(self) -> ExpectedLibrarySnapshot:
        snapshot = self._store.open_current()
        self._snapshots[snapshot.generation_id] = snapshot
        return snapshot

    def open_generation(self, generation_id: str) -> ExpectedLibrarySnapshot:
        snapshot = self._snapshots.get(generation_id)
        if snapshot is None:
            snapshot = self._store.open_generation(generation_id)
            self._snapshots[generation_id] = snapshot
        return snapshot


def build_question_answering(
    run,
    *,
    engine=pageindex,
    library_store: ExpectedLibraryStore | None = None,
    pool_provider: Callable[[], tuple] = pool,
    client_factory=None,
    page_size_reader=rendered_page_size,
) -> QuestionAnswering:
    """Snapshot runtime choices and assemble the single answering module."""
    store = _RunLibrary(library_store or ExpectedLibraryStore(LIBRARY_DIR))
    synthesis_model = str(engine.settings.synthesis_model)
    retrieval, instances = _build_retrieval(
        engine=engine,
        library_store=store,
        pool_provider=pool_provider,
    )
    make_client = client_factory or engine.make_client
    synthesis_client = make_client(instances[0][1])

    def complete(prompt: str) -> str:
        run.begin_synthesis()
        response = synthesis_client.chat(
            model=synthesis_model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0},
        )
        return response["message"]["content"]

    def source(generation_id: str, document_id: str):
        return store.open_generation(generation_id).document(document_id).source

    citations = EvidenceCitationFactory(
        source_href=lambda generation_id, document_id: (
            bound.href
            if (bound := source(generation_id, document_id)) is not None
            else None
        ),
        page_size=lambda generation_id, document_id, page, pin_scale: (
            page_size_reader(bound, page, pin_scale=pin_scale)
            if (bound := source(generation_id, document_id)) is not None
            else None
        ),
    )
    return QuestionAnswering(
        retrieval,
        PromptAnswerSynthesizer(complete),
        citations,
    )


def answer_question(
    question: Question,
    run,
    **runtime_options,
) -> AnswerOutcome:
    """Answer through the shared runtime, including truthful assembly failure."""
    try:
        answering = build_question_answering(run, **runtime_options)
    except Exception:
        diagnostic = SearchDiagnostic(
            document="__runtime__",
            code="question_answering_unavailable",
            message="question answering runtime failed",
        )
        search = LibrarySearchResult(
            query=question.text.strip(),
            generation_id=None,
            documents=(),
            availability_diagnostics=(diagnostic,),
        )
        return AnswerOutcome.retrieval_unavailable(search)
    return answering.answer(question, run)


def build_whole_library_retrieval(
    *,
    engine=pageindex,
    library_store: ExpectedLibraryStore | None = None,
    pool_provider: Callable[[], tuple] = pool,
) -> WholeLibraryRetrieval:
    """Freeze one model and Ollama pool for a whole debug retrieval run."""
    retrieval, _instances = _build_retrieval(
        engine=engine,
        library_store=library_store or ExpectedLibraryStore(LIBRARY_DIR),
        pool_provider=pool_provider,
    )
    return retrieval


def _build_retrieval(*, engine, library_store, pool_provider):
    instances = tuple(pool_provider())
    if not instances:
        raise RuntimeError("Ollama pool is empty")
    return WholeLibraryRetrieval(
        engine,
        library_store=library_store,
        retrieval_model=str(engine.settings.model),
        instances=instances,
    ), instances
