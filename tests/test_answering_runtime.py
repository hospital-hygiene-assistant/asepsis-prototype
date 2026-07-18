"""CLI and HTTP share one runtime assembly for Question answering."""

from types import SimpleNamespace

from api.answering_runtime import (
    build_question_answering,
    build_whole_library_retrieval,
)
from api.question_answering import AnswerKind, Question
from pageindex import QuestionRun
from pageindex import DocumentRetrieval, PassageDecision, PassageDecisionKind
from pageindex.library import ExpectedLibraryStore, LibraryCandidate
from pageindex.nodes import PageNode


def test_one_question_snapshots_models_and_pool_once_for_every_document(tmp_path):
    nodes = {
        name: PageNode(
            node_id=name,
            title=name.title(),
            heading_level=1,
            line_idx=0,
            summary="",
            content=f"Verified {name} passage.",
        )
        for name in ("a", "b")
    }
    store = ExpectedLibraryStore(tmp_path / "library")
    store.publish({
        name: LibraryCandidate(
            canonical_markdown=f"# {name}\n\n{node.content}\n",
        )
        for name, node in nodes.items()
    })

    class Engine:
        settings = SimpleNamespace(model="retrieval-a", synthesis_model="synthesis-a")

        def __init__(self):
            self.calls = []

        def search_document(
            self, document, query, state, path, model=None, instances=None
        ):
            self.calls.append((document, model, instances))
            self.settings.model = "retrieval-b"
            self.settings.synthesis_model = "synthesis-b"
            node = nodes[document]
            if document == "a":
                return DocumentRetrieval((PassageDecision(
                    node.node_id,
                    PassageDecisionKind.PASSAGE_RETRIEVED,
                    "exact",
                    node.content,
                    document_id=document,
                ),))
            return DocumentRetrieval(())

    class Client:
        def __init__(self):
            self.models = []
            self.thinking = []

        def chat(self, *, model, messages, options, think):
            self.models.append(model)
            self.thinking.append(think)
            return {"message": {"content": (
                "SHORT_ANSWER: Antwort [1]\n"
                "RECOMMENDED_ACTION: Handlung [1]\n"
                "RATIONALE: Begründung [1]\n"
                "LIMITATIONS: Einschränkung [1]"
            )}}

    engine = Engine()
    client = Client()
    pool_calls = []

    def pool_provider():
        pool_calls.append("snapshot")
        return ((object(), "http://ollama-a"),)

    run = QuestionRun("run-test")
    answering = build_question_answering(
        run,
        engine=engine,
        library_store=store,
        pool_provider=pool_provider,
        client_factory=lambda _url: client,
    )

    outcome = answering.answer(Question("Welche Maßnahmen?"), run)

    assert outcome.kind is AnswerKind.ANSWERED
    assert pool_calls == ["snapshot"]
    assert [call[1] for call in engine.calls] == ["retrieval-a", "retrieval-a"]
    assert engine.calls[0][2] is engine.calls[1][2]
    assert client.models == ["synthesis-a"]
    assert client.thinking == [False]


def test_debug_retrieval_uses_the_same_frozen_runtime_seam(tmp_path):
    store = ExpectedLibraryStore(tmp_path / "library")
    store.publish({"guide": LibraryCandidate("# Guide\n\nVerified text.\n")})

    class Engine:
        settings = SimpleNamespace(model="retrieval-a")

        def __init__(self):
            self.call = None

        def search_document(
            self, document, query, state, path, model=None, instances=None
        ):
            self.call = (model, instances)
            return DocumentRetrieval(())

    engine = Engine()
    instances = ((object(), "http://ollama-a"),)
    pool_calls = []

    retrieval = build_whole_library_retrieval(
        engine=engine,
        library_store=store,
        pool_provider=lambda: pool_calls.append("snapshot") or instances,
    )
    retrieval.search("q", QuestionRun("debug-run"))

    assert pool_calls == ["snapshot"]
    assert engine.call == ("retrieval-a", instances)
