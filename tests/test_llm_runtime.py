"""Task-oriented model calls disable open-ended reasoning output."""

from pageindex.llm import _chat


def test_retrieval_verdict_disables_thinking_mode():
    seen = {}

    class Client:
        def chat(self, **kwargs):
            seen.update(kwargs)
            return {"message": {"content": '{"relevant": false}'}}

    assert _chat("verdict", Client(), model="gemma4") == '{"relevant": false}'
    assert seen["think"] is False
