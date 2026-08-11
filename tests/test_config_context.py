"""
Phase 0 — model + context configuration.

The bug this phase exists to prevent: before config.py, no LLM call passed
`num_ctx`, so Ollama silently applied its 4096-token default and truncated
long prompts with no error anywhere. Every test here is a guard on that.
"""
import pytest

import config as app_config
import pageindex

pytestmark = pytest.mark.unit


class TestModelSpecs:
    def test_target_model_is_registered(self):
        spec = app_config.spec_for("gemma4:e4b")
        assert spec.max_ctx == 131072, "gemma4:e4b ceiling should be 128k"
        assert 32768 <= spec.default_ctx <= 65536, (
            "working default must sit in the 32-64k quality band, "
            f"got {spec.default_ctx}")

    def test_default_is_not_the_ceiling(self):
        spec = app_config.spec_for("gemma4:e4b")
        assert spec.default_ctx < spec.max_ctx

    def test_unknown_model_warns_and_falls_back(self, capsys):
        app_config._warned_unknown.discard("not-a-real-model")
        spec = app_config.spec_for("not-a-real-model")
        err = capsys.readouterr().err
        assert "WARNING" in err and "not-a-real-model" in err
        assert spec.default_ctx == app_config.FALLBACK_SPEC.default_ctx
        assert spec.default_ctx > 4096, "must never inherit Ollama's 4096 default"

    def test_unknown_model_warns_only_once(self, capsys):
        app_config._warned_unknown.discard("noisy-model")
        app_config.spec_for("noisy-model")
        capsys.readouterr()
        app_config.spec_for("noisy-model")
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("requested,expected", [
        (0, 49152),          # 0 → spec default
        (1, 4096),           # below floor → clamped up
        (999_999, 131072),   # above ceiling → clamped down
        (32768, 32768),      # in band → untouched
    ])
    def test_ctx_is_clamped(self, requested, expected):
        cfg = app_config.RuntimeConfig(agent_ctx=requested)
        assert cfg.resolved_agent_ctx() == expected


class TestRuntimeConfig:
    def test_update_persists_and_clamps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "RUNTIME_CONFIG_PATH", tmp_path / "rt.json")
        app_config.update(agent_ctx=999_999)
        assert app_config.runtime().agent_ctx == 131072
        assert (tmp_path / "rt.json").exists()

    def test_update_ignores_none_and_unknown_keys(self):
        before = app_config.runtime().agent_ctx
        app_config.update(agent_ctx=None, not_a_field=123)
        assert app_config.runtime().agent_ctx == before

    def test_describe_exposes_the_slider_bounds(self):
        d = app_config.describe()
        assert d["min_ctx"] == 4096
        assert d["agent_ctx_max"] == 131072
        assert d["recommended_ctx_band"] == [32768, 65536]


class TestChatOptions:
    def test_retrieval_and_agent_windows_are_separable(self):
        app_config.update(retrieval_ctx=8192, agent_ctx=65536)
        assert app_config.chat_options("retrieval")["num_ctx"] == 8192
        assert app_config.chat_options("agent")["num_ctx"] == 65536

    def test_temperature_is_zero_by_default(self):
        assert app_config.chat_options("agent")["temperature"] == 0


class TestCallSitesPassNumCtx:
    """The regression net: no LLM call may go out without an explicit window."""

    def test_pageindex_chat_passes_num_ctx(self, fake_ollama):
        pageindex._chat("hello")
        fake_ollama.assert_count(1)
        fake_ollama.assert_every_call_has_num_ctx()
        assert fake_ollama.calls[0]["options"]["num_ctx"] == (
            app_config.runtime().resolved_retrieval_ctx())

    def test_agent_kind_uses_the_agent_window(self, fake_ollama):
        app_config.update(retrieval_ctx=8192, agent_ctx=65536)
        pageindex._chat("hello", kind="agent")
        assert fake_ollama.calls[0]["options"]["num_ctx"] == 65536

    def test_leaf_eval_and_section_check_pass_num_ctx(self, fake_ollama):
        leaf = pageindex.PageNode(node_id="l1", title="Leaf", heading_level=2,
                                  line_idx=0, summary="", content="text")
        section = pageindex.PageNode(node_id="s1", title="Sec", heading_level=1,
                                     line_idx=0, summary="", children=[leaf])
        pageindex._evaluate_leaf(leaf, "q", "doc", "crumb", "parent")
        pageindex._check_section_relevant(section, "q", "crumb")
        fake_ollama.assert_every_call_has_num_ctx()


class TestRequestModelParity:
    """/api/run and /api/chat drive the SAME retrieval, so every field that
    changes what retrieval does must exist on both request models.

    This is not hypothetical: `selected_answers` and `use_cache` were added to
    ChatRequest but missed on RunRequest, and every /api/run call 500'd on the
    missing attribute. Offline tests could not see it because they never
    constructed the request models.
    """

    @staticmethod
    def _models():
        import importlib.util
        from pathlib import Path
        root = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "asepsis_server", root / "tauri-app" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RunRequest, module.ChatRequest

    @pytest.mark.parametrize("field", [
        "tags", "selected_answers", "selection_mode", "use_cache", "index_module",
    ])
    def test_field_present_on_both(self, field):
        run_model, chat_model = self._models()
        assert field in run_model.model_fields, f"RunRequest is missing '{field}'"
        assert field in chat_model.model_fields, f"ChatRequest is missing '{field}'"

    def test_the_endpoints_can_read_what_they_declare(self):
        """Construct each model with defaults and touch every attribute the
        handlers use — an attribute that does not exist raises here."""
        run_model, chat_model = self._models()
        run = run_model(query="q")
        chat = chat_model(query="q")
        for obj in (run, chat):
            for attr in ("tags", "selected_answers", "use_cache"):
                getattr(obj, attr)


class TestThinkingControl:
    """gemma4 reasons before answering. Left on, it spent ~300 tokens working
    out how to summarise one paragraph (15.3s vs 2.3s with it off).

    The trap this guards: capping num_predict WITHOUT disabling thinking
    truncates the model mid-reasoning, so it emits no content at all and every
    summary silently falls back to the heuristic. That is a quiet, total
    quality failure, not a visible error.
    """

    @pytest.mark.parametrize("kind", ["summary", "prune", "facet", "retrieval"])
    def test_recognition_tasks_do_not_think(self, kind):
        assert app_config.think_for(kind) is False

    def test_the_final_answer_still_thinks(self):
        assert app_config.think_for("agent") is True, (
            "reasoning earns its keep on the cited answer, not on routing")

    def test_every_thinking_kind_has_room_to_finish(self):
        """A kind that thinks must not carry a cap so tight that reasoning
        eats the whole budget and leaves nothing for the answer."""
        for kind, thinks in app_config.THINK.items():
            if thinks:
                assert app_config.NUM_PREDICT[kind] >= 1000, (
                    f"'{kind}' thinks but is capped at "
                    f"{app_config.NUM_PREDICT[kind]} tokens — reasoning would "
                    f"consume it before any content is emitted")

    def test_chat_passes_think_through(self, fake_ollama, monkeypatch):
        import pageindex
        seen = {}

        class Recording:
            def chat(self, **kw):
                seen.update(kw)
                return {"message": {"content": "ok"}, "prompt_eval_count": 1}

        monkeypatch.setattr(pageindex, "_round_robin_client",
                            lambda: (Recording(), "u"))
        pageindex._chat("hello", kind="summary")
        assert seen["think"] is False
        pageindex._chat("hello", kind="agent")
        assert seen["think"] is True

    def test_models_without_think_support_still_work(self, monkeypatch):
        """Rejecting `think` must degrade to a plain call, not fail the run."""
        import pageindex
        calls = []

        class Fussy:
            def chat(self, **kw):
                calls.append(kw)
                if "think" in kw:
                    raise ValueError("unknown parameter: think")
                return {"message": {"content": "ok"}, "prompt_eval_count": 1}

        monkeypatch.setattr(pageindex, "_round_robin_client", lambda: (Fussy(), "u"))
        assert pageindex._chat("hello", kind="summary") == "ok"
        assert len(calls) == 2 and "think" not in calls[1]
