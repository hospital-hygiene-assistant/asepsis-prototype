"""The Ollama process pool.

Every spawn is mocked — nothing here starts a real `ollama serve`. It was at 21%
because it shells out, but it owns two things worth pinning: which processes this
host actually starts, and which ones it must not kill because it did not start
them.
"""

import subprocess

import pytest
from unittest.mock import MagicMock, patch

from api import ollama_pool
from api.ollama_pool import (
    BASE_PORT,
    ollama_bin,
    set_ollama_instances,
    shutdown_pool,
    start_configured_instances,
)


@pytest.fixture(autouse=True)
def clean_pool(monkeypatch):
    """The pool is process-global; leaking it between tests would be its own bug."""
    monkeypatch.setattr(ollama_pool, "_extra_procs", [])
    monkeypatch.setattr(ollama_pool, "_explainer_procs", [])
    monkeypatch.setattr(ollama_pool._pi, "reconfigure_clients", MagicMock())
    yield


@pytest.fixture
def spawner(monkeypatch):
    """A fake ollama that always starts successfully."""
    monkeypatch.setattr(ollama_pool, "ollama_bin", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(ollama_pool, "_wait_ready", lambda url, timeout=30.0: True)
    popen = MagicMock(spec=subprocess.Popen)
    monkeypatch.setattr(ollama_pool.subprocess, "Popen", MagicMock(return_value=popen))
    return ollama_pool.subprocess.Popen


class TestProbe:
    def test_a_reachable_url_probes_true(self):
        with patch.object(ollama_pool.urllib.request, "urlopen", return_value=MagicMock()):
            assert ollama_pool._probe("http://127.0.0.1:11434") is True

    def test_an_unreachable_url_probes_false_rather_than_raising(self):
        with patch.object(ollama_pool.urllib.request, "urlopen", side_effect=OSError("refused")):
            assert ollama_pool._probe("http://127.0.0.1:11434") is False


class TestProcessorStatus:
    def test_loaded_cpu_models_verify_the_requested_profile(self, monkeypatch):
        response = MagicMock()
        response.read.return_value = (
            b'{"models":[{"name":"qwen","size_vram":0}]}'
        )
        response.__enter__.return_value = response
        monkeypatch.setattr(ollama_pool.urllib.request, "urlopen", MagicMock(return_value=response))
        monkeypatch.setattr(ollama_pool, "CPU_ONLY", True)

        assert ollama_pool.processor_status("http://127.0.0.1:11434") == {
            "cpu_only_requested": True,
            "verified": True,
            "models": [{"name": "qwen", "size_vram": 0}],
        }

    def test_vram_use_is_reported_without_claiming_cpu_verification(self, monkeypatch):
        response = MagicMock()
        response.read.return_value = (
            b'{"models":[{"name":"qwen","size_vram":1024}]}'
        )
        response.__enter__.return_value = response
        monkeypatch.setattr(ollama_pool.urllib.request, "urlopen", MagicMock(return_value=response))
        monkeypatch.setattr(ollama_pool, "CPU_ONLY", True)

        status = ollama_pool.processor_status("http://127.0.0.1:11434")
        assert status["verified"] is False
        assert status["models"][0]["size_vram"] == 1024


class TestOllamaBin:
    def test_it_reports_the_binary_when_installed(self):
        with patch.object(ollama_pool.shutil, "which", return_value="/usr/bin/ollama"):
            assert ollama_bin() == "/usr/bin/ollama"

    def test_it_reports_nothing_when_absent(self):
        with patch.object(ollama_pool.shutil, "which", return_value=None):
            assert ollama_bin() is None


class TestSetInstances:
    def test_one_instance_starts_nothing(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: True)
        result = set_ollama_instances(1)
        spawner.assert_not_called()
        assert result["requested"] == 1

    @pytest.mark.parametrize("count", [-5, 0, 9, 500])
    def test_a_count_outside_the_safe_range_is_refused_before_spawning(
        self, count, spawner, monkeypatch
    ):
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: True)
        with pytest.raises(ValueError, match="between 1 and 8"):
            set_ollama_instances(count)
        spawner.assert_not_called()

    def test_extra_instances_are_started_on_ports_above_the_base(self, spawner, monkeypatch):
        # Nothing is listening above the base port, so they have to be spawned.
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(3)
        assert spawner.call_count == 2
        ports = [call.kwargs["env"]["OLLAMA_HOST"] for call in spawner.call_args_list]
        assert ports == [f"127.0.0.1:{BASE_PORT + 1}", f"127.0.0.1:{BASE_PORT + 2}"]
        for call in spawner.call_args_list:
            assert call.kwargs["env"]["GGML_VK_VISIBLE_DEVICES"] == "-1"
            assert call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"

    def test_a_port_already_serving_is_noted_but_not_adopted(self, spawner, monkeypatch):
        """Someone else's ollama is not ours to manage — least of all to kill."""
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: True)
        set_ollama_instances(2)
        spawner.assert_not_called()
        assert ollama_pool._extra_procs == [None], "held as a placeholder, not a process"

    def test_scaling_down_stops_only_what_we_started(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(3)
        started = list(ollama_pool._extra_procs)
        set_ollama_instances(1)
        assert ollama_pool._extra_procs == []
        for proc in started:
            proc.terminate.assert_called()

    def test_a_hung_instance_is_killed_after_terminate_times_out(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(2)
        proc = ollama_pool._extra_procs[0]
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=4)
        set_ollama_instances(1)
        proc.kill.assert_called_once()

    def test_no_binary_reports_an_error_rather_than_failing_silently(self, monkeypatch):
        monkeypatch.setattr(ollama_pool, "ollama_bin", lambda: None)
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        result = set_ollama_instances(3)
        assert result["errors"], "a pool that could not be built must say so"
        assert "not found in PATH" in result["errors"][0]

    def test_an_instance_that_never_comes_up_is_reported(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        monkeypatch.setattr(ollama_pool, "_wait_ready", lambda url, timeout=30.0: False)
        result = set_ollama_instances(2)
        assert any("did not start in time" in e for e in result["errors"])

    def test_only_responsive_instances_are_handed_to_the_engine(self, spawner, monkeypatch):
        """Reconfiguring onto a dead URL would send retrieval at nothing."""
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(3)
        urls = ollama_pool._pi.reconfigure_clients.call_args.args[0]
        assert urls == [f"http://127.0.0.1:{BASE_PORT}"]

    def test_a_pool_with_nothing_alive_still_falls_back_to_the_base(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: False)
        set_ollama_instances(1)
        urls = ollama_pool._pi.reconfigure_clients.call_args.args[0]
        assert urls == [f"http://127.0.0.1:{BASE_PORT}"]


class TestShutdown:
    def test_it_stops_what_we_started(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(3)
        started = list(ollama_pool._extra_procs)
        shutdown_pool()
        for proc in started:
            proc.terminate.assert_called()
        assert ollama_pool._extra_procs == []

    def test_it_leaves_instances_it_did_not_start_alone(self, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: True)
        set_ollama_instances(2)
        assert ollama_pool._extra_procs == [None]
        shutdown_pool()  # must not raise on the placeholder

    def test_it_can_be_called_twice(self, spawner, monkeypatch):
        # Registered with atexit *and* called from the app lifespan.
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(2)
        shutdown_pool()
        shutdown_pool()

    def test_a_process_that_will_not_die_is_killed(self, spawner, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        set_ollama_instances(2)
        proc = ollama_pool._extra_procs[0]
        proc.terminate.side_effect = OSError("already gone")
        shutdown_pool()
        proc.kill.assert_called_once()


class TestStartConfigured:
    def test_the_default_of_one_starts_nothing(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_INSTANCES", raising=False)
        with patch.object(ollama_pool, "set_ollama_instances") as spy:
            start_configured_instances()
        spy.assert_not_called()

    def test_more_than_one_brings_the_pool_up(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_INSTANCES", "4")
        with patch.object(ollama_pool, "set_ollama_instances") as spy:
            start_configured_instances()
        spy.assert_called_once_with(4)

    def test_an_unsafe_environment_count_is_refused_before_pool_setup(
        self, monkeypatch
    ):
        monkeypatch.setenv("OLLAMA_INSTANCES", "500")
        with patch.object(ollama_pool, "set_ollama_instances") as spy:
            with pytest.raises(ValueError, match="between 1 and 8"):
                start_configured_instances()
        spy.assert_not_called()


class TestEnsureExplainer:
    def test_an_already_running_explainer_is_reused(self, monkeypatch):
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: True)
        assert ollama_pool.ensure_explainer().endswith(str(ollama_pool.EXPLAINER_PORT))

    def test_without_a_binary_it_falls_back_to_the_base_instance(self, monkeypatch):
        monkeypatch.setattr(ollama_pool, "ollama_bin", lambda: None)
        monkeypatch.setattr(ollama_pool, "_probe",
                            lambda url, timeout=2.0: url.endswith(str(BASE_PORT)))
        assert ollama_pool.ensure_explainer() == f"http://127.0.0.1:{BASE_PORT}"

    def test_with_nothing_at_all_it_returns_nothing(self, monkeypatch):
        monkeypatch.setattr(ollama_pool, "ollama_bin", lambda: None)
        monkeypatch.setattr(ollama_pool, "_probe", lambda url, timeout=2.0: False)
        assert ollama_pool.ensure_explainer() is None

    def test_it_spawns_a_dedicated_instance_on_first_use(self, spawner, monkeypatch):
        calls = {"n": 0}

        def probe(url, timeout=2.0):
            # Absent on the first look, up once spawned.
            calls["n"] += 1
            return calls["n"] > 1

        monkeypatch.setattr(ollama_pool, "_probe", probe)
        assert ollama_pool.ensure_explainer().endswith(str(ollama_pool.EXPLAINER_PORT))
        spawner.assert_called_once()
        host = spawner.call_args.kwargs["env"]["OLLAMA_HOST"]
        assert host == f"127.0.0.1:{ollama_pool.EXPLAINER_PORT}"
