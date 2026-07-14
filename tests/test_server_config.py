"""Tests for the deployment-shape switches: headless mode and CORS origins.

These import server fresh under a patched environment, because both settings are
read once at import time.
"""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TAURI_APP = Path(__file__).parent.parent / "tauri-app"
sys.path.insert(0, str(TAURI_APP))


def _reload_server(monkeypatch, **env):
    """Import server with a specific environment, isolated from other tests."""
    for key in ("ASEPSIS_SERVE_UI", "ASEPSIS_CORS_ORIGINS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    sys.modules.pop("server", None)


@pytest.fixture
def server_env(monkeypatch):
    def build(**env):
        return next(_reload_server(monkeypatch, **env))

    return build


class TestEnvFlag:
    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", "", "  "])
    def test_falsey_values(self, server_env, raw):
        assert server_env(ASEPSIS_SERVE_UI=raw).SERVE_UI is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "anything"])
    def test_truthy_values(self, server_env, raw):
        assert server_env(ASEPSIS_SERVE_UI=raw).SERVE_UI is True

    def test_defaults_to_serving_the_console(self, server_env):
        # The Tauri launcher relies on this default; don't flip it.
        assert server_env().SERVE_UI is True


class TestHeadlessMode:
    def test_console_routes_are_absent_when_headless(self, server_env):
        module = server_env(ASEPSIS_SERVE_UI="0")
        mounts = [r.path for r in module.app.routes if hasattr(r, "path")]
        assert "/static" not in mounts

    def test_root_reports_disabled_ui_instead_of_crashing(self, server_env):
        module = server_env(ASEPSIS_SERVE_UI="0")
        response = TestClient(module.app).get("/")
        assert response.status_code == 200
        assert response.json()["ui"] == "disabled"

    def test_api_still_works_headless(self, server_env):
        module = server_env(ASEPSIS_SERVE_UI="0")
        assert TestClient(module.app).get("/api/status").status_code == 200

    def test_api_survives_a_missing_ui_directory(self, server_env, monkeypatch):
        """The mount used to have no check_dir=False, so deleting ui/ took the
        entire API down at import. Serving the console must stay optional."""
        module = server_env()
        monkeypatch.setattr(module, "UI_DIR", Path("/nonexistent/ui"))
        client = TestClient(module.app)
        assert client.get("/api/status").status_code == 200
        assert client.get("/").status_code == 404


class TestCorsOrigins:
    def test_defaults_to_the_next_dev_server_only(self, server_env):
        assert server_env().CORS_ORIGINS == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_dev_origin_is_allowed(self, server_env):
        client = TestClient(server_env().app)
        response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_unlisted_origin_gets_no_allow_header(self, server_env):
        client = TestClient(server_env().app)
        response = client.get("/api/status", headers={"Origin": "https://evil.example"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_explicit_origins_are_honoured(self, server_env):
        module = server_env(ASEPSIS_CORS_ORIGINS="https://demo.example, https://other.example")
        assert module.CORS_ORIGINS == ["https://demo.example", "https://other.example"]

    def test_empty_value_disables_cors_entirely(self, server_env):
        """Same-origin deployments want no CORS middleware at all."""
        module = server_env(ASEPSIS_CORS_ORIGINS="")
        assert module.CORS_ORIGINS == []
        client = TestClient(module.app)
        response = client.get("/api/status", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in response.headers

    def test_wildcard_is_still_available_but_opt_in(self, server_env):
        module = server_env(ASEPSIS_CORS_ORIGINS="*")
        client = TestClient(module.app)
        response = client.get("/api/status", headers={"Origin": "https://anywhere.example"})
        assert response.headers.get("access-control-allow-origin") == "*"
