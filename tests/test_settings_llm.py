from __future__ import annotations

from pathlib import Path

import pytest

from ragret.quick_qa_agent import quick_qa_llm_configured, set_quick_qa_llm_config
from server.config import Settings, apply_quick_qa_llm, load_settings


class TestLlmSettings:
    def test_reads_ragret_llm_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGRET_LLM_BASE_URL", "https://api.example/v1")
        monkeypatch.setenv("RAGRET_LLM_MODEL", "test-model")
        monkeypatch.setenv("RAGRET_LLM_API_KEY", "secret-key")
        s = Settings()
        assert s.llm_base_url == "https://api.example/v1"
        assert s.llm_model == "test-model"
        assert s.llm_api_key == "secret-key"

    def test_load_settings_from_dotenv(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "RAGRET_HOST=0.0.0.0",
                    "RAGRET_PORT=9999",
                    "RAGRET_LLM_BASE_URL=https://dotenv.local/v1",
                    "RAGRET_LLM_MODEL=dotenv-model",
                    "RAGRET_LLM_API_KEY=dotenv-key",
                ]
            ),
            encoding="utf-8",
        )
        s = load_settings(repo_root=tmp_path)
        assert s.host == "0.0.0.0"
        assert s.port == 9999
        assert s.llm_base_url == "https://dotenv.local/v1"
        assert s.llm_model == "dotenv-model"
        assert s.llm_api_key == "dotenv-key"

    def test_cli_overrides_dotenv(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            "RAGRET_LLM_MODEL=from-env\nRAGRET_LLM_BASE_URL=https://env/v1\n",
            encoding="utf-8",
        )
        s = load_settings(repo_root=tmp_path, llm_model="from-cli")
        assert s.llm_model == "from-cli"
        assert s.llm_base_url == "https://env/v1"

    def test_apply_quick_qa_llm(self) -> None:
        set_quick_qa_llm_config(base_url="", model="", api_key="")
        assert quick_qa_llm_configured() is False
        apply_quick_qa_llm(
            Settings(
                llm_base_url="https://x/v1",
                llm_model="m",
                llm_api_key="k",
            )
        )
        assert quick_qa_llm_configured() is True
        set_quick_qa_llm_config(base_url="", model="", api_key="")
