"""Tests for configuration loading."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from src.config import AppConfig, load_config


class TestAppConfigDefaults:
    """Test that AppConfig provides sensible defaults."""

    def test_default_config_creates_successfully(self) -> None:
        config = AppConfig()
        assert config.app.name == "Halachic Q&A"
        assert config.app.language == "he"

    def test_default_embedding_config(self) -> None:
        config = AppConfig()
        assert config.embedding.model == "intfloat/multilingual-e5-large"
        assert config.embedding.device == "auto"
        assert config.embedding.batch_size == 32

    def test_default_chunking_config(self) -> None:
        config = AppConfig()
        assert config.chunking.target_tokens == 450
        assert config.chunking.max_tokens == 800
        assert config.chunking.min_tokens == 50
        assert config.chunking.overlap_tokens == 50

    def test_default_retrieval_config(self) -> None:
        config = AppConfig()
        assert config.retrieval.top_k == 5
        assert config.retrieval.min_similarity == 0.3
        assert config.retrieval.use_reranker is False

    def test_default_generation_config(self) -> None:
        config = AppConfig()
        assert config.generation.provider == "anthropic"
        assert config.generation.temperature == 0.2

    def test_default_api_keys_are_none(self) -> None:
        config = AppConfig()
        assert config.anthropic_api_key is None
        assert config.openai_api_key is None


class TestLoadConfig:
    """Test loading config from YAML files."""

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_data = {
            "app": {"name": "Test App", "version": "0.1.0"},
            "retrieval": {"top_k": 10},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data))

        config = load_config(config_file)
        assert config.app.name == "Test App"
        assert config.app.version == "0.1.0"
        assert config.retrieval.top_k == 10
        # Other fields keep defaults
        assert config.embedding.model == "intfloat/multilingual-e5-large"

    def test_load_missing_yaml_uses_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.app.name == "Halachic Q&A"

    def test_env_vars_set_api_keys(self, tmp_path: Path, monkeypatch: object) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{}")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")  # type: ignore[attr-defined]
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-456")  # type: ignore[attr-defined]

        config = load_config(config_file)
        assert config.anthropic_api_key == "test-key-123"
        assert config.openai_api_key == "test-openai-456"

    def test_load_project_config_yaml(self) -> None:
        """Test loading the actual project config.yaml."""
        config = load_config("config.yaml")
        assert config.app.name == "Halachic Q&A"
        assert config.storage.sqlite_path == "./db/app.db"
