"""Configuration loader for the Halachic Q&A application."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AppInfo(BaseModel):
    """Application metadata."""

    name: str = "Halachic Q&A"
    version: str = "1.0.0"
    language: str = "he"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "intfloat/multilingual-e5-large"
    device: str = "auto"
    batch_size: int = 32


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    target_tokens: int = 450
    max_tokens: int = 800
    min_tokens: int = 50
    overlap_tokens: int = 50


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    top_k: int = 5
    initial_candidates: int = 20
    min_similarity: float = 0.3
    use_reranker: bool = False


class GenerationConfig(BaseModel):
    """LLM generation configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    temperature: float = 0.2


class StorageConfig(BaseModel):
    """Storage paths configuration."""

    chroma_dir: str = "./db/chroma"
    sqlite_path: str = "./db/app.db"
    books_dir: str = "./data/books"
    processed_dir: str = "./data/processed"


class AppConfig(BaseModel):
    """Root application configuration."""

    app: AppInfo = Field(default_factory=AppInfo)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # API keys loaded from environment
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Fully populated AppConfig instance.
    """
    load_dotenv()

    yaml_data: dict = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            yaml_data = yaml.safe_load(f) or {}

    config = AppConfig(**yaml_data)

    # Override API keys from environment
    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    config.openai_api_key = os.getenv("OPENAI_API_KEY")

    return config
