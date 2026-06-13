from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "rag-service"

    # Database
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"

    # Provider selection
    embedding_provider: Literal["openai", "fake"] = "openai"
    llm_provider: Literal["claude", "openai", "fake"] = "claude"

    # Keys (read from env; never hardcode)
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Models
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536
    llm_model: str = "claude-opus-4-8"

    # Retrieval / generation
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    max_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Module-level dimension used by ORM column + migration.
# Reads the env once at import; defaults to 1536 (text-embedding-3-small).
EMBED_DIM: int = get_settings().embed_dim
