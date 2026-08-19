"""Application configuration.

Centralizes all environment-driven settings using pydantic-settings so every
other module imports a single, typed `settings` object instead of reading
`os.environ` directly. Values are loaded from a `.env` file in development
and from real environment variables in deployment (Docker/Jetson Orin).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Attributes:
        database_url: SQLAlchemy connection string for PostgreSQL.
        environment: Deployment environment name (development/staging/production).
        log_level: Root logging level.
        cors_allow_origins: Comma-separated list of allowed CORS origins,
            parsed into a list. Includes the Vite dev server default
            (http://localhost:5173) so the dashboard can talk to the API
            during local development before it is built and served
            statically by the backend itself.
        api_v1_prefix: URL prefix for all versioned REST endpoints.
        project_name: Human-readable service name, used in OpenAPI docs.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://aegis:aegis@localhost:5432/aegis_v2x"
    environment: str = "development"
    log_level: str = "INFO"
    cors_allow_origins: str = "http://localhost:5173,http://localhost:8000"
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Aegis-V2X Backend"

    @property
    def cors_origins_list(self) -> list[str]:
        """Return `cors_allow_origins` split into a clean list of origins."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance (loaded once per process)."""
    return Settings()


settings = get_settings()
