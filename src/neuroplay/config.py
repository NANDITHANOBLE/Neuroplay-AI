"""
Centralized configuration management for NeuroPlay-AI.
Loads environment variables and exposes typed settings used across
every module (backend, models, frontend, cv_module).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Resolves to the project root (neuroplay-ai/), two levels up from this file
BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_name: str = "NeuroPlay-AI"
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    # Database (fully wired in Phase 3)
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/neuroplay.db")

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"

    # API (fully wired in Phase 17)
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
