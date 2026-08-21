"""
config.py
---------
Centralised configuration for the VisualFind backend.

All tuneable settings are read from environment variables (or a .env file).
Import `settings` wherever you need a config value — never hardcode paths.

Usage:
    from config import settings
    print(settings.api_port)       # 8000
    print(settings.max_images)     # 1600
    print(settings.data_dir)       # Path(...)/backend/data
"""

import os
from pathlib import Path
from typing import List

# Load .env file from project root (if it exists) — must happen before any os.getenv() call
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment variables

# ── Base paths ────────────────────────────────────────────────────────────────
# This file lives in backend/, so parent is the project root.
_BACKEND_DIR  = Path(__file__).parent
_PROJECT_ROOT = _BACKEND_DIR.parent


def _env_list(key: str, default: str) -> List[str]:
    """Parse a comma-separated env var into a list of stripped strings."""
    raw = os.getenv(key, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


# ── Settings dataclass (plain Python — no pydantic-settings required) ─────────

class Settings:
    """
    Application settings loaded from environment variables.

    Attributes mirror the keys in .env.example.
    """

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str
    api_port: int
    log_level: str

    # ── Catalog & index ───────────────────────────────────────────────────────
    max_images: int             # 0 = no limit (use full catalog)
    embedding_batch_size: int

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: List[str]

    # ── Paths (derived, not from env) ─────────────────────────────────────────
    backend_dir:  Path
    project_root: Path
    data_dir:     Path
    images_dir:   Path
    metadata_csv: Path

    def __init__(self) -> None:
        self.api_host            = os.getenv("API_HOST", "0.0.0.0")
        self.api_port            = _env_int("API_PORT", 8000)
        self.log_level           = os.getenv("LOG_LEVEL", "info").lower()
        self.max_images          = _env_int("MAX_IMAGES", 0)
        self.embedding_batch_size = _env_int("EMBEDDING_BATCH_SIZE", 16)
        self.cors_origins        = _env_list(
            "CORS_ORIGINS",
            "http://localhost:8080,http://localhost:3000,http://localhost:5173",
        )

        # Paths
        self.backend_dir  = _BACKEND_DIR
        self.project_root = _PROJECT_ROOT
        self.data_dir     = _BACKEND_DIR / "data"
        self.images_dir   = _PROJECT_ROOT / "ut-zap50k-images" / "ut-zap50k-images"
        self.metadata_csv = _PROJECT_ROOT / "ut-zap50k-data" / "ut-zap50k-data" / "meta-data.csv"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings(host={self.api_host}:{self.api_port}, "
            f"log_level={self.log_level}, max_images={self.max_images})"
        )


# ── Singleton ──────────────────────────────────────────────────────────────────
settings = Settings()
