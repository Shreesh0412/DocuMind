"""Environment-backed application settings for DocuMind."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_config_value(name: str, default: str | None = None) -> str | None:
    """Read a setting from environment variables or Streamlit Cloud secrets.

    Environment variables take precedence, preserving local ``.env`` behaviour.
    Streamlit Community Cloud secrets are read through ``st.secrets``; they are
    not assumed to be exported as process environment variables.
    """
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except ModuleNotFoundError:
        # This also keeps the configuration module usable in lightweight tests.
        return default
    except Exception as error:
        # Streamlit raises this when no secrets.toml / cloud secrets are configured.
        if error.__class__.__name__ == "StreamlitSecretNotFoundError":
            return default
        raise


def _read_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = str(get_config_value(name, str(default))).strip()
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _read_bool(name: str, default: bool = False) -> bool:
    raw_value = get_config_value(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false.")


@dataclass(frozen=True)
class Settings:
    """Validated runtime limits and deployment switches."""

    max_pdf_file_size_mb: int = 20
    max_pdf_pages: int = 500
    max_batch_files: int = 5
    max_batch_size_mb: int = 50
    max_chunks_per_batch: int = 5000
    demo_mode: bool = False
    log_level: str = "INFO"

    @property
    def max_pdf_file_size_bytes(self) -> int:
        return self.max_pdf_file_size_mb * 1024 * 1024

    @property
    def max_batch_size_bytes(self) -> int:
        return self.max_batch_size_mb * 1024 * 1024


def get_settings() -> Settings:
    """Read settings from the environment, validating unsafe values early."""

    log_level = str(get_config_value("DOCUMIND_LOG_LEVEL", "INFO")).strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "DOCUMIND_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        )

    settings = Settings(
        max_pdf_file_size_mb=_read_int("MAX_PDF_FILE_SIZE_MB", 20),
        max_pdf_pages=_read_int("MAX_PDF_PAGES", 500),
        max_batch_files=_read_int("MAX_BATCH_FILES", 5),
        max_batch_size_mb=_read_int("MAX_BATCH_SIZE_MB", 50),
        max_chunks_per_batch=_read_int("MAX_CHUNKS_PER_BATCH", 5000),
        demo_mode=_read_bool("DOCUMIND_DEMO_MODE", False),
        log_level=log_level,
    )

    if settings.max_pdf_file_size_mb > settings.max_batch_size_mb:
        raise ValueError(
            "MAX_PDF_FILE_SIZE_MB cannot exceed MAX_BATCH_SIZE_MB."
        )
    return settings
