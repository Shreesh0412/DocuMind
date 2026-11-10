"""Small logging configuration suitable for local and hosted Streamlit runs."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging without writing document content or secrets."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
