"""PDF text extraction with explicit validation and bounded page processing."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_pdf_pages(
    pdf_bytes: bytes,
    filename: str,
    max_pages: int | None = None,
) -> list[dict]:
    """Extract selectable text from a PDF, optionally enforcing a page limit."""

    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    if Path(filename).suffix.casefold() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    if max_pages is not None and max_pages < 1:
        raise ValueError("The PDF page limit must be a positive integer.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError(f"{Path(filename).name} is password protected.")
            page_count = len(document)
            if max_pages is not None and page_count > max_pages:
                raise ValueError(
                    f"{Path(filename).name} has {page_count} pages; "
                    f"the limit is {max_pages} pages."
                )

            return [
                {
                    "source": Path(filename).name,
                    "page": page_number,
                    "text": page.get_text("text", sort=True).strip(),
                }
                for page_number, page in enumerate(document, start=1)
            ]
    except ValueError:
        raise
    except Exception as error:
        logger.warning(
            "PDF extraction failed for %s (%s).",
            Path(filename).name,
            type(error).__name__,
        )
        raise ValueError(
            f"Could not process {Path(filename).name}: {type(error).__name__}"
        ) from error
