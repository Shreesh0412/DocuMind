"""Pure validation helpers for uploaded PDF batch metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def validate_pdf_upload_batch(
    files: Iterable[tuple[str, int]],
    *,
    max_files: int,
    max_file_size_bytes: int,
    max_batch_size_bytes: int,
) -> int:
    """Validate names and sizes before spending CPU or embedding resources.

    Args:
        files: Sequence of ``(filename, size_in_bytes)`` pairs.
        max_files: Maximum number of files accepted in one batch.
        max_file_size_bytes: Maximum size for one PDF.
        max_batch_size_bytes: Maximum combined size for the batch.

    Returns:
        The total batch size in bytes.

    Raises:
        ValueError: When any upload limit or filename rule is violated.
    """

    file_list = list(files)
    if not file_list:
        return 0
    if max_files < 1 or max_file_size_bytes < 1 or max_batch_size_bytes < 1:
        raise ValueError("Upload limits must be positive integers.")
    if len(file_list) > max_files:
        raise ValueError(
            f"Too many files selected. The limit is {max_files} PDFs per batch."
        )

    total_size = 0
    for filename, file_size in file_list:
        safe_name = Path(str(filename)).name
        if not safe_name or Path(safe_name).suffix.casefold() != ".pdf":
            raise ValueError(f"Only PDF files are allowed: {safe_name or 'unnamed file' }.")
        if isinstance(file_size, bool) or not isinstance(file_size, int):
            raise ValueError(f"Invalid file size for {safe_name}.")
        if file_size <= 0:
            raise ValueError(f"{safe_name} is empty.")
        if file_size > max_file_size_bytes:
            limit_mb = max_file_size_bytes / (1024 * 1024)
            raise ValueError(
                f"{safe_name} exceeds the per-file limit of {limit_mb:g} MB."
            )
        total_size += file_size

    if total_size > max_batch_size_bytes:
        limit_mb = max_batch_size_bytes / (1024 * 1024)
        raise ValueError(
            f"The selected PDFs exceed the combined batch limit of {limit_mb:g} MB."
        )
    return total_size
