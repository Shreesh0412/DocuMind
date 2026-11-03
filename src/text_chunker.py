
def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[str]:
    """
    Split text into smaller overlapping chunks.

    Args:
        text: Original extracted text.
        chunk_size: Maximum approximate chunk size
                    in characters.
        overlap: Approximate overlapping characters.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and "
            "smaller than chunk size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length
        )

        end = target_end

        # Avoid cutting text in the middle of a sentence
        # or paragraph when a suitable boundary exists.
        if target_end < text_length:

            minimum_boundary = (
                start + int(chunk_size * 0.60)
            )

            separators = (
                "\n",
                ". ",
                "? ",
                "! "
            )

            for separator in separators:

                position = text.rfind(
                    separator,
                    start,
                    target_end
                )

                if position >= minimum_boundary:

                    end = position + len(separator)
                    break

            else:
                # If no sentence boundary exists,
                # try to split at a word boundary.
                position = text.rfind(
                    " ",
                    start,
                    target_end
                )

                if position >= minimum_boundary:
                    end = position

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Calculate the next starting point,
        # retaining some previous text.
        next_start = max(
            end - overlap,
            start + 1
        )

        # Avoid starting in the middle of a word.
        candidate = next_start

        while (
            next_start < text_length
            and not text[next_start].isspace()
        ):
            next_start += 1

        if next_start < text_length:

            while (
                next_start < text_length
                and text[next_start].isspace()
            ):
                next_start += 1

        else:
            # Fallback for unusually long words.
            next_start = candidate

        if next_start <= start:
            break

        start = next_start

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[dict]:
    """
    Convert extracted PDF pages into chunks.

    Each chunk preserves:
        - chunk_id
        - source
        - page
        - chunk_index
        - text
    """

    all_chunks = []

    for page_data in pages:

        source = page_data["source"]
        page_number = page_data["page"]
        text = page_data["text"]

        if not text.strip():
            continue

        page_chunks = split_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            chunk = {
                "chunk_id": (
                    f"{source}::page-{page_number}"
                    f"::chunk-{index}"
                ),

                "source": source,

                "page": page_number,

                "chunk_index": index,

                "text": chunk_text
            }

            all_chunks.append(chunk)

    return all_chunks
