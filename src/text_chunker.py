def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            minimum_boundary = start + int(chunk_size * 0.60)
            for separator in ("\n\n", "\n", ". ", "? ", "! "):
                position = text.rfind(separator, start, end)
                if position >= minimum_boundary:
                    end = position + len(separator)
                    break
            else:
                position = text.rfind(" ", start, end)
                if position >= minimum_boundary:
                    end = position

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = max(end - overlap, start + 1)
        while next_start < text_length and not text[next_start].isspace():
            next_start += 1
        while next_start < text_length and text[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks

def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    all_chunks = []

    for page_data in pages:
        source = page_data["source"]
        page_number = page_data["page"]
        text = page_data["text"]
        document_id = page_data.get("document_id", source)

        if not isinstance(text, str) or not text.strip():
            continue

        page_chunks = split_text(text, chunk_size, overlap)
        for index, chunk_text in enumerate(page_chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": (
                        f"{str(document_id)[:12]}::page-{page_number}"
                        f"::chunk-{index}"
                    ),
                    "document_id": document_id,
                    "source": source,
                    "page": page_number,
                    "chunk_index": index,
                    "text": chunk_text,
                }
            )

    return all_chunks
