from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot generate embeddings for empty text.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text item must be a non-empty string.")

    embeddings = get_embedding_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    embedding = get_embedding_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(embedding, dtype=np.float32)
