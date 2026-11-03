
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


# Pretrained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """

    model = SentenceTransformer(MODEL_NAME)

    return model


def embed_texts(
    texts: list[str]
) -> np.ndarray:
    """
    Convert multiple text chunks into embeddings.

    Returns:
        A NumPy array of float32 vectors.
    """

    if not texts:
        raise ValueError(
            "Cannot generate embeddings for empty text."
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )


def embed_query(
    query: str
) -> np.ndarray:
    """
    Convert a user's question into an embedding.
    """

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    model = get_embedding_model()

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.ascontiguousarray(
        embedding,
        dtype=np.float32
    )
