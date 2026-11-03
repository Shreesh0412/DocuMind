
import faiss
import numpy as np


class VectorStore:
    """
    Stores document embeddings and performs
    semantic similarity search using FAISS.
    """

    def __init__(self):

        self.index = None
        self.chunks = []

    def build(
        self,
        chunks: list[dict],
        embeddings: np.ndarray
    ):
        """
        Build a FAISS index from document embeddings.
        """

        if not chunks:
            raise ValueError(
                "No document chunks were provided."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "Number of chunks and embeddings "
                "must be equal."
            )

        # Ensure FAISS receives float32 vectors.
        vectors = np.ascontiguousarray(
            embeddings,
            dtype=np.float32
        )

        # Normalize vectors for cosine similarity.
        faiss.normalize_L2(vectors)

        # Get embedding dimensions.
        dimension = vectors.shape[1]

        # Create FAISS index.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Add all embeddings.
        self.index.add(vectors)

        # Preserve corresponding chunk metadata.
        self.chunks = list(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks.
        """

        if self.index is None:
            raise RuntimeError(
                "Vector index has not been built."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_vector = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension does not "
                "match the document embeddings."
            )

        # Normalize query vector.
        faiss.normalize_L2(query_vector)

        # Search for nearest vectors.
        k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_vector,
            k
        )

        retrieved_chunks = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            result = self.chunks[int(index)].copy()

            result["score"] = float(score)

            retrieved_chunks.append(result)

        return retrieved_chunks

    def count(self):
        """
        Return the total number of stored vectors.
        """

        if self.index is None:
            return 0

        return self.index.ntotal
