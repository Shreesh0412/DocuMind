
import json
import os

from pathlib import Path
from datetime import datetime, timezone

import faiss
import numpy as np


class VectorStore:
    """
    Persistent FAISS vector store.

    Responsibilities:
    1. Store document embeddings.
    2. Perform similarity searches.
    3. Save and load the FAISS index.
    4. Maintain document metadata.
    5. Add and remove individual documents.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    ):

        self.index = None

        self.chunks = []

        self.embeddings = np.empty(
            (0, 0),
            dtype=np.float32
        )

        self.documents = {}

        self.embedding_model = embedding_model

    # ---------------------------------------------
    # 1. BASIC DOCUMENT INFORMATION
    # ---------------------------------------------

    def has_document(self, document_id: str) -> bool:

        return document_id in self.documents

    def get_documents(self) -> list[dict]:

        return sorted(
            [
                dict(document)
                for document in self.documents.values()
            ],
            key=lambda document: document["filename"].lower()
        )

    def count(self) -> int:

        if self.index is None:
            return 0

        return self.index.ntotal

    def document_count(self) -> int:

        return len(self.documents)

    # ---------------------------------------------
    # 2. ADD DOCUMENT CHUNKS AND EMBEDDINGS
    # ---------------------------------------------

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: np.ndarray
    ):
        """
        Add new document chunks and their vectors
        to the existing FAISS index.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        vectors = np.array(
            embeddings,
            dtype=np.float32,
            order="C",
            copy=True
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                "Number of chunks and embeddings "
                "must match."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings contain invalid values."
            )

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms == 0):
            raise ValueError(
                "Zero-length embeddings cannot be indexed."
            )

        # Normalize vectors for cosine similarity.
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        # Create the index if this is the first document.
        if self.index is None:

            new_index = faiss.IndexFlatIP(dimension)
            new_index.add(vectors)

            self.index = new_index

        else:

            # Check embedding dimensions.
            if self.index.d != dimension:

                raise ValueError(
                    "Embedding dimensions do not match "
                    "the existing FAISS index."
                )

            self.index.add(vectors)

        # Preserve the embeddings separately.
        if self.embeddings.size == 0:

            self.embeddings = vectors.copy()

        else:

            self.embeddings = np.concatenate(
                [self.embeddings, vectors],
                axis=0
            )

        # Preserve the original chunk metadata.
        self.chunks.extend(
            dict(chunk) for chunk in chunks
        )

    # ---------------------------------------------
    # 3. REGISTER A PROCESSED DOCUMENT
    # ---------------------------------------------


    def register_document(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        file_size: int,
        chunk_size: int = 1000,
        overlap: int = 150
    ):
        """Register document processing information."""

        if self.has_document(document_id):
            raise ValueError(
                "This document is already registered."
            )

        self.documents[document_id] = {

            "document_id": document_id,

            "filename": filename,

            "page_count": page_count,

            "chunk_count": chunk_count,

            "file_size": file_size,

            "chunk_size": chunk_size,

            "overlap": overlap,

            "added_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

    # ---------------------------------------------
    # 4. REMOVE A DOCUMENT
    # ---------------------------------------------

    def remove_document(
        self,
        document_id: str
    ) -> bool:
        """
        Remove a document and all its associated vectors.

        The remaining FAISS index is rebuilt.
        """

        if not self.has_document(document_id):
            return False

        # Find chunk positions belonging to other documents.
        keep_indices = [

            index

            for index, chunk in enumerate(self.chunks)

            if chunk.get("document_id") != document_id
        ]

        # Keep only the corresponding chunks.
        self.chunks = [
            self.chunks[index]
            for index in keep_indices
        ]

        # Keep the matching embeddings.
        if keep_indices:

            self.embeddings = self.embeddings[
                keep_indices
            ].copy()

            # Rebuild the FAISS index.
            dimension = self.embeddings.shape[1]

            new_index = faiss.IndexFlatIP(
                dimension
            )

            new_index.add(
                self.embeddings
            )

            self.index = new_index

        else:

            # No documents remain.
            self.embeddings = np.empty(
                (0, 0),
                dtype=np.float32
            )

            self.index = None

        # Remove document metadata.
        del self.documents[document_id]

        return True

    # ---------------------------------------------
    # 5. SAVE EVERYTHING TO DISK
    # ---------------------------------------------

    def save(self, directory: str | Path):
        """
        Save FAISS index, embeddings and metadata.

        The manifest is written last so that a
        successful manifest indicates a completed save.
        """

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        index_path = directory / "index.faiss"
        embeddings_path = directory / "embeddings.npy"
        chunks_path = directory / "chunks.json"
        documents_path = directory / "documents.json"
        manifest_path = directory / "manifest.json"

        # Temporary file paths.
        index_temp = directory / "index.faiss.tmp"
        embeddings_temp = directory / "embeddings.npy.tmp"
        chunks_temp = directory / "chunks.json.tmp"
        documents_temp = directory / "documents.json.tmp"
        manifest_temp = directory / "manifest.json.tmp"

        # Save FAISS index.
        if self.index is not None:

            faiss.write_index(
                self.index,
                str(index_temp)
            )

        # Save embeddings.
        with open(embeddings_temp, "wb") as file:

            np.save(
                file,
                self.embeddings,
                allow_pickle=False
            )

        # Save chunk metadata.
        with open(chunks_temp, "w", encoding="utf-8") as file:

            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

        # Save registered documents.
        with open(documents_temp, "w", encoding="utf-8") as file:

            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2
            )

        # Create storage manifest.
        manifest = {

            "schema_version": self.SCHEMA_VERSION,

            "embedding_model": self.embedding_model,

            "vector_count": self.count(),

            "chunk_count": len(self.chunks),

            "document_count": len(self.documents),

            "dimension": (
                int(self.embeddings.shape[1])
                if self.embeddings.ndim == 2
                and self.embeddings.shape[0] > 0
                else None
            ),

            "saved_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        with open(manifest_temp, "w", encoding="utf-8") as file:

            json.dump(
                manifest,
                file,
                indent=2
            )

        # Move completed temporary files into place.
        if self.index is not None:

            os.replace(index_temp, index_path)

        else:

            index_path.unlink(missing_ok=True)

        os.replace(
            embeddings_temp,
            embeddings_path
        )

        os.replace(
            chunks_temp,
            chunks_path
        )

        os.replace(
            documents_temp,
            documents_path
        )

        # Write manifest last.
        os.replace(
            manifest_temp,
            manifest_path
        )

    # ---------------------------------------------
    # 6. LOAD EVERYTHING FROM DISK
    # ---------------------------------------------

    def load(
        self,
        directory: str | Path
    ) -> bool:
        """
        Load previously saved data.

        Returns:
            True if a saved store was loaded.
            False if no saved store exists.
        """

        directory = Path(directory)

        manifest_path = directory / "manifest.json"

        if not manifest_path.exists():
            return False

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            manifest = json.load(file)

        if manifest.get("schema_version") != self.SCHEMA_VERSION:

            raise ValueError(
                "Unsupported vector store version."
            )

        if manifest.get("embedding_model") != self.embedding_model:

            raise ValueError(
                "The saved store uses a different embedding model. "
                "Rebuild the document index using the current model."
            )

        # Load chunk metadata.
        with open(
            directory / "chunks.json",
            "r",
            encoding="utf-8"
        ) as file:

            loaded_chunks = json.load(file)

        # Load document metadata.
        with open(
            directory / "documents.json",
            "r",
            encoding="utf-8"
        ) as file:

            loaded_documents = json.load(file)

        # Load numerical embeddings.
        loaded_embeddings = np.load(
            directory / "embeddings.npy",
            allow_pickle=False
        )

        expected_count = manifest["vector_count"]

        if len(loaded_chunks) != expected_count:

            raise ValueError(
                "Chunk count does not match the saved manifest."
            )

        if loaded_embeddings.ndim != 2:

            raise ValueError(
                "Saved embeddings have an invalid shape."
            )

        if loaded_embeddings.shape[0] != expected_count:

            raise ValueError(
                "Embedding count does not match the manifest."
            )

        loaded_index = None

        if expected_count > 0:

            index_path = directory / "index.faiss"

            if not index_path.exists():

                raise FileNotFoundError(
                    "Saved FAISS index is missing."
                )

            loaded_index = faiss.read_index(
                str(index_path)
            )

            if loaded_index.ntotal != expected_count:

                raise ValueError(
                    "FAISS vector count does not match metadata."
                )

            if loaded_index.d != loaded_embeddings.shape[1]:

                raise ValueError(
                    "FAISS dimension does not match saved embeddings."
                )

            if not np.isfinite(loaded_embeddings).all():

                raise ValueError(
                    "Saved embeddings contain invalid values."
                )

        elif loaded_documents:

            raise ValueError(
                "Documents exist but the saved index is empty."
            )

        # Update the current vector store only after validation.
        self.index = loaded_index

        self.chunks = loaded_chunks

        self.embeddings = np.ascontiguousarray(
            loaded_embeddings,
            dtype=np.float32
        )

        self.documents = loaded_documents

        return True

    # ---------------------------------------------
    # 7. SEARCH THE VECTOR DATABASE
    # ---------------------------------------------


    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_score: float | None = None
    ) -> list[dict]:
        """
        Retrieve relevant chunks using cosine similarity.

        min_score:
            Minimum similarity score required.
            None means no threshold filtering.
        """

        if self.index is None:
            raise RuntimeError(
                "The FAISS index has not been built."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if min_score is not None:

            if not np.isfinite(min_score) or not -1 <= min_score <= 1:
                raise ValueError(
                    "min_score must be between -1 and 1."
                )

        query_vector = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError(
                "Query embedding contains invalid values."
            )

        if np.linalg.norm(query_vector) == 0:
            raise ValueError(
                "Query embedding cannot be a zero vector."
            )

        # Normalize query embedding.
        faiss.normalize_L2(query_vector)

        # Retrieve candidates.
        if min_score is not None:
            search_k = self.index.ntotal
        else:
            search_k = min(
                top_k,
                self.index.ntotal
            )

        scores, indices = self.index.search(
            query_vector,
            search_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            # Apply relevance threshold.
            if min_score is not None and score < min_score:
                continue

            result = self.chunks[int(index)].copy()

            result["score"] = float(score)

            results.append(result)

            if len(results) >= top_k:
                break

        return results
