import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

class VectorStore:
    SCHEMA_VERSION = 1

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index = None
        self.chunks: list[dict] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.documents: dict[str, dict] = {}
        self.embedding_model = embedding_model
        self._bm25_cache = None

    def has_document(self, document_id: str) -> bool:
        return document_id in self.documents

    def get_documents(self) -> list[dict]:
        return sorted(
            (dict(document) for document in self.documents.values()),
            key=lambda document: str(document.get("filename", "")).casefold(),
        )

    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def document_count(self) -> int:
        return len(self.documents)

    def copy(self) -> "VectorStore":
        self._validate_consistency()
        duplicate = VectorStore(self.embedding_model)
        duplicate.index = (
            faiss.clone_index(self.index)
            if self.index is not None
            else None
        )
        duplicate.chunks = [dict(chunk) for chunk in self.chunks]
        duplicate.embeddings = self.embeddings.copy()
        duplicate.documents = {
            document_id: dict(document)
            for document_id, document in self.documents.items()
        }
        return duplicate

    def _validate_consistency(self) -> None:
        vector_count = self.count()
        if vector_count != len(self.chunks):
            raise ValueError("FAISS index and chunk metadata counts do not match.")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != vector_count:
            raise ValueError("Stored embeddings and FAISS index counts do not match.")
        if vector_count and self.index.d != self.embeddings.shape[1]:
            raise ValueError("Stored embedding dimensions do not match the FAISS index.")

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("No chunks provided.")

        vectors = np.array(embeddings, dtype=np.float32, order="C", copy=True)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks and embeddings must match.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Zero-length embeddings cannot be indexed.")

        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]

        if self.index is not None and self.index.d != dimension:
            raise ValueError("Embedding dimensions do not match the existing FAISS index.")
        if self.embeddings.size and self.embeddings.shape[1] != dimension:
            raise ValueError("Stored embedding dimensions do not match.")

        if self.index is None:
            new_index = faiss.IndexFlatIP(dimension)
        else:
            new_index = faiss.clone_index(self.index)
        new_index.add(vectors)

        combined = (
            vectors.copy()
            if self.embeddings.size == 0
            else np.concatenate((self.embeddings, vectors), axis=0)
        )
        new_chunks = self.chunks + [dict(chunk) for chunk in chunks]

        self.index = new_index
        self.embeddings = np.ascontiguousarray(combined, dtype=np.float32)
        self.chunks = new_chunks
        self._invalidate_bm25_cache()

    def register_document(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        file_size: int,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        if self.has_document(document_id):
            raise ValueError("This document is already registered.")
        if not filename or page_count < 0 or chunk_count <= 0 or file_size < 0:
            raise ValueError("Invalid document metadata.")
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid chunking configuration.")

        self.documents[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "file_size": file_size,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

    def remove_document(self, document_id: str) -> bool:
        if not self.has_document(document_id):
            return False

        keep_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.get("document_id") != document_id
        ]
        new_chunks = [self.chunks[index] for index in keep_indices]

        if keep_indices:
            new_embeddings = np.ascontiguousarray(
                self.embeddings[keep_indices], dtype=np.float32
            )
            new_index = faiss.IndexFlatIP(new_embeddings.shape[1])
            new_index.add(new_embeddings)
        else:
            new_embeddings = np.empty((0, 0), dtype=np.float32)
            new_index = None

        del self.documents[document_id]
        self.chunks = new_chunks
        self.embeddings = new_embeddings
        self.index = new_index
        self._invalidate_bm25_cache()
        return True

    def save(self, directory: str | Path) -> None:
        self._validate_consistency()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "index": directory / "index.faiss",
            "embeddings": directory / "embeddings.npy",
            "chunks": directory / "chunks.json",
            "documents": directory / "documents.json",
            "manifest": directory / "manifest.json",
        }
        temporary = {
            name: path.with_name(path.name + ".tmp")
            for name, path in paths.items()
        }

        try:
            if self.index is not None:
                faiss.write_index(self.index, str(temporary["index"]))

            with open(temporary["embeddings"], "wb") as file:
                np.save(file, self.embeddings, allow_pickle=False)
            with open(temporary["chunks"], "w", encoding="utf-8") as file:
                json.dump(self.chunks, file, ensure_ascii=False, indent=2)
            with open(temporary["documents"], "w", encoding="utf-8") as file:
                json.dump(self.documents, file, ensure_ascii=False, indent=2)

            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "embedding_model": self.embedding_model,
                "vector_count": self.count(),
                "chunk_count": len(self.chunks),
                "document_count": len(self.documents),
                "dimension": int(self.embeddings.shape[1]) if self.count() else None,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(temporary["manifest"], "w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2)

            if self.index is not None:
                os.replace(temporary["index"], paths["index"])
            else:
                paths["index"].unlink(missing_ok=True)
            for name in ("embeddings", "chunks", "documents", "manifest"):
                os.replace(temporary[name], paths[name])
        finally:
            for path in temporary.values():
                path.unlink(missing_ok=True)

    def load(self, directory: str | Path) -> bool:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported vector store version.")
        if manifest.get("embedding_model") != self.embedding_model:
            raise ValueError(
                "The saved store uses a different embedding model. "
                "Rebuild the document index using the current model."
            )

        with open(directory / "chunks.json", "r", encoding="utf-8") as file:
            loaded_chunks = json.load(file)
        with open(directory / "documents.json", "r", encoding="utf-8") as file:
            loaded_documents = json.load(file)
        loaded_embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)

        expected_vectors = int(manifest.get("vector_count", -1))
        if expected_vectors < 0:
            raise ValueError("The saved manifest has an invalid vector count.")
        if len(loaded_chunks) != int(manifest.get("chunk_count", -1)):
            raise ValueError("Chunk count does not match the saved manifest.")
        if len(loaded_chunks) != expected_vectors:
            raise ValueError("Chunk count does not match the vector count.")
        if len(loaded_documents) != int(manifest.get("document_count", -1)):
            raise ValueError("Document count does not match the saved manifest.")
        if loaded_embeddings.ndim != 2 or loaded_embeddings.shape[0] != expected_vectors:
            raise ValueError("Saved embeddings have an invalid shape or count.")
        if not np.isfinite(loaded_embeddings).all():
            raise ValueError("Saved embeddings contain invalid values.")

        loaded_index = None
        if expected_vectors:
            index_path = directory / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError("Saved FAISS index is missing.")
            loaded_index = faiss.read_index(str(index_path))
            if loaded_index.ntotal != expected_vectors:
                raise ValueError("FAISS vector count does not match metadata.")
            if loaded_index.d != loaded_embeddings.shape[1]:
                raise ValueError("FAISS dimension does not match saved embeddings.")
            if manifest.get("dimension") != loaded_index.d:
                raise ValueError("FAISS dimension does not match the saved manifest.")
        elif loaded_documents:
            raise ValueError("Documents exist but the saved index is empty.")

        self.index = loaded_index
        self.chunks = loaded_chunks
        self.embeddings = np.ascontiguousarray(loaded_embeddings, dtype=np.float32)
        self.documents = loaded_documents
        self._invalidate_bm25_cache()
        self._validate_consistency()
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        min_score: float | None = None,
        document_ids: set[str] | None = None,
    ) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("The FAISS index has not been built.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if min_score is not None:
            if not np.isfinite(min_score) or not -1.0 <= min_score <= 1.0:
                raise ValueError("min_score must be between -1 and 1.")
        if document_ids is not None and not document_ids:
            return []

        query_vector = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)
        if query_vector.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match the FAISS index.")
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains invalid values.")
        if np.linalg.norm(query_vector) == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        faiss.normalize_L2(query_vector)
        search_k = (
            self.index.ntotal
            if min_score is not None or document_ids is not None
            else min(top_k, self.index.ntotal)
        )
        scores, indices = self.index.search(query_vector, search_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            if document_ids is not None and chunk.get("document_id") not in document_ids:
                continue
            if min_score is not None and float(score) < min_score:
                continue

            result = chunk.copy()
            result["score"] = float(score)
            results.append(result)
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into useful lexical tokens for BM25."""
        if not isinstance(text, str) or not text.strip():
            return []

        # Split identifiers such as EmployeeManager before punctuation removal.
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "can",
            "could", "did", "do", "does", "for", "from", "how", "i",
            "in", "into", "is", "it", "its", "may", "of", "on", "or",
            "our", "should", "that", "the", "their", "them", "there",
            "these", "this", "those", "to", "user", "was", "were",
            "what", "when", "where", "which", "who", "why", "will",
            "with", "would", "you", "your",
        }
        return [token for token in tokens if token not in stop_words]

    def _invalidate_bm25_cache(self) -> None:
        self._bm25_cache = None

    def _get_bm25_cache(self) -> tuple[list[Counter], list[int]]:
        """Build the lexical term-frequency cache lazily from stored chunks."""
        if self._bm25_cache is None:
            term_frequencies = [
                Counter(self._tokenize(str(chunk.get("text", ""))))
                for chunk in self.chunks
            ]
            document_lengths = [sum(frequencies.values()) for frequencies in term_frequencies]
            self._bm25_cache = (term_frequencies, document_lengths)
        return self._bm25_cache

    def _bm25_rank(
        self,
        query_tokens: list[str],
        eligible_indices: list[int],
        candidate_limit: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[int, float]]:
        """Return (chunk index, BM25 score), highest score first."""
        if not query_tokens or not eligible_indices or candidate_limit <= 0:
            return []

        term_frequencies, document_lengths = self._get_bm25_cache()
        corpus_size = len(eligible_indices)
        average_length = (
            sum(document_lengths[index] for index in eligible_indices) / corpus_size
        ) or 1.0

        document_frequency = Counter()
        for index in eligible_indices:
            document_frequency.update(term_frequencies[index].keys())

        scores: list[tuple[int, float]] = []
        for index in eligible_indices:
            frequencies = term_frequencies[index]
            length = document_lengths[index]
            score = 0.0

            for term in set(query_tokens):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue

                df = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (corpus_size - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * length / average_length
                )
                score += inverse_document_frequency * (
                    frequency * (k1 + 1.0) / denominator
                )

            if score > 0.0:
                scores.append((index, float(score)))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:candidate_limit]

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 4,
        min_score: float | None = None,
        document_ids: set[str] | None = None,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        rrf_k: int = 60,
        candidate_k: int | None = None,
    ) -> list[dict]:
        """Search with FAISS semantic similarity and BM25 keyword matching.

        The two ranked lists are combined with Reciprocal Rank Fusion (RRF).
        ``score`` remains the cosine-similarity value for compatibility with
        existing answer/citation code; ``hybrid_score`` is the fused rank score.

        ``min_score`` filters semantic candidates only. Lexical matches can
        still enter the candidate pool, which is the main benefit of hybrid
        retrieval when an exact term is not represented strongly by embeddings.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("Search query text cannot be empty.")
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("The FAISS index has not been built.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if min_score is not None and (
            not np.isfinite(min_score) or not -1.0 <= min_score <= 1.0
        ):
            raise ValueError("min_score must be between -1 and 1.")
        if document_ids is not None and not document_ids:
            return []
        if (
            not np.isfinite(semantic_weight)
            or not np.isfinite(keyword_weight)
            or semantic_weight < 0
            or keyword_weight < 0
            or semantic_weight + keyword_weight <= 0
        ):
            raise ValueError("Hybrid weights must be non-negative and not both zero.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        query_vector = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)
        if query_vector.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match the FAISS index.")
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains invalid values.")
        if np.linalg.norm(query_vector) == 0:
            raise ValueError("Query embedding cannot be a zero vector.")
        faiss.normalize_L2(query_vector)

        eligible_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if document_ids is None or chunk.get("document_id") in document_ids
        ]
        if not eligible_indices:
            return []

        limit = min(
            len(eligible_indices),
            candidate_k if candidate_k is not None else max(20, top_k * 4),
        )

        # Semantic ranking over the selected-document subset.
        semantic_values = self.embeddings[eligible_indices] @ query_vector[0]
        semantic_pairs = []
        for index, value in zip(eligible_indices, semantic_values):
            score = float(value)
            if min_score is None or score >= min_score:
                semantic_pairs.append((index, score))
        semantic_pairs.sort(key=lambda item: (-item[1], item[0]))
        semantic_pairs = semantic_pairs[:limit]
        semantic_ranks = {
            index: rank for rank, (index, _) in enumerate(semantic_pairs, start=1)
        }
        semantic_score_map = {
            index: score for index, score in zip(eligible_indices, semantic_values)
        }

        # BM25 ranking uses the same selected-document subset.
        query_tokens = self._tokenize(query_text)
        keyword_pairs = self._bm25_rank(
            query_tokens,
            eligible_indices,
            candidate_limit=limit,
        )
        keyword_ranks = {
            index: rank for rank, (index, _) in enumerate(keyword_pairs, start=1)
        }
        keyword_score_map = dict(keyword_pairs)

        candidate_indices = set(semantic_ranks) | set(keyword_ranks)
        if not candidate_indices:
            return []

        fused = []
        for index in candidate_indices:
            semantic_rank = semantic_ranks.get(index)
            keyword_rank = keyword_ranks.get(index)
            hybrid_score = 0.0
            if semantic_rank is not None:
                hybrid_score += semantic_weight / (rrf_k + semantic_rank)
            if keyword_rank is not None:
                hybrid_score += keyword_weight / (rrf_k + keyword_rank)

            fused.append(
                (
                    index,
                    hybrid_score,
                    semantic_score_map.get(index, 0.0),
                    keyword_score_map.get(index, 0.0),
                    semantic_rank,
                    keyword_rank,
                )
            )

        fused.sort(
            key=lambda item: (
                -item[1],
                -(item[2]),
                -(item[3]),
                item[0],
            )
        )

        results = []
        for (
            index,
            hybrid_score,
            semantic_score,
            bm25_score,
            semantic_rank,
            keyword_rank,
        ) in fused[:top_k]:
            result = self.chunks[index].copy()
            result.update(
                {
                    # Keep the old field name for existing source displays.
                    "score": float(semantic_score),
                    "semantic_score": float(semantic_score),
                    "bm25_score": float(bm25_score),
                    "hybrid_score": float(hybrid_score),
                    "semantic_rank": semantic_rank,
                    "keyword_rank": keyword_rank,
                }
            )
            results.append(result)
        return results

