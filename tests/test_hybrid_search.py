import unittest

import numpy as np

from src.vector_store import VectorStore


class TestHybridSearch(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore()
        self.chunks = [
            {
                "chunk_id": "semantic-1",
                "document_id": "doc-a",
                "source": "employees.pdf",
                "page": 1,
                "text": "Employee overview and organization information.",
            },
            {
                "chunk_id": "semantic-2",
                "document_id": "doc-a",
                "source": "employees.pdf",
                "page": 2,
                "text": "Employee records are maintained in the local database.",
            },
            {
                "chunk_id": "keyword-1",
                "document_id": "doc-a",
                "source": "employees.pdf",
                "page": 4,
                "text": "Employee CRUD operations include create, read, update, and delete.",
            },
            {
                "chunk_id": "other-document",
                "document_id": "doc-b",
                "source": "other.pdf",
                "page": 1,
                "text": "Employee CRUD operations and payroll administration.",
            },
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.store.add_chunks(self.chunks, embeddings)

    def test_keyword_match_can_retrieve_below_similarity_threshold(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search_hybrid(
            query,
            "employee CRUD operations create read update delete",
            top_k=3,
            min_score=0.95,
            document_ids={"doc-a"},
        )
        self.assertTrue(results)
        self.assertIn("keyword-1", {item["chunk_id"] for item in results})
        keyword_result = next(item for item in results if item["chunk_id"] == "keyword-1")
        self.assertIsNone(keyword_result["semantic_rank"])
        self.assertIsNotNone(keyword_result["keyword_rank"])
        self.assertGreater(keyword_result["bm25_score"], 0)

    def test_hybrid_returns_both_score_types_and_preserves_score(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search_hybrid(
            query,
            "employee CRUD operations",
            top_k=3,
            min_score=None,
        )
        self.assertTrue(results)
        for item in results:
            self.assertIn("score", item)
            self.assertIn("semantic_score", item)
            self.assertIn("bm25_score", item)
            self.assertIn("hybrid_score", item)
            self.assertAlmostEqual(item["score"], item["semantic_score"])

    def test_selected_document_filter_applies_to_both_retrievers(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search_hybrid(
            query,
            "employee CRUD operations",
            top_k=10,
            document_ids={"doc-b"},
        )
        self.assertTrue(results)
        self.assertTrue(all(item["document_id"] == "doc-b" for item in results))

    def test_empty_document_selection_returns_no_results(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search_hybrid(
            query,
            "employee CRUD operations",
            document_ids=set(),
        )
        self.assertEqual(results, [])

    def test_stopword_only_query_falls_back_to_semantic_search(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search_hybrid(
            query,
            "what is it and how can you do this",
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "semantic-1")
        self.assertEqual(results[0]["bm25_score"], 0.0)

    def test_bm25_cache_is_invalidated_after_adding_chunks(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        self.store.search_hybrid(query, "newterm")
        self.assertIsNotNone(self.store._bm25_cache)

        self.store.add_chunks(
            [
                {
                    "chunk_id": "new-chunk",
                    "document_id": "doc-c",
                    "source": "new.pdf",
                    "page": 1,
                    "text": "newterm is an indexed keyword",
                }
            ],
            np.array([[0.0, 1.0]], dtype=np.float32),
        )
        self.assertIsNone(self.store._bm25_cache)
        results = self.store.search_hybrid(query, "newterm", top_k=4)
        self.assertIn("new-chunk", {item["chunk_id"] for item in results})

    def test_rejects_invalid_weights(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.search_hybrid(
                query,
                "employee",
                semantic_weight=0,
                keyword_weight=0,
            )

    def test_rejects_empty_query_text(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.store.search_hybrid(query, "  ")


if __name__ == "__main__":
    unittest.main()
