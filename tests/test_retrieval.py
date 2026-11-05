import unittest

import numpy as np

from src.vector_store import VectorStore

class TestRetrieval(unittest.TestCase):

    def setUp(self):

        self.store = VectorStore()

        self.chunks = [

            {
                "chunk_id": "chunk1",
                "document_id": "doc1",
                "source": "cloud.pdf",
                "page": 1,
                "text": "Cloud computing resources."
            },

            {
                "chunk_id": "chunk2",
                "document_id": "doc2",
                "source": "ml.pdf",
                "page": 1,
                "text": "Machine learning algorithms."
            },

            {
                "chunk_id": "chunk3",
                "document_id": "doc3",
                "source": "os.pdf",
                "page": 1,
                "text": "Operating systems."
            }

        ]

        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0]
            ],
            dtype=np.float32
        )

        self.store.add_chunks(
            self.chunks,
            embeddings
        )

    def test_top_k_retrieval(self):

        query = np.array(
            [[1.0, 0.0]],
            dtype=np.float32
        )

        results = self.store.search(
            query,
            top_k=2
        )

        self.assertEqual(
            len(results),
            2
        )

        self.assertEqual(
            results[0]["source"],
            "cloud.pdf"
        )

    def test_similarity_threshold(self):

        query = np.array(
            [[1.0, 0.0]],
            dtype=np.float32
        )

        results = self.store.search(
            query,
            top_k=5,
            min_score=0.90
        )

        self.assertEqual(
            len(results),
            1
        )

    def test_no_relevant_results(self):
        query = np.array([[1.0, 1.0]], dtype=np.float32)
        results = self.store.search(query, top_k=5, min_score=0.95)
        self.assertEqual(results, [])

    def test_invalid_threshold(self):

        query = np.array(
            [[1.0, 0.0]],
            dtype=np.float32
        )

        with self.assertRaises(ValueError):

            self.store.search(
                query,
                top_k=2,
                min_score=2.0
            )

    def test_search_restricts_selected_documents(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search(
            query,
            top_k=5,
            document_ids={"doc2"},
        )
        self.assertTrue(results)
        self.assertTrue(all(item["document_id"] == "doc2" for item in results))

    def test_empty_document_selection_returns_no_results(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        results = self.store.search(query, top_k=5, document_ids=set())
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()
