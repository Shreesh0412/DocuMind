
import tempfile
import unittest

import numpy as np

from src.vector_store import VectorStore


class TestPersistentVectorStore(unittest.TestCase):

    def create_sample_store(self):

        store = VectorStore()

        # Sample document chunks.
        chunks = [

            {
                "chunk_id": "a_page1_chunk1",
                "document_id": "document_a",
                "source": "cloud.pdf",
                "page": 1,
                "text": "Cloud computing provides resources."
            },

            {
                "chunk_id": "b_page1_chunk1",
                "document_id": "document_b",
                "source": "ml.pdf",
                "page": 1,
                "text": "Machine learning studies data."
            }

        ]

        # Small sample embeddings.
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            dtype=np.float32
        )

        store.add_chunks(
            chunks,
            embeddings
        )

        store.register_document(
            "document_a",
            "cloud.pdf",
            1,
            1,
            1000
        )

        store.register_document(
            "document_b",
            "ml.pdf",
            1,
            1,
            1200
        )

        return store

    def test_save_and_load(self):

        store = self.create_sample_store()

        with tempfile.TemporaryDirectory() as directory:

            store.save(directory)

            loaded_store = VectorStore()

            result = loaded_store.load(directory)

            self.assertTrue(result)

            self.assertEqual(
                loaded_store.count(),
                2
            )

            self.assertEqual(
                loaded_store.document_count(),
                2
            )

    def test_search_after_loading(self):

        store = self.create_sample_store()

        with tempfile.TemporaryDirectory() as directory:

            store.save(directory)

            loaded_store = VectorStore()

            loaded_store.load(directory)

            query = np.array(
                [[1.0, 0.0]],
                dtype=np.float32
            )

            results = loaded_store.search(
                query,
                top_k=1
            )

            self.assertEqual(
                results[0]["source"],
                "cloud.pdf"
            )

    def test_remove_document(self):

        store = self.create_sample_store()

        removed = store.remove_document(
            "document_a"
        )

        self.assertTrue(removed)

        self.assertEqual(
            store.count(),
            1
        )

        self.assertEqual(
            store.document_count(),
            1
        )

        self.assertFalse(
            store.has_document("document_a")
        )

    def test_removal_persists(self):

        store = self.create_sample_store()

        with tempfile.TemporaryDirectory() as directory:

            store.save(directory)

            store.remove_document("document_a")

            store.save(directory)

            loaded_store = VectorStore()

            loaded_store.load(directory)

            self.assertEqual(
                loaded_store.count(),
                1
            )

            self.assertEqual(
                loaded_store.document_count(),
                1
            )

    def test_duplicate_document_detection(self):

        store = self.create_sample_store()

        self.assertTrue(
            store.has_document("document_a")
        )

        self.assertFalse(
            store.has_document("document_c")
        )


if __name__ == "__main__":
    unittest.main()
