import unittest
from unittest.mock import patch

import numpy as np

from src.embeddings import embed_query, embed_texts

class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        return np.array(
            [[1.0, 0.0] if "cloud" in text.lower() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

class TestEmbeddings(unittest.TestCase):
    @patch("src.embeddings.get_embedding_model", return_value=FakeEmbeddingModel())
    def test_embed_texts_returns_float32_matrix(self, _model):
        result = embed_texts(["Cloud computing", "Machine learning"])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.float32)

    @patch("src.embeddings.get_embedding_model", return_value=FakeEmbeddingModel())
    def test_embed_query_returns_single_row(self, _model):
        result = embed_query("Cloud computing")
        self.assertEqual(result.shape, (1, 2))

    def test_rejects_empty_text_list(self):
        with self.assertRaises(ValueError):
            embed_texts([])

    def test_rejects_blank_query(self):
        with self.assertRaises(ValueError):
            embed_query("   ")

    def test_rejects_blank_items(self):
        with self.assertRaises(ValueError):
            embed_texts(["valid", "   "])

if __name__ == "__main__":
    unittest.main()
