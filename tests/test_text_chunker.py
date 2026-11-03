
import unittest

from src.text_chunker import (
    split_text,
    chunk_pages
)


class TestTextChunker(unittest.TestCase):

    def test_split_text(self):

        text = (
            "This is a sample sentence. "
        ) * 100

        chunks = split_text(
            text,
            chunk_size=200,
            overlap=40
        )

        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertTrue(chunk.strip())

    def test_empty_text(self):

        chunks = split_text("")

        self.assertEqual(chunks, [])

    def test_preserves_metadata(self):

        pages = [
            {
                "source": "test.pdf",
                "page": 1,
                "text": (
                    "This is a test document. "
                ) * 100
            }
        ]

        chunks = chunk_pages(
            pages,
            chunk_size=200,
            overlap=40
        )

        self.assertGreater(len(chunks), 1)

        first_chunk = chunks[0]

        self.assertEqual(
            first_chunk["source"],
            "test.pdf"
        )

        self.assertEqual(
            first_chunk["page"],
            1
        )

        self.assertIn(
            "chunk_id",
            first_chunk
        )

    def test_invalid_parameters(self):

        with self.assertRaises(ValueError):
            split_text(
                "Some text",
                chunk_size=100,
                overlap=100
            )


if __name__ == "__main__":
    unittest.main()
