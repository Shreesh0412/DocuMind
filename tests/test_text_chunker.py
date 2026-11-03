
import unittest

from src.text_chunker import (
    split_text,
    chunk_pages
)


class TestTextChunker(unittest.TestCase):

    def test_text_is_split(self):

        text = (
            "Cloud computing provides resources. "
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

        result = split_text("")

        self.assertEqual(result, [])

    def test_invalid_overlap(self):

        with self.assertRaises(ValueError):

            split_text(
                "Sample text",
                chunk_size=100,
                overlap=100
            )

    def test_metadata_preservation(self):

        pages = [
            {
                "source": "test.pdf",
                "page": 1,
                "text": (
                    "This is a sample document. "
                ) * 100
            }
        ]

        chunks = chunk_pages(
            pages,
            chunk_size=200,
            overlap=40
        )

        self.assertGreater(len(chunks), 1)

        self.assertEqual(
            chunks[0]["source"],
            "test.pdf"
        )

        self.assertEqual(
            chunks[0]["page"],
            1
        )


if __name__ == "__main__":
    unittest.main()
