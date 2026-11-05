import unittest

import pymupdf

from src.pdf_processor import extract_pdf_pages

class TestPDFProcessor(unittest.TestCase):
    def make_pdf(self, page_texts: list[str]) -> bytes:
        document = pymupdf.open()
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        data = document.tobytes()
        document.close()
        return data

    def test_extracts_text_and_page_numbers(self):
        pages = extract_pdf_pages(
            self.make_pdf(["First page content", "Second page content"]),
            "sample.pdf",
        )
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page"], 1)
        self.assertEqual(pages[1]["page"], 2)
        self.assertIn("First page content", pages[0]["text"])
        self.assertEqual(pages[0]["source"], "sample.pdf")

    def test_rejects_empty_bytes(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_pdf_pages(b"", "empty.pdf")

    def test_rejects_corrupt_pdf(self):
        with self.assertRaises(ValueError):
            extract_pdf_pages(b"not a PDF", "broken.pdf")

    def test_empty_page_is_preserved(self):
        pages = extract_pdf_pages(
            self.make_pdf(["", "Page with text"]),
            "sample.pdf",
        )
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["text"], "")

if __name__ == "__main__":
    unittest.main()
