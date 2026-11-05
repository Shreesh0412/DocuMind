import unittest
from unittest.mock import patch

from src.advanced_features import summarize_document

class TestDocumentSummarization(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "source": "notes.pdf",
                "page": 2,
                "chunk_index": 1,
                "text": "Cloud computing provides access to computing resources.",
            },
            {
                "source": "notes.pdf",
                "page": 3,
                "chunk_index": 1,
                "text": "Virtualization runs multiple virtual machines on one host.",
            },
        ]

    @patch(
        "src.advanced_features._call_gemini",
        return_value="Cloud computing provides resources [S1]. Virtualization supports VMs [S2].",
    )
    def test_summary_returns_valid_sources_and_coverage(self, _mock_gemini):
        summary, sources, coverage = summarize_document(
            self.chunks,
            "notes.pdf",
        )
        self.assertIn("[S1]", summary)
        self.assertEqual([source["page"] for source in sources], [2, 3])
        self.assertEqual(coverage["used_chunks"], 2)
        self.assertFalse(coverage["sampled"])

    @patch(
        "src.advanced_features._call_gemini",
        return_value="INSUFFICIENT_EVIDENCE",
    )
    def test_summary_rejects_insufficient_evidence(self, _mock_gemini):
        with self.assertRaisesRegex(ValueError, "not contain enough information"):
            summarize_document(self.chunks, "notes.pdf")

    @patch(
        "src.advanced_features._call_gemini",
        return_value="This answer has no citation.",
    )
    def test_summary_requires_valid_citations(self, _mock_gemini):
        with self.assertRaisesRegex(ValueError, "verifiable source citations"):
            summarize_document(self.chunks, "notes.pdf")

if __name__ == "__main__":
    unittest.main()
