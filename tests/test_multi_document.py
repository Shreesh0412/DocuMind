import unittest
from unittest.mock import patch

from src.advanced_features import compare_documents

def sample_documents():
    return [
        {
            "document_id": "doc-a",
            "filename": "Cloud.pdf",
            "chunks": [
                {
                    "document_id": "doc-a",
                    "source": "Cloud.pdf",
                    "page": 1,
                    "chunk_index": 1,
                    "text": "Cloud computing provides on-demand access to computing resources.",
                }
            ],
        },
        {
            "document_id": "doc-b",
            "filename": "Virtualization.pdf",
            "chunks": [
                {
                    "document_id": "doc-b",
                    "source": "Virtualization.pdf",
                    "page": 4,
                    "chunk_index": 1,
                    "text": "Virtualization allows multiple virtual machines to run on a physical host.",
                }
            ],
        },
    ]

class TestMultiDocumentComparison(unittest.TestCase):

    @patch(
        "src.advanced_features._call_gemini",
        return_value=(
            "## Similarities\nBoth relate to computing resources [S1] [S2].\n\n"
            "## Differences\nCloud computing focuses on service delivery [S1], "
            "while virtualization describes multiple virtual machines [S2]."
        ),
    )
    def test_comparison_returns_valid_sources(self, _mock_gemini):
        report, sources, coverage = compare_documents(sample_documents())

        self.assertIn("Similarities", report)
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            {source["document_id"] for source in sources},
            {"doc-a", "doc-b"},
        )
        self.assertEqual(coverage["document_count"], 2)

    @patch(
        "src.advanced_features._call_gemini",
        return_value="Only this document is cited [S1].",
    )
    def test_requires_citations_from_two_documents(self, _mock_gemini):
        with self.assertRaisesRegex(ValueError, "only one selected document"):
            compare_documents(sample_documents())

    def test_requires_at_least_two_documents(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            compare_documents(sample_documents()[:1])

    def test_limits_comparison_to_five_documents(self):
        documents = sample_documents()
        documents.extend(
            {
                "document_id": f"doc-{index}",
                "filename": f"Document-{index}.pdf",
                "chunks": [
                    {
                        "document_id": f"doc-{index}",
                        "source": f"Document-{index}.pdf",
                        "page": 1,
                        "chunk_index": 1,
                        "text": f"Sample content {index}.",
                    }
                ],
            }
            for index in range(2, 6)
        )
        with self.assertRaisesRegex(ValueError, "maximum of 5"):
            compare_documents(documents)

    def test_requires_chunks_for_each_document(self):
        documents = sample_documents()
        documents[1]["chunks"] = []
        with self.assertRaisesRegex(ValueError, "No indexed text chunks"):
            compare_documents(documents)

    def test_requires_comparison_focus(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            compare_documents(sample_documents(), comparison_focus="   ")

    @patch(
        "src.advanced_features._call_gemini",
        return_value="INSUFFICIENT_EVIDENCE",
    )
    def test_handles_insufficient_evidence(self, _mock_gemini):
        with self.assertRaisesRegex(ValueError, "not contain enough evidence"):
            compare_documents(sample_documents())

if __name__ == "__main__":
    unittest.main()
