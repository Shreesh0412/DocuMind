import unittest

from scripts.evaluate_retrieval import evaluate_cases, result_matches_case

class FakeStore:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, embedding, top_k, min_score):
        self.queries.append((embedding, top_k, min_score))
        return self.results[:top_k]

class TestRetrievalEvaluation(unittest.TestCase):
    def setUp(self):
        self.results = [
            {
                "source": "cloud.pdf",
                "document_id": "doc-1",
                "page": 3,
                "text": "A hypervisor manages virtual machines.",
                "score": 0.91,
            },
            {
                "source": "ml.pdf",
                "document_id": "doc-2",
                "page": 4,
                "text": "Supervised learning uses labelled data.",
                "score": 0.80,
            },
        ]

    def test_case_match_uses_all_supplied_expectations(self):
        case = {
            "expected_document": "cloud.pdf",
            "expected_pages": [3],
            "expected_keywords": ["hypervisor"],
        }
        self.assertTrue(result_matches_case(self.results[0], case))
        self.assertFalse(result_matches_case(self.results[1], case))

    def test_evaluation_metrics(self):
        store = FakeStore(self.results)
        cases = [
            {"id": "Q1", "question": "What manages VMs?", "expected_document": "cloud.pdf"},
            {"id": "Q2", "question": "What is supervised learning?", "expected_document": "ml.pdf"},
        ]
        report = evaluate_cases(
            store,
            cases,
            top_k=2,
            query_encoder=lambda question: question,
        )
        self.assertEqual(report["hits"], 2)
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertAlmostEqual(report["mrr_at_k"], 0.75)

    def test_rejects_empty_dataset(self):
        with self.assertRaises(ValueError):
            evaluate_cases(FakeStore([]), [], query_encoder=lambda _: None)

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            evaluate_cases(
                FakeStore([]),
                [{"question": " "}],
                query_encoder=lambda _: None,
            )

if __name__ == "__main__":
    unittest.main()
