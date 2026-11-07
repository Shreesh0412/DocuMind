import unittest

from scripts.evaluate_retrieval import evaluate_cases


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, embedding, top_k, min_score):
        self.calls.append(("semantic", embedding, top_k, min_score))
        return [
            {
                "source": "report.pdf",
                "page": 4,
                "text": "Employee CRUD operations.",
                "score": 0.8,
            }
        ][:top_k]

    def search_hybrid(self, embedding, query_text, top_k, min_score):
        self.calls.append(("hybrid", embedding, query_text, top_k, min_score))
        return [
            {
                "source": "report.pdf",
                "page": 4,
                "text": "Employee CRUD operations.",
                "score": 0.8,
                "semantic_score": 0.8,
                "bm25_score": 2.1,
                "hybrid_score": 0.03,
                "semantic_rank": 2,
                "keyword_rank": 1,
            }
        ][:top_k]


class TestHybridEvaluation(unittest.TestCase):
    def test_hybrid_mode_uses_query_text_and_reports_hybrid_scores(self):
        store = FakeStore()
        report = evaluate_cases(
            store,
            [
                {
                    "id": "Q1",
                    "question": "Which employee CRUD operations are supported?",
                    "expected_document": "report.pdf",
                    "expected_pages": [4],
                    "expected_keywords": ["CRUD"],
                }
            ],
            top_k=5,
            query_encoder=lambda question: "encoded-query",
            mode="hybrid",
        )
        self.assertEqual(report["mode"], "hybrid")
        self.assertEqual(report["hit_rate_at_k"], 1.0)
        self.assertEqual(store.calls[0][0], "hybrid")
        self.assertEqual(store.calls[0][2], "Which employee CRUD operations are supported?")
        self.assertEqual(report["results"][0]["retrieved"][0]["bm25_score"], 2.1)

    def test_semantic_mode_remains_the_default_baseline(self):
        store = FakeStore()
        report = evaluate_cases(
            store,
            [
                {
                    "question": "Which employee CRUD operations are supported?",
                    "expected_document": "report.pdf",
                }
            ],
            query_encoder=lambda question: "encoded-query",
        )
        self.assertEqual(report["mode"], "semantic")
        self.assertEqual(store.calls[0][0], "semantic")

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            evaluate_cases(
                FakeStore(),
                [{"question": "Question?", "expected_document": "report.pdf"}],
                query_encoder=lambda question: "encoded-query",
                mode="unknown",
            )


if __name__ == "__main__":
    unittest.main()
