import unittest

from src.advanced_features import (
    _extract_json_object,
    validate_mcq_payload,
)

class TestMCQGeneratorValidation(unittest.TestCase):
    def setUp(self):
        self.source_map = {
            "S1": {
                "id": "S1",
                "source": "probability.pdf",
                "page": 4,
                "score": 0.82,
                "text": "The total probability theorem...",
            },
            "S2": {
                "id": "S2",
                "source": "probability.pdf",
                "page": 5,
                "score": 0.78,
                "text": "A conditional probability...",
            },
        }

    def test_extract_plain_json(self):
        payload = _extract_json_object('{"questions": []}')
        self.assertEqual(payload, {"questions": []})

    def test_extract_fenced_json(self):
        payload = _extract_json_object(
            '```json\n{"questions": []}\n```'
        )
        self.assertEqual(payload, {"questions": []})

    def test_validate_mcq_and_source_mapping(self):
        payload = {
            "questions": [
                {
                    "question": "Which statement is correct?",
                    "options": {
                        "A": "Option one",
                        "B": "Option two",
                        "C": "Option three",
                        "D": "Option four",
                    },
                    "correct_answer": "B",
                    "explanation": "The excerpt supports option two.",
                    "source_ids": ["[S1]", "S99"],
                }
            ]
        }

        result = validate_mcq_payload(payload, self.source_map, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["correct_answer"], "B")
        self.assertEqual(result[0]["source_ids"], ["S1"])
        self.assertEqual(result[0]["sources"][0]["page"], 4)

    def test_reject_question_without_valid_source(self):
        payload = {
            "questions": [
                {
                    "question": "Unsupported question?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_answer": "A",
                    "explanation": "Explanation.",
                    "source_ids": ["S404"],
                }
            ]
        }
        with self.assertRaises(ValueError):
            validate_mcq_payload(payload, self.source_map, 5)

    def test_reject_invalid_correct_answer(self):
        payload = {
            "questions": [
                {
                    "question": "Which option?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_answer": "E",
                    "explanation": "Explanation.",
                    "source_ids": ["S1"],
                }
            ]
        }
        with self.assertRaises(ValueError):
            validate_mcq_payload(payload, self.source_map, 5)

    def test_drop_duplicate_questions(self):
        question = {
            "question": "What is probability?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_answer": "A",
            "explanation": "Supported explanation.",
            "source_ids": ["S1"],
        }
        payload = {"questions": [question, dict(question)]}
        result = validate_mcq_payload(payload, self.source_map, 5)
        self.assertEqual(len(result), 1)

if __name__ == "__main__":
    unittest.main()
