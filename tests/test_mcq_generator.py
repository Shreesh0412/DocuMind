import unittest
from unittest.mock import patch

from src.advanced_features import (
    _mcq_response_json_schema,
    _extract_json_object,
    generate_mcqs,
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

    def test_mcq_generation_uses_structured_json_and_larger_output_budget(self):
        chunks = [
            {
                "source": "probability.pdf",
                "page": 4,
                "chunk_index": 0,
                "text": "The total probability theorem combines conditional probabilities.",
            }
        ]
        response = (
            '{"questions":[{"question":"Which theorem combines conditional probabilities?",'
            '"options":{"A":"Bayes theorem","B":"Total probability theorem",'
            '"C":"Addition rule","D":"Multiplication rule"},'
            '"correct_answer":"B","explanation":"The excerpt identifies the total probability theorem.",'
            '"source_ids":["S1"]}]}'
        )

        with patch(
            "src.advanced_features._call_gemini",
            return_value=response,
        ) as mock_generation:
            questions, coverage = generate_mcqs(
                chunks,
                "probability.pdf",
                question_count=5,
            )

        self.assertEqual(len(questions), 1)
        self.assertEqual(coverage["generated_questions"], 1)
        self.assertEqual(mock_generation.call_args.kwargs["max_output_tokens"], 8192)
        schema = mock_generation.call_args.kwargs["response_json_schema"]
        self.assertEqual(schema["properties"]["questions"]["maxItems"], 5)
        self.assertIn("response_json_schema", mock_generation.call_args.kwargs)
        self.assertIn("system_instruction", mock_generation.call_args.kwargs)

    def test_empty_questions_response_reports_insufficient_evidence(self):
        chunks = [
            {
                "source": "empty.pdf",
                "page": 1,
                "chunk_index": 0,
                "text": "A short passage with too little useful information.",
            }
        ]
        with patch(
            "src.advanced_features._call_gemini",
            return_value='{"questions": []}',
        ):
            with self.assertRaisesRegex(ValueError, "not contain enough information"):
                generate_mcqs(chunks, "empty.pdf", question_count=5)

    def test_mcq_schema_contains_required_fields_and_four_options(self):
        schema = _mcq_response_json_schema(7)
        item_schema = schema["properties"]["questions"]["items"]
        option_schema = item_schema["properties"]["options"]

        self.assertEqual(option_schema["required"], ["A", "B", "C", "D"])
        self.assertEqual(
            item_schema["properties"]["correct_answer"]["enum"],
            ["A", "B", "C", "D"],
        )
        self.assertEqual(schema["properties"]["questions"]["maxItems"], 7)
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertIs(item_schema["additionalProperties"], False)
        self.assertIs(option_schema["additionalProperties"], False)

if __name__ == "__main__":
    unittest.main()
