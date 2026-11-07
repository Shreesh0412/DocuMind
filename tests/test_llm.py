import unittest
from unittest.mock import patch

from src.llm import generate_answer, generate_with_retry


class TestAnswerGeneration(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "source": "cloud.pdf",
                "page": 7,
                "text": "A hypervisor manages virtual machines.",
                "score": 0.91,
            },
            {
                "source": "cloud.pdf",
                "page": 8,
                "text": "Virtualization improves hardware utilization.",
                "score": 0.84,
            },
        ]

    @patch(
        "src.llm.generate_with_retry",
        return_value="A hypervisor manages virtual machines [S1].",
    )
    def test_returns_answer_and_cited_source(self, _mock_generation):
        answer, sources = generate_answer(
            "What manages virtual machines?",
            self.chunks,
        )
        self.assertIn("[S1]", answer)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "cloud.pdf")
        self.assertEqual(sources[0]["page"], 7)

    @patch(
        "src.llm.generate_with_retry",
        return_value="INSUFFICIENT_EVIDENCE",
    )
    def test_handles_insufficient_evidence(self, _mock_generation):
        answer, sources = generate_answer("Unsupported question", self.chunks)
        self.assertIn("couldn't find sufficient information", answer)
        self.assertEqual(sources, [])

    @patch(
        "src.llm.generate_with_retry",
        return_value="This answer has no citation.",
    )
    def test_rejects_answer_without_citation(self, _mock_generation):
        answer, sources = generate_answer("Question", self.chunks)
        self.assertIn("verifiable citations", answer)
        self.assertEqual(sources, [])

    @patch(
        "src.llm.generate_with_retry",
        return_value="Unsupported claim [S99].",
    )
    def test_rejects_only_invalid_citations(self, _mock_generation):
        answer, sources = generate_answer("Question", self.chunks)
        self.assertIn("verifiable citations", answer)
        self.assertEqual(sources, [])

    def test_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            generate_answer("  ", self.chunks)

    def test_rejects_empty_retrieval(self):
        with self.assertRaises(ValueError):
            generate_answer("Question", [])


class TestGeminiRequestConfiguration(unittest.TestCase):
    def test_output_token_limit_is_configurable(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"text": "ok"})()

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        client = FakeClient()
        with patch("src.llm.get_gemini_client", return_value=client):
            result = generate_with_retry("test prompt", max_output_tokens=3200)

        self.assertEqual(result, "ok")
        self.assertEqual(
            client.models.kwargs["config"].max_output_tokens,
            3200,
        )

    def test_structured_json_output_configuration(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"text": '{"questions": []}'})()

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        schema = {
            "type": "object",
            "properties": {"questions": {"type": "array"}},
            "additionalProperties": False,
            "required": ["questions"],
        }
        client = FakeClient()
        with patch("src.llm.get_gemini_client", return_value=client):
            result = generate_with_retry(
                "test prompt",
                max_output_tokens=8192,
                response_json_schema=schema,
                system_instruction="Return JSON only.",
            )

        config = client.models.kwargs["config"]
        self.assertEqual(result, '{"questions": []}')
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, schema)
        config_payload = config.model_dump(by_alias=True, exclude_none=True)
        self.assertIn("responseJsonSchema", config_payload)
        self.assertNotIn("responseSchema", config_payload)
        self.assertEqual(config.system_instruction, "Return JSON only.")

    def test_rejects_response_stopped_by_output_token_limit(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                candidate = type(
                    "Candidate",
                    (),
                    {"finish_reason": "MAX_TOKENS"},
                )()
                return type(
                    "Response",
                    (),
                    {"text": '{"questions": [', "candidates": [candidate]},
                )()

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        with patch("src.llm.get_gemini_client", return_value=FakeClient()):
            with self.assertRaisesRegex(RuntimeError, "output-token limit"):
                generate_with_retry("test prompt", max_output_tokens=100)


if __name__ == "__main__":
    unittest.main()
