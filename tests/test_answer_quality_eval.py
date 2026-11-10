import unittest

from scripts.evaluate_answer_quality import (
    create_template,
    evaluate_dataset,
    evaluate_response,
    is_abstention,
)


class TestAnswerQualityEvaluation(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "Q1",
            "question": "What technologies are used?",
            "answerable": True,
            "expected_evidence": [
                {"source": "Report.pdf", "pages": [2]}
            ],
            "required_concepts": [
                {"label": "Java", "any_of": ["Java 17"]},
                {"label": "SQLite", "any_of": ["SQLite"]},
            ],
        }
        self.response = {
            "id": "Q1",
            "answer": "The application uses Java 17 and SQLite [S1].",
            "sources": [
                {
                    "id": "S1",
                    "source": "Report.pdf",
                    "page": 2,
                    "text": "Developed using Java 17 and SQLite.",
                }
            ],
            "review": {
                "answer_supported": True,
                "citations_support_claims": True,
            },
        }

    def test_answerable_supported_answer_passes_automated_checks(self):
        result = evaluate_response(self.case, self.response)
        self.assertTrue(result["citation_ids_valid"])
        self.assertTrue(result["citation_locations_match_expected"])
        self.assertTrue(result["automated_grounding_pass"])
        self.assertTrue(result["review_complete"])

    def test_invalid_citation_id_fails(self):
        response = dict(self.response)
        response["answer"] = "The application uses Java 17 [S99]."
        result = evaluate_response(self.case, response)
        self.assertFalse(result["citation_ids_valid"])
        self.assertEqual(result["invalid_citation_ids"], ["S99"])
        self.assertFalse(result["automated_grounding_pass"])

    def test_wrong_source_page_fails_evidence_location_check(self):
        response = dict(self.response)
        response["sources"] = [
            {"id": "S1", "source": "Report.pdf", "page": 3, "text": "..."}
        ]
        result = evaluate_response(self.case, response)
        self.assertFalse(result["citation_locations_match_expected"])
        self.assertFalse(result["automated_grounding_pass"])

    def test_all_required_concept_groups_must_be_covered(self):
        response = dict(self.response)
        response["answer"] = "The application uses Java 17 [S1]."
        result = evaluate_response(self.case, response)
        self.assertEqual(result["concept_coverage"], 0.5)
        self.assertFalse(result["automated_grounding_pass"])

    def test_unanswerable_case_requires_abstention_without_sources(self):
        case = {
            "id": "Q2", "answerable": False,
            "expected_evidence": [], "required_concepts": [],
        }
        response = {
            "id": "Q2",
            "answer": "I couldn't find sufficient information in your uploaded documents.",
            "sources": [],
        }
        result = evaluate_response(case, response)
        self.assertTrue(result["correct_abstention"])

    def test_unanswerable_case_flags_answer_as_hallucination_risk(self):
        case = {
            "id": "Q2", "answerable": False,
            "expected_evidence": [], "required_concepts": [],
        }
        response = {"id": "Q2", "answer": "There are 250 employees [S1].", "sources": []}
        result = evaluate_response(case, response)
        self.assertFalse(result["correct_abstention"])

    def test_cross_document_case_requires_citations_from_each_document(self):
        case = {
            "id": "Q3", "answerable": True,
            "expected_evidence": [
                {"source": "Guide.pdf", "pages": [6]},
                {"source": "Report.pdf", "pages": [3, 4]},
            ],
            "required_concepts": [],
        }
        response = {
            "id": "Q3",
            "answer": "The guide asks for a problem statement [S1], which the report provides [S2].",
            "sources": [
                {"id": "S1", "source": "Guide.pdf", "page": 6},
                {"id": "S2", "source": "Report.pdf", "page": 3},
            ],
        }
        result = evaluate_response(case, response)
        self.assertTrue(result["citation_locations_match_expected"])

    def test_template_contains_every_dataset_case(self):
        dataset = {"cases": [{"id": "Q1"}, {"id": "Q2"}]}
        template = create_template(dataset)
        self.assertEqual([item["id"] for item in template["responses"]], ["Q1", "Q2"])

    def test_abstention_detection(self):
        self.assertTrue(is_abstention("INSUFFICIENT_EVIDENCE"))
        self.assertTrue(is_abstention("This is not specified in the provided documents."))
        self.assertFalse(is_abstention("The system uses SQLite."))

    def test_dataset_report_tracks_missing_responses(self):
        dataset = {"cases": [self.case, {"id": "Q2", "answerable": False}]}
        report = evaluate_dataset(dataset, {"responses": [self.response]})
        self.assertEqual(report["responses_entered"], 1)
        self.assertEqual(report["missing_response_ids"], ["Q2"])
        self.assertEqual(report["valid_citation_id_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
