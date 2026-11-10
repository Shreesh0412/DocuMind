import unittest

from src.citation_validator import validate_citations


class TestValidateCitations(unittest.TestCase):
    def test_single_valid_citation_is_preserved(self):
        answer, valid, invalid = validate_citations(
            "Supported claim [S1].", {"S1"}
        )
        self.assertEqual(answer, "Supported claim [S1].")
        self.assertEqual(valid, ["S1"])
        self.assertEqual(invalid, [])

    def test_single_invalid_citation_is_removed(self):
        answer, valid, invalid = validate_citations(
            "Unsupported source [S2].", {"S1"}
        )
        self.assertEqual(answer, "Unsupported source.")
        self.assertEqual(valid, [])
        self.assertEqual(invalid, ["S2"])

    def test_group_with_all_valid_ids_is_preserved(self):
        answer, valid, invalid = validate_citations(
            "Claim supported by both [S1, S2].", {"S1", "S2"}
        )
        self.assertEqual(answer, "Claim supported by both [S1, S2].")
        self.assertEqual(valid, ["S1", "S2"])
        self.assertEqual(invalid, [])

    def test_mixed_group_removes_only_invalid_id(self):
        answer, valid, invalid = validate_citations(
            "Claim supported by [S1, S2, S4].", {"S1", "S4"}
        )
        self.assertEqual(answer, "Claim supported by [S1, S4].")
        self.assertEqual(valid, ["S1", "S4"])
        self.assertEqual(invalid, ["S2"])

    def test_group_with_all_invalid_ids_is_removed(self):
        answer, valid, invalid = validate_citations(
            "Claim [S2, S3].", {"S1"}
        )
        self.assertEqual(answer, "Claim.")
        self.assertEqual(valid, [])
        self.assertEqual(invalid, ["S2", "S3"])

    def test_whitespace_and_duplicate_ids_are_normalized(self):
        answer, valid, invalid = validate_citations(
            "Claim [ S1 , S1,  S2 ].", {"S1", "S2"}
        )
        self.assertEqual(answer, "Claim [S1, S2].")
        self.assertEqual(valid, ["S1", "S2"])
        self.assertEqual(invalid, [])

    def test_ids_across_multiple_groups_are_deduplicated_in_results(self):
        answer, valid, invalid = validate_citations(
            "First [S1]. Second [S1, S2].", {"S1"}
        )
        self.assertEqual(answer, "First [S1]. Second [S1].")
        self.assertEqual(valid, ["S1"])
        self.assertEqual(invalid, ["S2"])

    def test_non_string_answer_raises_type_error(self):
        with self.assertRaises(TypeError):
            validate_citations(None, {"S1"})

if __name__ == "__main__":
    unittest.main()
