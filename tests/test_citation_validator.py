import unittest

from src.citation_validator import validate_citations

class TestCitationValidator(unittest.TestCase):

    def setUp(self):

        self.available_ids = {
            "S1",
            "S2",
            "S3"
        }

    def test_valid_citations(self):

        answer = (
            "Cloud computing provides resources [S1]. "
            "Virtualization manages VMs [S2]."
        )

        cleaned, valid, invalid = validate_citations(
            answer,
            self.available_ids
        )

        self.assertEqual(
            valid,
            ["S1", "S2"]
        )

        self.assertEqual(
            invalid,
            []
        )

        self.assertEqual(
            cleaned,
            answer
        )

    def test_invalid_citations(self):

        answer = (
            "Cloud computing is useful [S99]."
        )

        cleaned, valid, invalid = validate_citations(
            answer,
            self.available_ids
        )

        self.assertEqual(valid, [])

        self.assertEqual(
            invalid,
            ["S99"]
        )

        self.assertNotIn(
            "[S99]",
            cleaned
        )

    def test_mixed_citations(self):

        answer = (
            "Cloud computing provides resources [S1]. "
            "Another statement [S99]."
        )

        cleaned, valid, invalid = validate_citations(
            answer,
            self.available_ids
        )

        self.assertEqual(
            valid,
            ["S1"]
        )

        self.assertEqual(
            invalid,
            ["S99"]
        )

    def test_duplicate_citations(self):

        answer = "Statement [S1]. More details [S1]."

        cleaned, valid, invalid = validate_citations(
            answer,
            self.available_ids
        )

        self.assertEqual(
            valid,
            ["S1"]
        )

        self.assertEqual(
            invalid,
            []
        )

    def test_no_citations(self):

        answer = "This is an unsupported answer."

        cleaned, valid, invalid = validate_citations(
            answer,
            self.available_ids
        )

        self.assertEqual(valid, [])

        self.assertEqual(invalid, [])

        self.assertEqual(cleaned, answer)

if __name__ == "__main__":
    unittest.main()
