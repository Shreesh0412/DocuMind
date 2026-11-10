import unittest

from scripts.evaluate_answer_quality import extract_citation_ids


class GroupedCitationParsingTests(unittest.TestCase):
    def test_single_citation(self):
        self.assertEqual(extract_citation_ids("Claim [S1]."), ["S1"])

    def test_grouped_citations(self):
        self.assertEqual(extract_citation_ids("Claim [S1, S2]."), ["S1", "S2"])

    def test_multiple_groups_and_single_citations(self):
        self.assertEqual(
            extract_citation_ids("A [S1, S2], B [S3], C [S2, S4]."),
            ["S1", "S2", "S3", "S4"],
        )

    def test_whitespace_and_case(self):
        self.assertEqual(extract_citation_ids("Claim [ s1 , s2 ]."), ["S1", "S2"])

    def test_duplicate_ids_are_deduplicated(self):
        self.assertEqual(extract_citation_ids("[S1] then [S1, S2]"), ["S1", "S2"])

    def test_unregistered_id_is_still_extracted_for_validation(self):
        self.assertEqual(extract_citation_ids("Claim [S1, S99]."), ["S1", "S99"])

    def test_no_citations(self):
        self.assertEqual(extract_citation_ids("No sources cited."), [])


if __name__ == '__main__':
    unittest.main()
