import unittest

from src.upload_validation import validate_pdf_upload_batch


class TestPDFUploadValidation(unittest.TestCase):
    def setUp(self):
        self.limits = {
            "max_files": 3,
            "max_file_size_bytes": 10 * 1024 * 1024,
            "max_batch_size_bytes": 20 * 1024 * 1024,
        }

    def test_accepts_batch_within_limits(self):
        total = validate_pdf_upload_batch(
            [("one.pdf", 4_000), ("two.PDF", 6_000)],
            **self.limits,
        )
        self.assertEqual(total, 10_000)

    def test_rejects_too_many_files(self):
        with self.assertRaisesRegex(ValueError, "limit is 3"):
            validate_pdf_upload_batch(
                [(f"{index}.pdf", 100) for index in range(4)],
                **self.limits,
            )

    def test_rejects_non_pdf_filename(self):
        with self.assertRaisesRegex(ValueError, "Only PDF"):
            validate_pdf_upload_batch(
                [("notes.txt", 100)],
                **self.limits,
            )

    def test_rejects_empty_file(self):
        with self.assertRaisesRegex(ValueError, "is empty"):
            validate_pdf_upload_batch(
                [("empty.pdf", 0)],
                **self.limits,
            )

    def test_rejects_file_over_individual_limit(self):
        with self.assertRaisesRegex(ValueError, "per-file limit"):
            validate_pdf_upload_batch(
                [("large.pdf", 10 * 1024 * 1024 + 1)],
                **self.limits,
            )

    def test_rejects_batch_over_combined_limit(self):
        with self.assertRaisesRegex(ValueError, "combined batch limit"):
            validate_pdf_upload_batch(
                [
                    ("one.pdf", 7 * 1024 * 1024),
                    ("two.pdf", 7 * 1024 * 1024),
                    ("three.pdf", 7 * 1024 * 1024),
                ],
                **self.limits,
            )

    def test_rejects_invalid_size_type(self):
        with self.assertRaisesRegex(ValueError, "Invalid file size"):
            validate_pdf_upload_batch(
                [("one.pdf", True)],
                **self.limits,
            )


if __name__ == "__main__":
    unittest.main()
