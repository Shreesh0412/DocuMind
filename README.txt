DocuMind citation validator fix

Files in this package:
- src/citation_validator.py: updated citation parser/cleaner
- tests/test_citation_validator_groups.py: 8 focused unit tests

Install from your DocuMind project root (with the virtual environment active):

unzip -o ~/Downloads/Citation_Validator_Fix.zip -d .

Then run the focused tests:

python -m unittest discover -s tests -p 'test_citation_validator_groups.py' -v

Then run the full test suite:

python -m unittest discover -s tests -v

The validator now supports [S1] and [S1, S2]. When a citation group contains
both available and unavailable IDs, it keeps only the available IDs and
returns unavailable IDs in invalid_ids.
