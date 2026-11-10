# DocuMind code review notes

## Files reviewed

The review covers `app.py`, `requirements.txt`, all modules under `src/`, the retrieval and Gemini utility scripts, and the automated tests.

## Corrections made

- Removed decorative section banners, comment-only clutter, and excess blank lines while retaining useful function documentation and prompt instructions.
- Fixed the retrieval test that was labelled as a no-results case but actually expected one result. It now uses a query whose similarity falls below the configured threshold and expects an empty result list.
- Replaced the API-calling `tests/test_gemini.py` with `scripts/check_gemini_connection.py`. The unittest discovery command now stays offline; make a live Gemini request only by running the smoke-test script explicitly.
- Moved retrieval evaluation into `scripts/evaluate_retrieval.py` and corrected its project-root path so its default storage and dataset paths resolve from the project root.
- Added a configurable Gemini output-token limit. Summaries, larger quizzes, and multi-document comparisons now request more output capacity than ordinary chat answers.
- Corrected the Streamlit processing status text so it no longer reports steps after the helper has already finished them.
- Kept chunk identifiers scoped by document ID so separate PDFs with the same filename/page/chunk positions do not receive colliding IDs.
- Retained persistent-store validation, transactional in-memory updates, document-filtered retrieval, and source-citation checks.

## Validation performed

- Python compilation: passed.
- Python 3.11 grammar check: passed.
- Offline unit tests: 57 passed.
- No live Gemini API calls were made by the unit test suite.

## Validation limitations

The execution environment used for this review did not have Streamlit, FAISS, Sentence Transformers, or the Google GenAI SDK installed. The test suite was run with temporary test-only compatibility stubs for FAISS, Sentence Transformers, and the Gemini SDK; PyMuPDF and NumPy were available. Therefore, the test result validates application logic and mocked module interactions, but it does **not** replace running the full app with your actual installed dependencies. The temporary stubs are not included in this project bundle.

Run the commands in `README.md` inside your Mac virtual environment and test the live Streamlit/Gemini workflow before treating the release as fully integration-tested.


## Phase 9 hardening

- Added environment-validated upload limits for PDF size, page count, batch file count, combined batch size, and generated chunk count.
- Added Streamlit's server-side per-file upload cap and a read-only demo mode that disables uploads and document removal. Demo mode is explicitly not authentication or per-user isolation.
- Added storage-load failure protection so uploads and document removal are paused when the saved library cannot be loaded.
- Changed vector-store persistence to immutable, checksummed snapshots. The `CURRENT` pointer is atomically updated after a snapshot is fully written; previous valid generations are retained and used as recovery candidates.
- Kept backward compatibility for the previous flat-file store layout; the next save migrates it to snapshots.
- Added environment-backed configuration, console logging, major-version dependency bounds, a verified-environment lockfile script, deployment guidance, and tests for upload limits and snapshot recovery.

## Phase 9 validation

- Python compilation: passed.
- 18 direct configuration, upload-validation, and PDF-processing tests: passed with the installed NumPy/PyMuPDF/dotenv packages.
- Full offline suite: 117 tests passed using temporary test-only compatibility stubs for FAISS, Sentence Transformers, and the Google GenAI SDK, because those production packages were not installed in the audit container. These stubs are not included in the project. Run the suite in the project's real `.venv` to complete integration validation.
- No live Gemini API calls were made.
