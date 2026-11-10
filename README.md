# DocuMind

AI-powered PDF question answering and document analysis using local embeddings, FAISS, and Google Gemini.

## Features

- PDF text extraction with page metadata
- Overlapping text chunking
- Local Sentence Transformers embeddings
- Persistent FAISS similarity search
- Gemini-grounded answers with validated source IDs
- Document summarization
- MCQ generation and quiz scoring
- Multi-document comparison
- Local document library and document-specific search
- Retrieval evaluation with Hit Rate@K and MRR
- Configurable PDF size, page, batch, and chunk limits
- Read-only demo mode for curated public sample documents
- Checksummed, versioned vector-store snapshots with recovery fallback
- Dedicated Chat, Summarize, Quiz, and Compare workspaces
- Sidebar document library, search scope, and collapsible retrieval tuning

## Requirements

- Python 3.11 or newer
- A Google Gemini API key
- Internet access for the initial embedding-model download and Gemini requests

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`. Do not commit `.env` or `data/documind_store/`.

## Run

```bash
streamlit run app.py
```

## Tests

Run the offline unit tests. They do not make live Gemini API calls:

```bash
python -m unittest discover -s tests -v
```

The test suite runs offline and does not make live Gemini API calls. To make a deliberate Gemini connectivity request, run this script separately:

```bash
python -m scripts.check_gemini_connection
```

Do not put live API calls in test modules; unittest discovery imports test files automatically.

## Retrieval evaluation

Copy the example dataset, then edit its expected filenames, pages, and keywords to match your PDFs:

```bash
cp tests/retrieval_eval_set.example.json tests/retrieval_eval_set.json
python -m scripts.evaluate_retrieval --dataset tests/retrieval_eval_set.json --top-k 5
```

The report contains Hit Rate@K and Mean Reciprocal Rank (MRR). It depends on a human-labelled evaluation set; it is not an automatic measure of answer correctness.

## Upload limits and demo mode

Default upload limits are 20 MB per PDF, 500 pages per PDF, 5 PDFs per batch,
50 MB combined per batch, and 5,000 chunks per batch. Configure these with the
`MAX_PDF_FILE_SIZE_MB`, `MAX_PDF_PAGES`, `MAX_BATCH_FILES`,
`MAX_BATCH_SIZE_MB`, and `MAX_CHUNKS_PER_BATCH` environment variables.

Set `DOCUMIND_DEMO_MODE=true` for a public read-only demo. This disables PDF
uploads and document removal. It is not authentication: every visitor can query
the same library, so deploy only curated, non-sensitive sample documents.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for hosting, secrets, persistence,
backup, and dependency-lock instructions.

## Workspaces and interface

DocuMind uses a compact workspace selector to keep the main tasks separate:

- **Chat:** ask questions about the selected PDFs and inspect retrieved source passages.
- **Summarize:** create and download a cited document summary.
- **Quiz:** generate, attempt, and review source-grounded MCQs.
- **Compare:** compare two or more selected PDFs and download the report.

Manage uploads and the document library in the sidebar. Select the documents
to search under **Search scope**. Advanced retrieval tuning is kept in a
collapsed expander so the common workflow stays uncluttered.

See [docs/UI_GUIDE.md](docs/UI_GUIDE.md) for a quick walkthrough.

## Local data

DocuMind saves FAISS, embeddings, chunks, and document metadata in
`data/documind_store/`. New saves create immutable, checksummed generations in
`data/documind_store/snapshots/` and atomically switch the `CURRENT` pointer
only after a generation is complete. Up to three generations are retained for
recovery. Existing flat-layout stores remain readable and migrate on their next
save. Back up the entire `data/documind_store/` directory before replacing the
code or moving the library to another machine. Do not upload private PDFs or
this directory to a public repository.

## Review notes

See [REVIEW_NOTES.md](REVIEW_NOTES.md) for corrections made and validation limitations.
