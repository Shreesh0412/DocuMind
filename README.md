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

The reviewed test suite contains 57 tests. To make a deliberate Gemini connectivity request, run this script separately:

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

## Local data

DocuMind saves the FAISS index, embeddings, chunks, and document metadata in `data/documind_store/`. Back up this directory before replacing the code or moving the local library to another machine. Do not upload private PDFs or this directory to a public repository.

## Review notes

See [REVIEW_NOTES.md](REVIEW_NOTES.md) for corrections made and validation limitations.
