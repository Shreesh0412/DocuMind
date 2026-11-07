from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PROJECT_ROOT / "data" / "documind_store"
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "retrieval_eval_set.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def result_matches_case(result: dict, case: dict) -> bool:
    expected_document = case.get("expected_document")
    expected_document_id = case.get("expected_document_id")
    expected_pages = case.get("expected_pages") or []
    expected_keywords = case.get("expected_keywords") or []

    if expected_document and str(result.get("source", "")).casefold() != str(expected_document).casefold():
        return False
    if expected_document_id and result.get("document_id") != expected_document_id:
        return False
    if expected_pages and int(result.get("page", -1)) not in {int(page) for page in expected_pages}:
        return False
    if expected_keywords:
        text = str(result.get("text", "")).casefold()
        if not any(str(keyword).casefold() in text for keyword in expected_keywords):
            return False
    return bool(expected_document or expected_document_id or expected_pages or expected_keywords)


def evaluate_cases(
    store,
    cases: list[dict],
    top_k: int = 5,
    query_encoder: Callable[[str], object] | None = None,
    mode: str = "semantic",
) -> dict:
    if not cases:
        raise ValueError("The evaluation dataset has no test cases.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    if mode not in {"semantic", "hybrid"}:
        raise ValueError("mode must be 'semantic' or 'hybrid'.")
    if query_encoder is None:
        from src.embeddings import embed_query
        query_encoder = embed_query

    rows = []
    reciprocal_ranks = []
    hits = 0

    for case in cases:
        question = str(case.get("question", "")).strip()
        if not question:
            raise ValueError(f"Evaluation case {case.get('id', '?')} has an empty question.")

        query_embedding = query_encoder(question)
        if mode == "hybrid":
            results = store.search_hybrid(
                query_embedding,
                query_text=question,
                top_k=top_k,
                min_score=None,
            )
        else:
            results = store.search(
                query_embedding,
                top_k=top_k,
                min_score=None,
            )

        relevant_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if result_matches_case(result, case)
            ),
            None,
        )
        hit = relevant_rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / relevant_rank if hit else 0.0)

        retrieved = []
        for item in results:
            entry = {
                "source": item.get("source"),
                "page": item.get("page"),
                "score": float(item.get("score", 0.0)),
            }
            for field in ("semantic_score", "bm25_score", "hybrid_score", "semantic_rank", "keyword_rank"):
                if field in item:
                    entry[field] = item[field]
            retrieved.append(entry)

        rows.append(
            {
                "id": case.get("id", f"Q{len(rows) + 1}"),
                "question": question,
                "hit_at_k": hit,
                "first_relevant_rank": relevant_rank,
                "retrieved": retrieved,
            }
        )

    total = len(cases)
    return {
        "mode": mode,
        "total_cases": total,
        "top_k": top_k,
        "hits": hits,
        "hit_rate_at_k": hits / total,
        "mrr_at_k": sum(reciprocal_ranks) / total,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocuMind semantic or hybrid retrieval against a labelled test set."
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("semantic", "hybrid"),
        default="semantic",
        help="semantic keeps the old FAISS-only baseline; hybrid combines FAISS and BM25 with RRF.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
        raise SystemExit("Evaluation dataset must be a JSON object with a 'cases' list.")

    store = VectorStore(embedding_model=EMBEDDING_MODEL)
    if not store.load(args.store):
        raise SystemExit(f"No saved DocuMind index found at: {args.store}")

    report = evaluate_cases(
        store,
        dataset["cases"],
        top_k=args.top_k,
        mode=args.mode,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nSaved evaluation report to: {args.output}")


if __name__ == "__main__":
    main()
