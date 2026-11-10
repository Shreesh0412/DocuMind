from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "answer_eval_set.json"
DEFAULT_RESULTS = PROJECT_ROOT / "tests" / "answer_eval_results.json"
CITATION_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
ABSTENTION_PATTERNS = (
    r"\binsufficient_evidence\b",
    r"could(?: not|n't) find sufficient information",
    r"do(?:es)? not contain enough information",
    r"not enough information (?:in|from) (?:the )?(?:provided )?documents?",
    r"cannot be determined from (?:the )?(?:provided )?documents?",
    r"not specified in (?:the )?(?:provided )?documents?",
    r"cannot answer .* from (?:the )?(?:provided )?documents?",
)


def is_abstention(answer: str) -> bool:
    normalized = answer.strip()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in ABSTENTION_PATTERNS)


def _filename(value: Any) -> str:
    return Path(str(value or "")).name.casefold()


def _concept_satisfied(answer: str, concept: dict) -> bool:
    candidates = concept.get("any_of", [])
    if isinstance(candidates, str):
        candidates = [candidates]
    text = answer.casefold()
    return any(str(candidate).strip().casefold() in text for candidate in candidates if str(candidate).strip())


def evaluate_response(case: dict, response: dict) -> dict:
    answer = str(response.get("answer", "") or "").strip()
    sources = response.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    source_map = {
        str(source.get("id", "")).upper(): source
        for source in sources
        if isinstance(source, dict) and source.get("id")
    }
    cited_ids = list(dict.fromkeys(item.upper() for item in CITATION_PATTERN.findall(answer)))
    invalid_ids = [source_id for source_id in cited_ids if source_id not in source_map]
    citations_valid = bool(cited_ids) and not invalid_ids

    expected_evidence = case.get("expected_evidence", []) or []
    evidence_groups = []
    for expected in expected_evidence:
        expected_source = _filename(expected.get("source"))
        expected_pages = {int(page) for page in expected.get("pages", [])}
        matched = any(
            _filename(source.get("source")) == expected_source
            and (not expected_pages or int(source.get("page", -1)) in expected_pages)
            for source_id, source in source_map.items()
            if source_id in cited_ids
        )
        evidence_groups.append(matched)

    evidence_locations_match = (
        all(evidence_groups) if evidence_groups else None
    )
    answerable = bool(case.get("answerable", True))
    abstained = is_abstention(answer)

    concepts = case.get("required_concepts", []) or []
    concept_results = [
        {
            "label": concept.get("label", "concept"),
            "satisfied": _concept_satisfied(answer, concept),
        }
        for concept in concepts
    ]
    concept_coverage = (
        sum(item["satisfied"] for item in concept_results) / len(concept_results)
        if concept_results else None
    )
    concepts_complete = all(item["satisfied"] for item in concept_results)

    if answerable:
        automated_grounding_pass = bool(
            answer and not abstained and concepts_complete
            and citations_valid and evidence_locations_match is True
        )
        abstention_correct = None
    else:
        automated_grounding_pass = None
        abstention_correct = bool(abstained and not cited_ids and not sources)

    review = response.get("review", {})
    if not isinstance(review, dict):
        review = {}
    answer_supported = review.get("answer_supported")
    citation_supported = review.get("citations_support_claims")
    if answer_supported not in (True, False):
        answer_supported = None
    if citation_supported not in (True, False):
        citation_supported = None

    return {
        "id": case.get("id"),
        "answerable": answerable,
        "answered_with_abstention": abstained,
        "cited_ids": cited_ids,
        "invalid_citation_ids": invalid_ids,
        "citation_ids_valid": citations_valid if answerable else (not cited_ids),
        "expected_evidence_groups": evidence_groups,
        "citation_locations_match_expected": evidence_locations_match,
        "concepts": concept_results,
        "concept_coverage": concept_coverage,
        "automated_grounding_pass": automated_grounding_pass,
        "correct_abstention": abstention_correct,
        "human_answer_supported": answer_supported,
        "human_citations_support_claims": citation_supported,
        "review_complete": answer_supported is not None and citation_supported is not None,
    }


def evaluate_dataset(dataset: dict, results: dict) -> dict:
    cases = dataset.get("cases", [])
    responses = results.get("responses", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain a non-empty 'cases' list.")
    if not isinstance(responses, list):
        raise ValueError("Results file must contain a 'responses' list.")

    response_map = {
        str(item.get("id")): item
        for item in responses
        if isinstance(item, dict) and item.get("id")
    }

    rows = []
    missing_ids = []
    answerable_rows = []
    unanswerable_rows = []
    for case in cases:
        case_id = str(case.get("id", ""))
        response = response_map.get(case_id)
        if response is None or not str(response.get("answer", "") or "").strip():
            missing_ids.append(case_id)
            continue
        row = evaluate_response(case, response)
        rows.append(row)
        (answerable_rows if row["answerable"] else unanswerable_rows).append(row)

    def rate(values: list[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    concept_rates = [row["concept_coverage"] for row in answerable_rows if row["concept_coverage"] is not None]
    human_answer_reviews = [row["human_answer_supported"] for row in rows if row["human_answer_supported"] is not None]
    human_citation_reviews = [row["human_citations_support_claims"] for row in rows if row["human_citations_support_claims"] is not None]

    return {
        "total_cases": len(cases),
        "responses_entered": len(rows),
        "missing_response_ids": missing_ids,
        "answerable_cases": len(answerable_rows),
        "automated_grounding_pass_rate": rate([
            bool(row["automated_grounding_pass"]) for row in answerable_rows
        ]),
        "average_required_concept_coverage": (
            sum(concept_rates) / len(concept_rates) if concept_rates else None
        ),
        "valid_citation_id_rate": rate([
            bool(row["citation_ids_valid"]) for row in answerable_rows
        ]),
        "expected_citation_location_rate": rate([
            row["citation_locations_match_expected"] is True for row in answerable_rows
        ]),
        "unanswerable_cases": len(unanswerable_rows),
        "correct_abstention_rate": rate([
            bool(row["correct_abstention"]) for row in unanswerable_rows
        ]),
        "human_reviewed_cases": sum(row["review_complete"] for row in rows),
        "human_answer_support_rate": rate(human_answer_reviews),
        "human_citation_support_rate": rate(human_citation_reviews),
        "note": (
            "Keyword coverage and expected source/page checks are automated proxies, "
            "not proof of factual correctness or citation entailment. Set the two review "
            "fields after manually checking every answer against the cited passage text."
        ),
        "case_results": rows,
    }


def create_template(dataset: dict) -> dict:
    return {
        "description": (
            "Paste each DocuMind answer and its displayed source entries. "
            "After checking the cited excerpts, set both review values to true or false."
        ),
        "responses": [
            {
                "id": case["id"],
                "answer": "",
                "sources": [],
                "review": {
                    "answer_supported": None,
                    "citations_support_claims": None,
                },
            }
            for case in dataset.get("cases", [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocuMind answer coverage, abstention, and citation locations without calling Gemini."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--init-template", action="store_true", help="Create a blank response file from the dataset.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for the evaluation report JSON.")
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if args.init_template:
        template = create_template(dataset)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created response template: {args.results}")
        print("No Gemini API calls were made.")
        return

    if not args.results.exists():
        raise SystemExit(
            f"Response file not found: {args.results}\n"
            "Create it with --init-template, then fill in answers and cited sources."
        )

    with args.results.open("r", encoding="utf-8") as file:
        results = json.load(file)

    report = evaluate_dataset(dataset, results)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
