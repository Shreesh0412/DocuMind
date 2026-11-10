import re

# Supports both [S1] and grouped citations such as [S1, S2].
CITATION_GROUP_PATTERN = re.compile(
    r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]"
)
CITATION_ID_PATTERN = re.compile(r"S\d+")


def validate_citations(
    answer: str,
    available_source_ids: set[str],
) -> tuple[str, list[str], list[str]]:
    """Validate citations and remove unavailable source IDs.

    Supports individual citations ([S1]) and grouped citations ([S1, S2]).
    If a group contains both valid and invalid IDs, only the valid IDs remain.
    The returned valid/invalid ID lists preserve first-seen order and omit
    duplicates.
    """
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    # Extract every ID from every recognised citation group.
    groups = list(CITATION_GROUP_PATTERN.finditer(answer))
    found_ids = [
        source_id
        for group in groups
        for source_id in CITATION_ID_PATTERN.findall(group.group(1))
    ]

    valid_ids = list(dict.fromkeys(
        source_id for source_id in found_ids
        if source_id in available_source_ids
    ))
    invalid_ids = list(dict.fromkeys(
        source_id for source_id in found_ids
        if source_id not in available_source_ids
    ))

    def clean_group(match: re.Match[str]) -> str:
        ids_in_group = CITATION_ID_PATTERN.findall(match.group(1))
        kept_ids = list(dict.fromkeys(
            source_id for source_id in ids_in_group
            if source_id in available_source_ids
        ))
        return f"[{', '.join(kept_ids)}]" if kept_ids else ""

    cleaned_answer = CITATION_GROUP_PATTERN.sub(clean_group, answer)
    cleaned_answer = re.sub(r"[ \t]{2,}", " ", cleaned_answer)
    cleaned_answer = re.sub(r" +([.,;:!?])", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\n{3,}", "\n\n", cleaned_answer)

    return cleaned_answer.strip(), valid_ids, invalid_ids
