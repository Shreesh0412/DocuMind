import re

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")

def validate_citations(
    answer: str,
    available_source_ids: set[str],
) -> tuple[str, list[str], list[str]]:
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    found_ids = CITATION_PATTERN.findall(answer)
    valid_ids = list(dict.fromkeys(
        source_id for source_id in found_ids
        if source_id in available_source_ids
    ))
    invalid_ids = list(dict.fromkeys(
        source_id for source_id in found_ids
        if source_id not in available_source_ids
    ))

    cleaned_answer = CITATION_PATTERN.sub(
        lambda match: match.group(0)
        if match.group(1) in available_source_ids
        else "",
        answer,
    )
    cleaned_answer = re.sub(r"[ \t]{2,}", " ", cleaned_answer)
    cleaned_answer = re.sub(r" +([.,;:!?])", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\n{3,}", "\n\n", cleaned_answer)

    return cleaned_answer.strip(), valid_ids, invalid_ids
