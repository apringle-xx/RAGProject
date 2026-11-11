"""Shared, conservative checks; these do not replace semantic faithfulness evaluation."""

import re


def is_conversation(question: str) -> bool:
    # Only whole-message social/capability requests may bypass source requirements.
    normalized = " ".join(re.findall(r"\w+", question.casefold()))
    return normalized in {
        "hello",
        "hello there",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "hi how are you",
        "hello how are you",
        "hey how are you",
        "what can you do",
        "help",
    }


def is_abstention(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:i (?:don['’]t|do not) know|"
            r"(?:cannot|can['’]t|couldn['’]t|could not) "
            r"(?:determine|find|answer|verify|confirm|provide|complete|ground)|"
            r"not (?:provided|available|found) (?:in|from)|"
            r"no (?:relevant |sufficient |available )?(?:information|evidence))\b",
            text,
            re.I,
        )
    )


def citation_labels(text: str) -> list[str]:
    # Reject whitespace, lowercase and mixed-list deviations, including unclosed labels.
    tokens = re.findall(r"\[\s*s(?:ource)?[\s:_-]*\d[^\]]*(?:\]|$)", text, flags=re.I)
    if any(not re.fullmatch(r"\[S[1-9]\d*\]", token) for token in tokens):
        raise ValueError("Malformed source citation.")
    return list(dict.fromkeys(token[1:-1] for token in tokens))
