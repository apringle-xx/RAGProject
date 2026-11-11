import pytest

from rag.agent import NaiveRAG, RetrievalAgent
from rag.evaluate import score
from rag.grounding import citation_labels, is_abstention, is_conversation
from rag.providers import OfflineModel
from rag.types import Answer, Turn
from tests.test_agent import ScriptedModel, search


@pytest.mark.parametrize(
    "text",
    [
        "I cannot determine that from these documents.",
        "I can't find this information.",
        "I couldn't confirm Acme revenue.",
        "No relevant evidence was found.",
        "I don't know based on the available documents.",
    ],
)
def test_scoring_and_guard_share_abstention_wording(library, text):
    assert is_abstention(text)
    case = {"facts": [], "sources": [], "abstain": True}
    assert score(Answer(text), case, set())["correct"] == 1
    model = ScriptedModel([Turn(text)])
    assert RetrievalAgent(library, model).answer("What is Acme revenue?").text == text


def test_uncited_fact_is_blocked_with_no_search_and_no_hits(library):
    model = ScriptedModel([Turn("Atlas launches tomorrow.")])
    result = RetrievalAgent(library, model).answer("When does Atlas launch?")
    assert "tomorrow" not in result.text and not result.citations
    model = ScriptedModel([search("Atlas"), Turn("Atlas launches tomorrow.")])
    result = RetrievalAgent(library, model).answer("When does Atlas launch?")
    assert "tomorrow" not in result.text and result.trace[0]["hits"] == 0


@pytest.mark.parametrize("token", ["[S99 ]", "[S 99]", "[s99]", "[S1, S99]", "[S99"])
def test_malformed_citations_cannot_hide_beside_valid_one(library, token):
    library.ingest("x.txt", b"Atlas owner Maya")
    model = ScriptedModel([search("Atlas"), Turn(f"Maya [S1]. Someone else {token}.")])
    result = RetrievalAgent(library, model).answer("Who owns Atlas?")
    assert "couldn't verify" in result.text and not result.citations
    with pytest.raises(ValueError):
        citation_labels(f"Maya [S1] {token}")


def test_baseline_greeting_survives_unconditional_retrieval(library):
    library.ingest("x.txt", b"Atlas owner Maya")
    result = NaiveRAG(library, OfflineModel()).answer("Hello")
    assert result.text.startswith("Hello") and len(result.trace) == 1
    assert is_conversation("Hello!")
    assert not is_conversation("Hello, when does Atlas launch?")


def test_unicode_factual_suffix_never_disappears_from_question(library):
    assert not is_conversation("hi 何时发布？")
    assert not is_conversation("hi 12345")
    model = ScriptedModel([Turn("Atlas launches tomorrow.")])
    result = RetrievalAgent(library, model).answer("hi 何时发布？")
    assert "tomorrow" not in result.text


def test_combined_greeting_remains_conversational(library):
    assert is_conversation("Hi, how are you?")
    result = RetrievalAgent(library, OfflineModel()).answer("Hi, how are you?")
    assert result.text.startswith("Hello") and not result.trace


def test_offline_provider_ignores_old_tool_evidence_for_new_question():
    old = '{"sources":[{"label":"S1","text":"Atlas launches in March"}]}'
    messages = [
        {"role": "user", "content": "When does Atlas launch?"},
        {"role": "tool", "content": old},
        {"role": "assistant", "content": "March [S1]"},
        {"role": "user", "content": "When does Borealis launch?"},
    ]
    turn = OfflineModel().complete(messages, tools=True)
    assert turn.calls and "Borealis" in turn.calls[0].arguments
    messages.append({"role": "tool", "content": '{"sources":[]}'})
    assert "don't know" in OfflineModel().complete(messages, tools=True).text
