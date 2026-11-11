from rag.evaluate import comparison_table, run_evaluation, score
from rag.types import Answer, Chunk, Citation


def test_fixed_evaluation_exercises_both_pipelines_without_network():
    report = run_evaluation(offline=True)
    assert set(report["systems"]) == {"Naive single-shot", "Agentic"}
    for metrics in report["systems"].values():
        assert len(metrics["results"]) == 8
        assert metrics["errors"] == 0
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["source_recall"] <= 1
    assert report["systems"]["Agentic"]["no_retrieval_accuracy"] == 1
    assert report["systems"]["Naive single-shot"]["no_retrieval_accuracy"] == 0
    assert "OFFLINE DEMO" in comparison_table(report)
    assert len(report["corpus_sha256"]) == 64


def test_scoring_checks_all_facts_sources_and_abstention():
    chunk = Chunk("known", "doc", "atlas.txt", "text", 0, "Atlas owner Maya Chen")
    case = {"facts": [["Maya Chen"], ["Oslo"]], "sources": ["atlas.txt", "people.txt"]}
    metrics = score(Answer("Maya Chen [S1]", [Citation("S1", chunk)]), case, {"known"})
    assert metrics["correct"] == 0
    assert metrics["source_recall"] == 0.5
    assert metrics["valid_citations"] == 1
    unknown = {"facts": [], "sources": [], "abstain": True}
    assert score(Answer("I don't know."), unknown, set())["correct"] == 1
    assert score(Answer("Revenue is 100 million."), unknown, set())["correct"] == 0
