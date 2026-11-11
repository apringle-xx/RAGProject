"""Run a fixed corpus and question set with identical providers and retrieval k.

Keyword/alias scoring is deterministic and intentionally modest: it is not a
semantic faithfulness judge. Source recall checks cited source filenames, and
citation validity checks that cited chunk IDs belong to this evaluation corpus.
"""

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from rag.agent import NaiveRAG, RetrievalAgent
from rag.providers import OfflineEmbeddings, OfflineModel, OpenAIEmbeddings, OpenAIModel
from rag.service import Library
from rag.store import VectorStore
from rag.types import Answer

EVALS = Path(__file__).resolve().parents[1] / "evals"


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score(answer: Answer, case: dict, known_chunks: set[str]) -> dict:
    text = normalized(answer.text)
    abstained = bool(
        re.search(r"don t know|do not know|not (?:provided|available|found)|no .*information", text)
    )
    if case.get("abstain"):
        correct = abstained and not answer.citations
    else:
        correct = not abstained and all(
            any(normalized(alias) in text for alias in alternatives)
            for alternatives in case["facts"]
        )
    expected = set(case["sources"])
    actual = {citation.chunk.source for citation in answer.citations}
    return {
        "correct": int(correct),
        "source_recall": len(expected & actual) / len(expected) if expected else None,
        "valid_citations": sum(c.chunk.id in known_chunks for c in answer.citations),
        "citation_count": len(answer.citations),
        "no_retrieval_pass": int(not answer.trace) if case.get("no_retrieval") else None,
    }


def run_evaluation(*, offline: bool = False, k: int = 2) -> dict:
    cases = json.loads((EVALS / "questions.json").read_text())
    embedder = OfflineEmbeddings() if offline else OpenAIEmbeddings()
    model = OfflineModel() if offline else OpenAIModel()
    with TemporaryDirectory(prefix="rag-eval-") as directory:
        library = Library(VectorStore(directory, embedder.name), embedder)
        digest = hashlib.sha256()
        for path in sorted((EVALS / "corpus").glob("*.txt")):
            data = path.read_bytes()
            digest.update(path.name.encode() + b"\0" + data)
            library.ingest(path.name, data)
        known = {hit.chunk.id for hit in library.search("Atlas support security people", k=20)}
        systems = {
            "Naive single-shot": NaiveRAG(library, model, k=k),
            "Agentic": RetrievalAgent(library, model, k=k),
        }
        report = {
            "mode": "offline wiring smoke test" if offline else "OpenAI",
            "embedding": embedder.name,
            "model": "extractive-demo" if offline else model.model,
            "k": k,
            "corpus_sha256": digest.hexdigest(),
            "questions_sha256": hashlib.sha256((EVALS / "questions.json").read_bytes()).hexdigest(),
            "systems": {},
        }
        for name, system in systems.items():
            results = []
            for case in cases:
                try:
                    answer = system.answer(case["question"])
                    results.append(
                        {"id": case["id"], **score(answer, case, known), "answer": asdict(answer)}
                    )
                except Exception as exc:
                    # Never print raw provider messages, which may include sensitive content.
                    results.append(
                        {
                            "id": case["id"],
                            "correct": 0,
                            "error": type(exc).__name__,
                            "source_recall": 0 if case["sources"] else None,
                            "no_retrieval_pass": 0 if case.get("no_retrieval") else None,
                        }
                    )
            successful = [r for r in results if "answer" in r]
            recalls = [r["source_recall"] for r in results if r["source_recall"] is not None]
            count = sum(r["citation_count"] for r in successful)
            no_retrieval = [
                r["no_retrieval_pass"] for r in results if r["no_retrieval_pass"] is not None
            ]
            report["systems"][name] = {
                "accuracy": mean(r["correct"] for r in results),
                "source_recall": mean(recalls) if recalls else 0,
                "citation_validity": (
                    sum(r["valid_citations"] for r in successful) / count if count else 0
                ),
                "mean_latency_ms": mean(r["answer"]["latency_ms"] for r in successful)
                if successful
                else 0,
                "mean_searches": mean(len(r["answer"]["trace"]) for r in successful)
                if successful
                else 0,
                "mean_model_calls": mean(r["answer"]["model_calls"] for r in successful)
                if successful
                else 0,
                "no_retrieval_accuracy": mean(no_retrieval) if no_retrieval else 0,
                "errors": len(results) - len(successful),
                "results": results,
            }
        return report


def comparison_table(report: dict) -> str:
    lines = [
        f"Mode: {report['mode']} | model: {report['model']} | k={report['k']}",
        "System            | Accuracy | Source recall | Valid cites | "
        "Latency ms | Searches | LLM calls | Errors",
        "------------------|----------|---------------|-------------|------------|----------|-----------|-------",
    ]
    for name, metrics in report["systems"].items():
        lines.append(
            f"{name:17} | {metrics['accuracy']:8.1%} | {metrics['source_recall']:13.1%} | "
            f"{metrics['citation_validity']:11.1%} | {metrics['mean_latency_ms']:10.1f} | "
            f"{metrics['mean_searches']:8.2f} | {metrics['mean_model_calls']:9.2f} | "
            f"{metrics['errors']:6}"
        )
    lines.append("Keyword accuracy and source coverage do not establish semantic faithfulness.")
    if report["mode"].startswith("offline"):
        lines.append("OFFLINE DEMO: validates wiring only; does not measure OpenAI quality.")
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="No API calls; demo smoke test only")
    parser.add_argument("--k", type=int, default=2, choices=range(1, 21))
    parser.add_argument("--output", type=Path, help="Optional detailed JSON result path")
    args = parser.parse_args()
    try:
        report = run_evaluation(offline=args.offline, k=args.k)
    except Exception as exc:
        parser.exit(1, f"Evaluation setup failed ({type(exc).__name__}). Check configuration.\n")
    print(comparison_table(report))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if any(s["errors"] for s in report["systems"].values()):
        parser.exit(1, "Some cases failed; inspect --output for exception types.\n")


if __name__ == "__main__":
    main()
