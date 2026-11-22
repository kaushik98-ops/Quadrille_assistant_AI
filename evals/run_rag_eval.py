"""Evaluate RAG quality on a small labeled set.

  python evals/run_rag_eval.py              # retrieval only: no LLM calls, free
  python evals/run_rag_eval.py --answers    # also generate answers with your LLM and check them

Uses your configured embedding provider, indexes evals/corpus into a throwaway folder, and
never touches your real documents. Exits with status 1 if quality falls below the limits.
"""
import argparse
import asyncio
import tempfile
from pathlib import Path

from assistant.chat.service import ChatService
from assistant.core.config import settings
from assistant.core.llm.factory import get_llm_client
from assistant.core.logging import setup_logging
from assistant.core.memory import ConversationMemory
from assistant.rag.embeddings import Embedder, get_embedder
from assistant.rag.evaluation import (
    EvalCase,
    check_answer,
    load_cases,
    score_case,
    split_pages,
    suggest_threshold,
    summarize,
)
from assistant.rag.ingest import ingest_pages
from assistant.rag.retriever import Retriever
from assistant.rag.service import RagService
from assistant.rag.store import VectorStore

HERE = Path(__file__).parent


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def build_index(corpus_dir: str, store: VectorStore, embedder: Embedder) -> list[str]:
    files = sorted(Path(corpus_dir).glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {corpus_dir}")
    for f in files:
        ingest_pages(f.name, split_pages(f.read_text(encoding="utf-8")), embedder, store)
    return [f.name for f in files]


async def run_answers(cases: list[EvalCase], embedder: Embedder, store: VectorStore,
                      tmp: str, k: int) -> list[tuple[EvalCase, str, list[str]]]:
    chat = ChatService(get_llm_client(), ConversationMemory(str(Path(tmp) / "eval.db")))
    rag = RagService(chat, Retriever(embedder, store, k, settings.rag_min_score))
    outcomes = []
    for i, case in enumerate(cases):
        sources, stream = await rag.ask(f"eval-{i}", case.question)  # a fresh conversation per case
        text = "".join([chunk async for chunk in stream])
        outcomes.append((case, text, check_answer(case, text, sources)))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("--corpus", default=str(HERE / "corpus"))
    parser.add_argument("--k", type=int, default=settings.rag_top_k)
    parser.add_argument("--answers", action="store_true", help="also generate and check answers (uses your LLM)")
    parser.add_argument("--min-hit-rate", type=float, default=0.8, help="fail if hit@k is below this")
    parser.add_argument("--min-answer-pass", type=float, default=0.8, help="fail if answer pass rate is below this")
    args = parser.parse_args()

    setup_logging("WARNING")
    cases = load_cases(args.questions)
    embedder = get_embedder()
    failed = False

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = VectorStore(str(Path(tmp) / "chroma"), "eval")
        names = build_index(args.corpus, store, embedder)
        print(f"Indexed {len(names)} documents ({', '.join(names)}); {len(cases)} questions; k={args.k}\n")

        print("RETRIEVAL")
        results = []
        for case in cases:
            hits = store.query(embedder.embed([case.question])[0], args.k)
            result = score_case(case, hits)
            results.append(result)
            if case.expect_answer:
                tag = "HIT " if result.rank else "MISS"
                detail = f"rank={result.rank}" if result.rank else "not retrieved"
                print(f"  {tag} {detail:14} top={result.top_score:.2f}  {case.question}")
            else:
                print(f"  NONE {'(unanswerable)':14} top={result.top_score:.2f}  {case.question}")

        summary = summarize(results, settings.rag_min_score)
        print(f"\n  hit@1 {pct(summary['hit_at_1'])} | hit@{args.k} {pct(summary['hit_at_k'])} | MRR {pct(summary['mrr'])}")
        print(
            f"  At RAG_MIN_SCORE={settings.rag_min_score}: "
            f"{pct(summary['answerable_at_threshold'])} of answerable questions keep their passage, "
            f"{pct(summary['abstain_at_threshold'])} of unanswerable ones retrieve nothing"
        )
        suggested, note = suggest_threshold(results)
        print(f"  Threshold: {note}" + (f" -> try RAG_MIN_SCORE={suggested}" if suggested else ""))
        if (summary["hit_at_k"] or 0) < args.min_hit_rate:
            print(f"  FAIL: hit@{args.k} is below {args.min_hit_rate:.0%}")
            failed = True

        if args.answers:
            print(f"\nANSWERS (using {settings.llm_provider})")
            outcomes = asyncio.run(run_answers(cases, embedder, store, tmp, args.k))
            passed = 0
            for case, text, problems in outcomes:
                passed += not problems
                print(f"  {'PASS' if not problems else 'FAIL'}  {case.question}")
                if problems:
                    print(f"        problems: {'; '.join(problems)}")
                    print(f"        answer:   {' '.join(text.split())[:160]}")
            rate = passed / len(outcomes)
            print(f"\n  answer pass rate {pct(rate)} ({passed}/{len(outcomes)})")
            if rate < args.min_answer_pass:
                print(f"  FAIL: below {args.min_answer_pass:.0%}")
                failed = True

    raise SystemExit(1 if failed else 0)


main()
