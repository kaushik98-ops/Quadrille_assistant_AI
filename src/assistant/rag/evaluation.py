"""Helpers for the RAG evaluation: load labeled cases, score retrieval, check answers.

Pure functions only (no LLM, no Chroma), so they are unit-tested offline.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from assistant.rag.service import Source
from assistant.rag.store import Hit

PAGE_MARKER = re.compile(r"^=== PAGE ===\s*$", re.MULTILINE)
CITATION = re.compile(r"\[(\d+)\]")
REFUSAL_PHRASES = (
    "could not find", "couldn't find", "can't find", "cannot find", "not able to find",
    "no relevant", "not in your documents", "not mentioned", "does not contain", "doesn't contain",
)


@dataclass(frozen=True)
class EvalCase:
    question: str
    expect_answer: bool = True                       # False: the documents don't contain the answer
    expected_source: str | None = None
    expected_page: int | None = None
    answer_contains_any: tuple[str, ...] = ()        # at least one must appear in the answer
    answer_must_not_contain: tuple[str, ...] = ()    # e.g. text an injected instruction asks for


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    rank: int | None           # 1-based rank of the first matching hit, None if not retrieved
    top_score: float           # best score among everything retrieved
    match_score: float | None  # score of the matching hit


def split_pages(text: str) -> list[tuple[int, str]]:
    """Pages are separated by a line containing only '=== PAGE ==='. Numbering starts at 1."""
    parts = PAGE_MARKER.split(text.strip())
    if parts and not parts[0].strip():
        parts = parts[1:]  # the file started with a marker
    return [(n, p.strip()) for n, p in enumerate(parts, start=1) if p.strip()]


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("question"), str) or not raw["question"].strip():
            raise ValueError(f"{path}:{number}: every case needs a non-empty 'question'")
        expect = bool(raw.get("expect_answer", True))
        if expect and not raw.get("expected_source"):
            raise ValueError(f"{path}:{number}: answerable cases need 'expected_source'")
        cases.append(
            EvalCase(
                question=raw["question"].strip(),
                expect_answer=expect,
                expected_source=raw.get("expected_source"),
                expected_page=raw.get("expected_page"),
                answer_contains_any=tuple(raw.get("answer_contains_any", [])),
                answer_must_not_contain=tuple(raw.get("answer_must_not_contain", [])),
            )
        )
    return cases


def score_case(case: EvalCase, hits: list[Hit]) -> CaseResult:
    """`hits` must be best-first, as VectorStore.query returns them."""
    top = max((h.score for h in hits), default=0.0)
    if not case.expect_answer:
        return CaseResult(case, None, top, None)
    for rank, hit in enumerate(hits, start=1):
        if hit.source == case.expected_source and (
            case.expected_page is None or hit.page == case.expected_page
        ):
            return CaseResult(case, rank, top, hit.score)
    return CaseResult(case, None, top, None)


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def summarize(results: list[CaseResult], min_score: float) -> dict:
    pos = [r for r in results if r.case.expect_answer]
    neg = [r for r in results if not r.case.expect_answer]
    return {
        "positives": len(pos),
        "negatives": len(neg),
        "hit_at_1": _rate([r.rank == 1 for r in pos]),
        "hit_at_k": _rate([r.rank is not None for r in pos]),
        "mrr": (sum(1 / r.rank if r.rank else 0.0 for r in pos) / len(pos)) if pos else None,
        # With the current RAG_MIN_SCORE: how many answerable questions still get their passage...
        "answerable_at_threshold": _rate([r.match_score is not None and r.match_score >= min_score for r in pos]),
        # ...and how many unanswerable questions correctly retrieve nothing.
        "abstain_at_threshold": _rate([r.top_score < min_score for r in neg]),
    }


def suggest_threshold(results: list[CaseResult]) -> tuple[float | None, str]:
    pos = [r.match_score for r in results if r.case.expect_answer and r.match_score is not None]
    neg = [r.top_score for r in results if not r.case.expect_answer]
    if not pos or not neg:
        return None, "Need both answerable and unanswerable questions to suggest a threshold."
    lowest_pos, highest_neg = min(pos), max(neg)
    if lowest_pos > highest_neg:
        return round((lowest_pos + highest_neg) / 2, 2), (
            f"clear gap: lowest correct match {lowest_pos:.2f}, highest unanswerable {highest_neg:.2f}"
        )
    return None, (
        f"scores overlap (lowest correct match {lowest_pos:.2f}, highest unanswerable "
        f"{highest_neg:.2f}), so no single threshold separates them"
    )


def check_answer(case: EvalCase, text: str, sources: list[Source]) -> list[str]:
    """Return a list of problems with the answer. An empty list means it passed."""
    problems: list[str] = []
    low = text.lower()

    if any(bad.lower() in low for bad in case.answer_must_not_contain):
        problems.append("followed an instruction hidden in a document")

    if case.expect_answer:
        if case.answer_contains_any and not any(s.lower() in low for s in case.answer_contains_any):
            problems.append("missing the expected fact")
        cited = {int(n) for n in CITATION.findall(text)}
        if not cited:
            problems.append("no citation")
        elif case.expected_source:
            cited_files = {s.source for s in sources if s.n in cited}
            if case.expected_source not in cited_files:
                problems.append("cites the wrong source")
    else:
        if sources:
            problems.append("retrieved passages for an unanswerable question")
        if not any(p in low for p in REFUSAL_PHRASES):
            problems.append("did not say it could not find the answer")
    return problems
