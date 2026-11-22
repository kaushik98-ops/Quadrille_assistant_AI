from pathlib import Path

import pytest

from assistant.rag.evaluation import (
    EvalCase,
    check_answer,
    load_cases,
    score_case,
    split_pages,
    suggest_threshold,
    summarize,
)
from assistant.rag.service import Source
from assistant.rag.store import Hit

EVALS = Path(__file__).resolve().parents[2] / "evals"


def hit(source, page, score):
    return Hit(text="t", source=source, page=page, score=score)


def test_split_pages_numbering():
    assert split_pages("page one\n=== PAGE ===\npage two") == [(1, "page one"), (2, "page two")]
    assert split_pages("only one page") == [(1, "only one page")]
    # A leading marker is ignored, and an empty page still uses up its number.
    assert split_pages("=== PAGE ===\nA\n=== PAGE ===\n\n=== PAGE ===\nC") == [(1, "A"), (3, "C")]


def test_load_cases_valid(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        '{"question": "Q1?", "expected_source": "a.txt", "expected_page": 2, "answer_contains_any": ["x"]}\n'
        "\n"
        '{"question": "Q2?", "expect_answer": false}\n',
        encoding="utf-8",
    )
    cases = load_cases(f)
    assert cases[0] == EvalCase("Q1?", True, "a.txt", 2, ("x",), ())
    assert cases[1].expect_answer is False and cases[1].expected_source is None


def test_load_cases_rejects_bad_lines(tmp_path):
    bad_files = [
        "{broken",
        '{"expected_source": "a.txt"}',
        '{"question": "  "}',
        '{"question": "Q?"}',  # answerable but no expected_source
    ]
    for i, content in enumerate(bad_files):
        f = tmp_path / f"bad{i}.jsonl"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_cases(f)


def test_score_case_rank_and_scores():
    case = EvalCase("q", True, "a.txt", 2)
    hits = [hit("b.txt", 1, 0.9), hit("a.txt", 1, 0.8), hit("a.txt", 2, 0.7)]
    r = score_case(case, hits)
    assert (r.rank, r.top_score, r.match_score) == (3, 0.9, 0.7)  # right file, wrong page, doesn't count
    anywhere = score_case(EvalCase("q", True, "a.txt", None), hits)
    assert anywhere.rank == 2
    assert score_case(case, [hit("b.txt", 1, 0.9)]).rank is None
    assert score_case(case, []).top_score == 0.0


def test_score_negative_case():
    r = score_case(EvalCase("q", False), [hit("a.txt", 1, 0.4)])
    assert (r.rank, r.top_score, r.match_score) == (None, 0.4, None)


def test_summarize():
    pos1, pos2 = EvalCase("q1", True, "a.txt", 1), EvalCase("q2", True, "a.txt", 1)
    neg = EvalCase("q3", False)
    results = [
        score_case(pos1, [hit("a.txt", 1, 0.8)]),                          # rank 1
        score_case(pos2, [hit("b.txt", 1, 0.5), hit("a.txt", 1, 0.2)]),    # rank 2, below the threshold
        score_case(neg, [hit("a.txt", 1, 0.1)]),
    ]
    s = summarize(results, min_score=0.3)
    assert (s["positives"], s["negatives"]) == (2, 1)
    assert s["hit_at_1"] == 0.5 and s["hit_at_k"] == 1.0 and s["mrr"] == pytest.approx(0.75)
    assert s["answerable_at_threshold"] == 0.5 and s["abstain_at_threshold"] == 1.0
    assert summarize([], 0.3)["hit_at_k"] is None


def test_suggest_threshold():
    pos = EvalCase("p", True, "a.txt", 1)
    neg = EvalCase("n", False)
    clean = [score_case(pos, [hit("a.txt", 1, 0.6)]), score_case(neg, [hit("a.txt", 1, 0.2)])]
    assert suggest_threshold(clean)[0] == 0.4
    overlap = [score_case(pos, [hit("a.txt", 1, 0.3)]), score_case(neg, [hit("a.txt", 1, 0.5)])]
    value, note = suggest_threshold(overlap)
    assert value is None and "overlap" in note
    assert suggest_threshold([score_case(pos, [hit("a.txt", 1, 0.6)])])[0] is None


def test_check_answer_positive():
    case = EvalCase("q", True, "lease.txt", 2, ("60 days",), ("PWNED",))
    sources = [Source(1, "lease.txt", 2, 0.8), Source(2, "other.txt", 1, 0.5)]
    assert check_answer(case, "It needs 60 days notice [1].", sources) == []
    assert "missing the expected fact" in check_answer(case, "Some notice [1].", sources)
    assert "no citation" in check_answer(case, "It needs 60 days.", sources)
    assert "cites the wrong source" in check_answer(case, "It needs 60 days [2].", sources)
    assert "followed an instruction hidden in a document" in check_answer(case, "PWNED [1] 60 days", sources)


def test_check_answer_negative():
    case = EvalCase("q", False)
    assert check_answer(case, "I couldn't find that in your documents.", []) == []
    problems = check_answer(case, "The capital is Canberra.", [Source(1, "a.txt", 1, 0.3)])
    assert "retrieved passages for an unanswerable question" in problems
    assert "did not say it could not find the answer" in problems


def test_shipped_eval_set_is_consistent():
    """Every case points at a real page of the corpus, and its expected fact is on that page."""
    pages = {}
    for f in (EVALS / "corpus").glob("*.txt"):
        pages[f.name] = dict(split_pages(f.read_text(encoding="utf-8")))
    cases = load_cases(EVALS / "questions.jsonl")
    assert len(cases) >= 10 and any(not c.expect_answer for c in cases)
    for c in cases:
        if not c.expect_answer:
            continue
        assert c.expected_source in pages, c.question
        text = pages[c.expected_source][c.expected_page].lower()
        if c.answer_contains_any:
            assert any(s.lower() in text for s in c.answer_contains_any), c.question
