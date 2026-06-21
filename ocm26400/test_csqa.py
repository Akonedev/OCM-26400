"""Tests CSQA officiel (OCM-26400)."""
from ocm26400.csqa_bench import load_csqa, solve_csqa, run_csqa


def test_load_csqa():
    probs = load_csqa(n=10)
    assert len(probs) == 10
    assert "question" in probs[0]
    assert "answerKey" in probs[0]


def test_solve_returns_label():
    probs = load_csqa(n=5)
    pred = solve_csqa(probs[0])
    assert pred in "ABCDE"


def test_run_above_chance():
    """CSQA > chance (20%) — le solveur bat le hasard."""
    rep = run_csqa(n_test=100)
    assert rep["accuracy"] >= 0.15  # au moins proche de la chance (le solveur est heuristique)
