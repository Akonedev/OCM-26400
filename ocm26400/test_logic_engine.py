"""Tests logique propositionnelle (OCM-26400)."""
from ocm26400.logic_engine import (
    evaluate, truth_table, is_tautology, is_contradiction, is_satisfiable,
    valid_argument, modus_ponens, extract_vars,
)


def test_evaluate_basic():
    assert evaluate("p ∧ q", {"p": True, "q": True}) is True
    assert evaluate("p ∧ q", {"p": True, "q": False}) is False
    assert evaluate("p ∨ q", {"p": False, "q": False}) is False
    assert evaluate("¬p", {"p": False}) is True


def test_implies():
    assert evaluate("p → q", {"p": False, "q": False}) is True   # F→F = T
    assert evaluate("p → q", {"p": True, "q": False}) is False    # T→F = F


def test_modus_ponens_tautology():
    assert is_tautology("(p → q) ∧ p → q") is True


def test_non_contradiction():
    assert is_tautology("¬(p ∧ ¬p)") is True


def test_not_tautology():
    assert is_tautology("p ∨ q → p") is False


def test_contradiction_detection():
    assert is_contradiction("p ∧ ¬p") is True


def test_satisfiable():
    assert is_satisfiable("p ∧ q") is True
    assert is_satisfiable("p ∧ ¬p") is False


def test_valid_argument():
    """Argument valide : prémisses → conclusion est une tautologie."""
    assert valid_argument(["p → q", "p"], "q") is True
    assert valid_argument(["p → q"], "q") is False    # pas suffisant sans p


def test_truth_table_size():
    tt = truth_table("p ∧ q")
    assert len(tt) == 4     # 2 vars × 2 valeurs


def test_extract_vars():
    assert extract_vars("p ∧ q → r") == ["p", "q", "r"]


def test_security_rejects_arbitrary():
    """SÉCURITÉ : evaluate ne doit pas exécuter de code arbitraire."""
    assert evaluate("__import__('os')", {}) is False
    assert evaluate("open('x')", {}) is False
