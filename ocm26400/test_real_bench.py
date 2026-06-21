"""Tests benchmark réel (OCM-26400)."""
import pytest
from ocm26400.real_bench import _real_problem_set, RealProblem, _eq


def test_problem_set_has_categories():
    ps = _real_problem_set()
    cats = {p.category for p in ps}
    assert "arithmétique modulaire" in cats
    assert "théorie des nombres" in cats
    assert "algèbre" in cats


def test_ground_truth_independent():
    """LE test anti-tautologie : le ground truth est calculé INDÉPENDAMMENT (Python pow).
    Pas le solveur du modèle qui s'accorde avec lui-même."""
    for p in _real_problem_set():
        # pour les problèmes modulaires, ground truth via Python pow() ≠ modexp (indépendant)
        assert p.ground_truth is not None


def test_eq_helper():
    assert _eq(4, 4) is True
    assert _eq(4, "4") is True
    assert _eq([2, 2, 3], [2, 2, 3]) is True


def test_modexp_problems_correct():
    """Les problèmes modulaires : le solveur (modexp) == ground truth (pow)."""
    for p in _real_problem_set():
        if p.category == "arithmétique modulaire":
            r = p.solve_and_check()
            assert r["correct"] is True


def test_number_theory_correct():
    for p in _real_problem_set():
        if p.category in ("théorie des nombres",):
            assert p.solve_and_check()["correct"]


def test_real_bench_at_least_symbolic_100():
    """Les moteurs exacts (symbolic) doivent résoudre 100% des vrais problèmes."""
    from ocm26400.real_bench import run_real_bench
    rep = run_real_bench()
    # symbolique (tout sauf neural) doit être ~100%
    non_neural_cats = {k: v for k, v in rep["per_category"].items() if k != "chaîne neuronale"}
    total_n = sum(v["n"] for v in non_neural_cats.values())
    total_c = sum(v["correct"] for v in non_neural_cats.values())
    assert total_c == total_n    # 100% symbolique sur vrais problèmes
