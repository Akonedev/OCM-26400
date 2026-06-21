"""Tests benchmark GSM8K officiel (OCM-26400)."""
import pytest
from ocm26400.gsm8k_bench import load_gsm8k, extract_answer, extract_numbers, solve_word_problem, run_gsm8k


def test_extract_answer_format():
    assert extract_answer("blah blah\n#### 18") == 18.0
    assert extract_answer("... #### 1,234") == 1234.0


def test_extract_numbers():
    assert extract_numbers("Janet has 16 eggs and 3 ducks") == [16.0, 3.0]


def test_solve_subtraction():
    """'has 10, gives 3, how many left' → 7."""
    pred = solve_word_problem("Tom has 10 apples. He gives 3 to his friend. How many left?")
    assert pred == 7


def test_solve_multiplication():
    pred = solve_word_problem("There are 4 boxes each with 6 apples. How many total?")
    assert pred == 24


def test_gsm8k_runs_on_official_data():
    """LE test : le modèle tourne sur le VRAI dataset GSM8K officiel (au moins quelques problèmes)."""
    rep = run_gsm8k(n=50)
    assert rep["dataset"].startswith("GSM8K")
    assert rep["n_problems"] > 0
    # accuracy honnête (rule-based, faible sur multi-étapes) — on vérifie juste que ça tourne
    assert 0.0 <= rep["accuracy_on_attempted"] <= 1.0
    assert rep["coverage"] > 0.5     # le solveur tente la majorité
