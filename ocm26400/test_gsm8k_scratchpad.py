"""Tests scratchpad cascade GSM8K (OCM-26400)."""
from ocm26400.gsm8k_scratchpad import solve_scratchpad_cascade, run_scratchpad_gsm8k, extract_step


def test_extract_step_subtraction():
    acc, desc = extract_step("She eats 3 for breakfast.", 16.0)
    assert acc == 13.0 and "16.0 - 3.0" in desc


def test_extract_step_multiplication():
    # "times" sans conflit avec "costs"
    acc, desc = extract_step("He has 3 times as many.", 9.0)
    assert acc == 27.0


def test_scratchpad_trace_visible():
    """Le scratchpad rend les intermédiaires VISIBLES (loi L1)."""
    pred, trace = solve_scratchpad_cascade("Tom has 10 apples. He gives 3 to his friend. How many left?")
    assert len(trace) > 0       # trace avec intermédiaires
    assert any("=" in t for t in trace)  # calcul intermédiaire visible


def test_run_official():
    rep = run_scratchpad_gsm8k(n_test=50)
    assert rep["dataset"].startswith("GSM8K")
    assert rep["paradigm"].startswith("L1")
    assert 0 <= rep["accuracy_on_attempted"] <= 1
