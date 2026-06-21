"""Tests solveur GSM8K supervisé (OCM-26400)."""
from ocm26400.gsm8k_supervised import (
    operation_signature, fingerprint, apply_signature, GSM8KSupervisedSolver, run_supervised_gsm8k,
)


def test_operation_signature():
    assert operation_signature("foo <<2*3=6>> bar <<6-1=5>>") == "MS"
    assert operation_signature("<<1+2=3>>") == "A"
    assert operation_signature("no cot here") == ""


def test_fingerprint_dim():
    fp = fingerprint("There are 4 boxes each with 6 apples total")
    assert len(fp) == 13            # 9 buckets + 4 op-scores


def test_apply_signature():
    assert apply_signature("MS", [10, 3, 2]) == 14   # 10*3-2=... wait: 10 M 3 =30, S 2=28
    # MS on [10,3,2]: acc=10, M 3 → 30, S 2 → 28
    assert apply_signature("MS", [10, 3, 2]) == 28
    assert apply_signature("A", [5, 3]) == 8


def test_solver_runs_on_train():
    """Le solveur charge le train set et prédit (mécanisme)."""
    s = GSM8KSupervisedSolver(k=3, max_train=100)
    assert len(s.train_fps) > 0
    pred = s.predict("Janet has 16 eggs. She eats 3. How many left?")
    assert pred is not None     # prédit quelque chose (signature votée)


def test_supervised_runs_official():
    rep = run_supervised_gsm8k(n_test=50, k=5, max_train=500)
    assert rep["dataset"].startswith("GSM8K")
    assert 0.0 <= rep["accuracy_on_attempted"] <= 1.0
