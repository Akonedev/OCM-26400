"""Tests cascade primitives grokkées (OCM-26400)."""
from ocm26400.language_cascade_grok import solve_gsm8k_grokked, _build_grokked_solvers


def test_solve_janet_grokked():
    """Janet résolu via primitives GROKKÉES (neural, pas hardcodé)."""
    blk_wn, blk_co, d, dev = _build_grokked_solvers("cpu")
    q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
         "and bakes muffins for her friends every day with four. She sells the remainder "
         "at the farmers market daily for $2 per fresh duck egg. How much in dollars "
         "does she make every day at the farmers market?")
    pred, trace = solve_gsm8k_grokked(q, blk_wn, blk_co, d, dev)
    assert pred == 18.0, f"Janet should be 18, got {pred}"
    assert len(trace) >= 3
