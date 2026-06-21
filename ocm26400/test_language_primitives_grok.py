"""Tests primitives linguistiques grokkées (OCM-26400)."""
from ocm26400.language_primitives_grok import (
    word_to_number, extract_all_numbers, cue_to_operation,
    solve_gsm8k_primitives, evaluate_primitives_gsm8k,
)


def test_word_to_number():
    assert word_to_number("three") == 3
    assert word_to_number("sixteen") == 16
    assert word_to_number("half") == 0.5
    assert word_to_number("xyz") is None


def test_extract_all_numbers_finds_words():
    """Primitive 1 : trouve les mots-nombres ET les digits."""
    nums = extract_all_numbers("She eats three and has 4 left")
    assert 3 in nums and 4 in nums


def test_cue_to_operation():
    assert cue_to_operation("She eats 3") == "S"
    assert cue_to_operation("each box costs") == "M"
    assert cue_to_operation("split equally") == "D"


def test_solve_janet():
    """LE test : Janet résolu par primitives grokkées → composition."""
    q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
         "and bakes muffins for her friends every day with four. She sells the remainder "
         "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
         "does she make every day at the farmer's market?")
    pred, trace = solve_gsm8k_primitives(q)
    assert pred == 18.0, f"Janet should be 18, got {pred}"
    assert len(trace) >= 3     # au moins 3 étapes scratchpad


def test_solve_simple_subtraction():
    pred, _ = solve_gsm8k_primitives("Tom has ten apples. He gives three away. How many left?")
    assert pred == 7


def test_evaluate_runs():
    rep = evaluate_primitives_gsm8k(n_test=50)
    assert rep["accuracy"] >= 0.02    # au moins 2% (meilleur que seq2seq)
