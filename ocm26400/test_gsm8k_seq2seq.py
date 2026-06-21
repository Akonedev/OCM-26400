"""Tests seq2seq COPY GSM8K (OCM-26400)."""
import pytest
from ocm26400.gsm8k_seq2seq import (
    _trace_to_actions, _actions_to_value, _question_numbers_with_pos, ACTIONS, ACT_TO_IDX,
    Seq2SeqCoT, train_seq2seq, predict_cot,
)


def test_actions_vocab():
    assert "OP_M" in ACTIONS and "COPY_0" in ACTIONS and "START" in ACTIONS


def test_trace_to_actions_runs():
    a = _trace_to_actions("Janet has 16 eggs and eats 3", "<<16-3=13>>\n#### 13")
    assert a[0] == ACT_TO_IDX["START"]
    assert ACT_TO_IDX["COPY_0"] in a       # 16 = 1er nombre
    assert ACT_TO_IDX["OP_S"] in a         # soustraction


def test_actions_to_value():
    # COPY_0(10) OP_M COPY_1(3) → 10*3=30
    acts = [ACT_TO_IDX["COPY_0"], ACT_TO_IDX["OP_M"], ACT_TO_IDX["COPY_1"]]
    assert _actions_to_value(acts, [10.0, 3.0]) == 30.0


def test_actions_subtraction():
    acts = [ACT_TO_IDX["COPY_0"], ACT_TO_IDX["OP_S"], ACT_TO_IDX["COPY_1"]]
    assert _actions_to_value(acts, [10.0, 3.0]) == 7.0


def test_model_forward():
    import torch
    m = Seq2SeqCoT(50)
    out = m(torch.randint(0, 50, (2, 10)), torch.randint(0, len(ACTIONS), (2, 8)))
    assert out.shape == (2, 8, len(ACTIONS))


def test_train_predict_run():
    m, vocab = train_seq2seq(n_train=80, n_steps=30)
    pred = predict_cot(m, vocab, "There are 4 boxes each with 6 apples.")
    assert pred is None or isinstance(pred, float)
