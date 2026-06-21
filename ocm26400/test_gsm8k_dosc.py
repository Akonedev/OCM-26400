"""Tests DOSC curriculum GSM8K (OCM-26400) — suit la procédure documentée."""
import pytest
import torch
from ocm26400.gsm8k_dosc import (
    DOSCModel, _count_steps, filter_by_steps, train_dosc_phase, predict_dosc,
)


def test_count_steps():
    assert _count_steps("foo <<1+2=3>> bar <<3-1=2>>") == 2
    assert _count_steps("no cot") == 0


def test_filter_by_steps():
    probs = [{"answer": "<<1+1=2>> #### 2"}, {"answer": "<<1+1=2>><<2+2=4>> #### 4"}]
    p1 = filter_by_steps(probs, 1, 1)
    assert len(p1) == 1
    p2 = filter_by_steps(probs, 1, 2)
    assert len(p2) == 2


def test_model_no_transformer():
    m = DOSCModel(50, d_model=64)
    from ocm26400.spectral_core import SpectralCoreBlock
    assert isinstance(m.core, SpectralCoreBlock)
    has_tf = any("Transformer" in type(mod).__name__ or "Attention" in type(mod).__name__
                 for mod in m.modules())
    assert not has_tf


def test_model_forward():
    m = DOSCModel(50, d_model=64, seq_len=20)
    from ocm26400.gsm8k_seq2seq import ACTIONS
    src = torch.randint(0, 50, (2, 20))
    tgt = torch.randint(0, len(ACTIONS), (2, 8))
    out = m(src, tgt)
    assert out.shape == (2, 8, len(ACTIONS))


def test_train_phase_runs():
    probs = [{"question": "Tom has 5 apples", "answer": "<<5-2=3>> #### 3"}] * 10
    vocab = {"<pad>": 0, "<unk>": 1, "tom": 2, "has": 3, "apples": 4}
    m = DOSCModel(len(vocab), d_model=64, seq_len=10)
    loss = train_dosc_phase(m, vocab, probs, n_steps=10, device="cpu", phase_name="test")
    assert loss >= 0


def test_predict_returns_float_or_none():
    probs = [{"question": "x", "answer": "<<1+1=2>> #### 2"}] * 5
    vocab = {"<pad>": 0, "<unk>": 1, "x": 2}
    m = DOSCModel(len(vocab), d_model=64, seq_len=5)
    train_dosc_phase(m, vocab, probs, n_steps=5, device="cpu")
    pred = predict_dosc(m, vocab, "x", "cpu")
    assert pred is None or isinstance(pred, float)
