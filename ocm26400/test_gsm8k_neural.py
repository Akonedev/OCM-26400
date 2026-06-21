"""Tests neural NL→CoT GSM8K (OCM-26400)."""
import pytest
from ocm26400.gsm8k_neural import (
    NLCoTModel, train_nlcot, predict_signature, _build_vocab, _signature_to_target,
)


def test_signature_to_target():
    steps, ops = _signature_to_target("MA")
    assert steps == 2 and len(ops) == 4


def test_model_forward():
    vocab = {"<pad>": 0, "<unk>": 1, "hello": 2}
    m = NLCoTModel(len(vocab))
    import torch
    sl, ol = m(torch.tensor([[2, 0, 0]]))
    assert sl.shape[-1] == 5 and ol.shape == (1, 4, 4)


def test_train_runs():
    """Le neural NL→CoT s'entraîne sur le train GSM8K (mécanisme)."""
    m, vocab = train_nlcot(n_train=100, n_steps_train=20)
    assert len(vocab) > 1


def test_predict_signature():
    m, vocab = train_nlcot(n_train=100, n_steps_train=20)
    sig = predict_signature(m, vocab, "Janet has 16 eggs. She eats 3. How many left?")
    assert isinstance(sig, str) and all(c in "MDAS" for c in sig)
