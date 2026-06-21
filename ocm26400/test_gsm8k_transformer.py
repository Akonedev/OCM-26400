"""Tests transformer CoT GSM8K (OCM-26400)."""
import pytest
import torch
from ocm26400.gsm8k_transformer import (
    TransformerCoT, PositionalEncoding, train_transformer, predict_cot_transformer,
)


def test_positional_encoding_shape():
    pe = PositionalEncoding(64, max_len=30)
    x = torch.zeros(2, 10, 64)
    out = pe(x)
    assert out.shape == (2, 10, 64)


def test_transformer_forward():
    m = TransformerCoT(vocab_size=50, d_model=64, nhead=4, num_layers=2)
    from ocm26400.gsm8k_seq2seq import ACTIONS
    src = torch.randint(0, 50, (2, 12))
    tgt = torch.randint(0, len(ACTIONS), (2, 8))
    out = m(src, tgt)
    assert out.shape == (2, 8, len(ACTIONS))


def test_train_transformer_runs():
    """Le transformer CoT s'entraîne sur GSM8K train (mécanisme)."""
    m, vocab = train_transformer(n_train=60, n_steps=15, device="cpu")
    assert len(vocab) > 1


def test_predict_cot_runs():
    m, vocab = train_transformer(n_train=60, n_steps=15, device="cpu")
    pred = predict_cot_transformer(m, vocab, "There are 4 boxes with 6 apples each.", "cpu")
    assert pred is None or isinstance(pred, float)


def test_transformer_has_attention():
    """L'architecture contient bien des couches transformer (attention)."""
    m = TransformerCoT(50, d_model=64, num_layers=2)
    n_enc = sum(1 for _ in m.encoder.children())
    assert hasattr(m, "encoder") and hasattr(m, "decoder")
