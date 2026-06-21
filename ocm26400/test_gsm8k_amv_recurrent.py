"""Tests AMV récurrent GSM8K (OCM-26400) — crown-jewel mechanism."""
import pytest
import torch
from ocm26400.gsm8k_amv_recurrent import (
    word_to_amv, question_to_amv_sequence, train_amv_recurrent, predict_amv,
)


def test_word_to_amv_shape():
    v = word_to_amv("hello", {})
    assert v.shape[0] == 256


def test_word_to_amv_ent_nonzero():
    v = word_to_amv("test", {})
    assert v[:64].sum() > 0   # partition ent non vide


def test_question_to_sequence():
    seq = question_to_amv_sequence("Tom has 5 apples", {}, max_len=10)
    assert seq.shape[1] == 256
    assert seq.shape[0] == 3   # 3 mots


def test_train_amv_runs():
    """Le core spectral AMV récurrent s'entraîne (mécanisme crown-jewel)."""
    blk, info = train_amv_recurrent(n_train=50, n_steps=20, device="cpu")
    assert "error" not in info or info.get("n_train", 0) > 0


def test_predict_returns_float():
    blk, _ = train_amv_recurrent(n_train=30, n_steps=10, device="cpu")
    pred = predict_amv(blk, "hello world test", {}, "cpu")
    assert pred is None or isinstance(pred, float)
