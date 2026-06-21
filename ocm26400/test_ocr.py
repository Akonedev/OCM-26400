"""Tests OCR réel MNIST (OCM-26400) — modalité OCR."""
import pytest
import torch
from ocm26400.ocr import OCRDigitRecognizer, train_ocr, recognize, load_mnist


def test_model_no_transformer():
    """CNN, pas de transformer/attention."""
    m = OCRDigitRecognizer()
    has_tf = any("Transformer" in type(mod).__name__ or "Attention" in type(mod).__name__
                 for mod in m.modules())
    assert not has_tf


def test_forward_shape():
    m = OCRDigitRecognizer()
    out = m(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_ocr_trains_and_above_chance():
    """OCR MNIST > 70% (largement au-dessus de la chance 10%)."""
    model, res = train_ocr(n_train=3000, n_test=500, n_steps=400)
    assert res["test_acc"] > 0.7


def test_recognize_returns_digit():
    model, _ = train_ocr(n_train=500, n_test=100, n_steps=50)
    Xtr, ytr, _, _ = load_mnist(500, 100)
    pred = recognize(model, Xtr[0].reshape(28, 28))
    assert 0 <= pred <= 9


def test_load_mnist_shape():
    Xtr, ytr, Xte, yte = load_mnist(100, 50)
    assert Xtr.shape[0] == 100 and Xte.shape[0] == 50
    assert len(set(ytr)) == 10
