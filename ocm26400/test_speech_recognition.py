"""Tests ASR (OCM-26400) — modalité parole."""
import pytest
import torch
from ocm26400.speech_recognition import (
    FormantClassifier, train_asr, recognize_phoneme, generate_formant_samples, VOWEL_FORMANTS,
)


def test_model_no_transformer():
    m = FormantClassifier()
    from ocm26400.spectral_core import SpectralCoreBlock
    assert isinstance(m.core, SpectralCoreBlock)
    has_tf = any("Transformer" in type(mod).__name__ or "Attention" in type(mod).__name__
                 for mod in m.modules())
    assert not has_tf


def test_generate_samples():
    X, y = generate_formant_samples(10)
    assert X.shape[0] == 10 * len(VOWEL_FORMANTS)
    assert y.max() < len(VOWEL_FORMANTS)


def test_asr_trains_above_chance():
    """ASR > 50% (chance = 1/8 = 12.5%)."""
    model, res = train_asr(n_per_vowel=40, n_steps=300)
    assert res["test_acc"] > 0.5


def test_recognize_phoneme():
    model, _ = train_asr(n_per_vowel=30, n_steps=200)
    # 'a' a F1=730, F2=1090 — le modèle devrait reconnaître 'a' ou un phonème proche
    pred = recognize_phoneme(model, 730, 1090)
    assert isinstance(pred, str) and len(pred) == 1
