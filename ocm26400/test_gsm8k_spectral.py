"""Tests NL→CoT spectral pur (OCM-26400) — MODEL UNIFIÉ, pas de transformer."""
import torch
from ocm26400.gsm8k_spectral import SpectralNLCoT, train_spectral_cot, predict_spectral
from ocm26400.gsm8k_seq2seq import ACTIONS


def test_model_no_attention():
    """LE test : le modèle ne contient AUCUN transformer/attention — pure spectral."""
    m = SpectralNLCoT(50, d_model=64)
    # aucune couche attention/transformer
    has_attn = any("Attention" in type(mod).__name__ or "Transformer" in type(mod).__name__
                   for mod in m.modules())
    assert not has_attn, "le modèle contient de l'attention/transformer (interdit)"
    # le noyau est bien SpectralCoreBlock
    from ocm26400.spectral_core import SpectralCoreBlock
    assert isinstance(m.encoder, SpectralCoreBlock)
    assert isinstance(m.decoder, SpectralCoreBlock)


def test_forward_shape():
    m = SpectralNLCoT(50, d_model=64, seq_len=20)
    src = torch.randint(0, 50, (2, 20))
    tgt = torch.randint(0, len(ACTIONS), (2, 8))
    out = m(src, tgt)
    assert out.shape == (2, 8, len(ACTIONS))


def test_train_predict():
    m, vocab = train_spectral_cot(n_train=50, n_steps=10, device="cpu")
    pred = predict_spectral(m, vocab, "4 boxes with 6 apples each", "cpu")
    assert pred is None or isinstance(pred, float)
