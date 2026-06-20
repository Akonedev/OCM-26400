"""Tests TDD — alignement amodal (OCM-26400, concept_amodal).

Honnête : on teste l'alignement contrastif + l'ancrage AMV (terme déféré par le
verdict, ajouté maintenant que P2/LearnedVocab existe). Les vues sont des encodeurs
simulés (placeholder modalités) — on valide la math, pas du vrai multimodal.
"""
import torch

from ocm26400.learned_vocab import LearnedVocab
from ocm26400.concept_amodal import (
    ModalityEncoder, amodal_align_loss, train_amodal,
    cross_view_retrieval, anchor_decode_accuracy,
)

N = 32
K = 3


def _setup():
    vocab = LearnedVocab(n=N, init="random", seed=0).freeze()
    encoders = [ModalityEncoder(N, seed=s) for s in range(K)]
    return vocab, encoders


def test_modality_encoder_shape():
    enc = ModalityEncoder(N)
    ids = torch.tensor([0, 5, 31])
    v = enc(ids)
    assert v.shape == (3, 64)


def test_amodal_loss_returns_consist_and_anchor():
    vocab, encs = _setup()
    ids = torch.arange(N)
    views = [enc(ids) for enc in encs]
    total, parts = amodal_align_loss(views, vocab, ids)
    assert total.requires_grad
    assert "consist" in parts and "anchor" in parts
    assert parts["anchor"] > 0     # vues init non ancrées => ancrage > 0


def test_views_start_misaligned():
    """Avant entraînement, retrieval@1 cross-vue ~ aléatoire (1/N)."""
    vocab, encs = _setup()
    r = cross_view_retrieval(encs, N)
    assert r < 0.2, f"vues déjà alignées à l'init? retrieval={r}"   # 1/N=0.031


def test_alignment_improves_with_training():
    """Après entraînement, retrieval@1 cross-vue monte fortement (alignement amodal)."""
    vocab, encs = _setup()
    before = cross_view_retrieval(encs, N)
    train_amodal(vocab, encs, N, n_steps=400)
    after = cross_view_retrieval(encs, N)
    assert after > before + 0.3, f"alignement n'améliore pas: {before:.2f}->{after:.2f}"
    assert after > 0.7, f"retrieval trop bas après entraînement: {after:.2f}"


def test_anchor_decode_recovers_concept():
    """L'ancrage AMV ramène chaque vue dans le dictionnaire : decode(vue(C)) -> C."""
    vocab, encs = _setup()
    train_amodal(vocab, encs, N, n_steps=400)
    acc = anchor_decode_accuracy(encs, vocab, N)
    assert acc > 0.7, f"ancrage insuffisant: decode recouvre {acc*100:.0f}% des concepts"
