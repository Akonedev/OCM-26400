"""Tests TDD — alignement amodal sur VUES LINGUISTIQUES RÉELLES (OCM-26400).

Valide l'amodal sur de vrais mots (real_vocab_dataset) avec 4 modalités réelles
(texte/morphologie/phonologie/sémantique). Remplace les vues simulées.
"""
import torch

from ocm26400.real_linguistic import (
    load_real_words, view_bag, RealViewEncoder, MODALITIES,
    build_views, amodal_real_loss, train_real_amodal, cross_view_retrieval_real,
)


def _words(n=60):
    return load_real_words(limit=n)


def test_load_real_words_has_features():
    """Les vrais mots ont les features multi-vues attendues."""
    w = load_real_words(limit=5)
    assert "word" in w[0] and "plural" in w[0] and "phoneme_pattern" in w[0]
    assert "category" in w[0]


def test_view_bag_distinct_modalities():
    """Les 4 modalités donnent des vecteurs différents pour un même mot."""
    w = _words()[0]
    bags = [view_bag(w, m) for m in MODALITIES]
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            assert not torch.allclose(bags[i], bags[j])


def test_view_bag_distinct_words():
    """Même modalité, mots différents => vecteurs différents."""
    words = _words(20)
    t1 = view_bag(words[0], "texte")
    t2 = view_bag(words[5], "texte")
    assert not torch.allclose(t1, t2)


def test_amodal_real_loss_differentiable():
    words = _words(20)
    encoders = {m: RealViewEncoder(seed=i) for i, m in enumerate(MODALITIES)}
    views = build_views(words, encoders)
    loss = amodal_real_loss(views)
    assert loss.requires_grad
    loss.backward()
    # les projections ont reçu du gradient
    assert encoders["texte"].mlp[0].weight.grad is not None


def test_alignment_improves_with_training():
    """Après entraînement, retrieval@1 cross-vue monte (vues réelles alignées)."""
    words = _words(80)
    encoders = {m: RealViewEncoder(seed=i) for i, m in enumerate(MODALITIES)}
    before = cross_view_retrieval_real(words, encoders)
    train_real_amodal(words, encoders, n_steps=400)
    after = cross_view_retrieval_real(words, encoders)
    assert after > before, f"l'alignement n'améliore pas: {before:.3f}->{after:.3f}"
    assert after > 0.5, f"retrieval trop bas après entraînement: {after:.3f}"
