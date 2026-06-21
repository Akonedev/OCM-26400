"""Tests concept grok — langage comme arithmétique (OCM-26400)."""
import torch
import pytest
from ocm26400.concept_grok import ConceptVocab, ConceptGrokModel, grok_concept_association


def test_concept_vocab_assigns_ids():
    v = ConceptVocab(vocab_size=100)
    id1 = v.add_word("cat")
    id2 = v.add_word("cats")
    assert id1 == 0 and id2 == 1
    assert v.word_to_id["cat"] == 0


def test_concept_vocab_dense_embeddings():
    v = ConceptVocab(vocab_size=100)
    v.add_word("hello")
    v.build_embeddings()
    emb = v.get_embedding(0)
    assert emb.shape[0] == 64  # LearnedVocab dense (PART)


def test_concept_vocab_encode_text():
    v = ConceptVocab(vocab_size=100)
    seq, ids = v.encode_text("the cat sat")
    assert seq.shape == (3, 256)  # 3 mots, AMV-256
    assert len(ids) == 3


def test_concept_grok_model_forward():
    m = ConceptGrokModel()
    x = torch.randn(2, 5, 256)  # batch=2, seq=5, d_model=256
    out = m(x)
    assert out.shape == (2, 5, 256)


def test_concept_grok_model_no_transformer():
    m = ConceptGrokModel()
    from ocm26400.spectral_core import SpectralCoreBlock
    assert isinstance(m.core, SpectralCoreBlock)
    has_tf = any("Transformer" in type(mod).__name__ or "Attention" in type(mod).__name__
                 for mod in m.modules())
    assert not has_tf


def test_grok_concept_association():
    """LE test : le SpectralCoreBlock grok des associations concept→concept."""
    pairs = [("bell", "bells"), ("cat", "cats"), ("dog", "dogs"), ("run", "runs")]
    model, res = grok_concept_association(pairs, n_steps=500)
    assert res["grok_acc"] >= 0.5, f"concept grok should work, got {res['grok_acc']}"
    assert res["loss"] == "1-cos (crown-jewel), PAS cross-entropy"
