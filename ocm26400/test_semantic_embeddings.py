"""Tests embeddings sémantiques PPMI+SVD (OCM-26400) — audit H17."""
import numpy as np
from ocm26400.semantic_embeddings import (
    SemanticEmbeddings, subwords, build_cooccurrence, ppmi,
)

WORDS = ["run", "running", "runner", "runs", "rerun", "chat", "chats", "chatter",
         "happy", "happiness", "unhappy", "lion", "tiger", "wolf", "zebra",
         "computer", "computing", "compute"]


def test_subwords_bag():
    sw = subwords("run")
    assert "run" in sw and "#run#" in sw
    assert any(s == "#ru" or s == "run" for s in sw)


def test_cooccurrence_shape():
    wv, sv, M = build_cooccurrence(WORDS)
    assert M.shape[0] == len(set(WORDS))
    assert M.shape[1] == len(sv)
    assert M.sum() > 0


def test_ppmi_nonneg_and_zeroed():
    _, _, M = build_cooccurrence(WORDS)
    P = ppmi(M)
    assert (P >= 0).all()              # PPMI = max(0, pmi)
    # les non-co-occurrences restent 0
    assert (P == 0).any()


def test_related_words_more_similar_than_unrelated():
    """LE test : la similarité reflète la structure morphologique (vs hash)."""
    emb = SemanticEmbeddings(WORDS, dim=32)
    # apparentés (partagent subwords) > sans rapport
    assert emb.similarity("run", "running") > emb.similarity("run", "zebra")
    assert emb.similarity("chat", "chats") > emb.similarity("chat", "tiger")
    assert emb.similarity("happy", "happiness") > emb.similarity("happy", "lion")
    assert emb.similarity("computer", "computing") > emb.similarity("computer", "wolf")


def test_unrelated_words_zero_similarity():
    """Pas de faux-positif : mots sans subword commun → ~0 (corrige le hash)."""
    emb = SemanticEmbeddings(WORDS, dim=32)
    assert abs(emb.similarity("run", "zebra")) < 0.01
    assert abs(emb.similarity("computer", "lion")) < 0.01


def test_nearest_returns_related():
    emb = SemanticEmbeddings(WORDS, dim=32)
    near = emb.nearest("run", k=3)
    assert len(near) == 3
    near_words = [w for w, _ in near]
    # les plus proches de run doivent être des variants (pas zebra/lion)
    assert "zebra" not in near_words and "lion" not in near_words
    assert any("run" in w for w in near_words)   # running/runner/runs/rerun


def test_unknown_word_zero_vector():
    emb = SemanticEmbeddings(WORDS, dim=32)
    v = emb.word_vector("xyzqwerty")
    assert np.linalg.norm(v) < 1e-9     # inconnu → vecteur nul (honnête)


def test_word_vector_dimension():
    emb = SemanticEmbeddings(WORDS, dim=32)
    v = emb.word_vector("run")
    assert v.shape[0] >= 1 and v.shape[0] <= 32
