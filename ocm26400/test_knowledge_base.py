"""Tests TDD — KnowledgeBase (retrieval + abstention, OCM-26400).

Valide la brique 'recherche dans la base de connaissance' + abstention ('je ne sais
pas' -> mode apprentissage). Intègre P2 LearnedVocab + P3 abstention.
"""
import torch

from ocm26400.learned_vocab import LearnedVocab
from ocm26400.knowledge_base import KnowledgeBase


def _kb(n=40, threshold=0.5):
    vocab = LearnedVocab(n=n, init="ortho", seed=0).freeze()
    return KnowledgeBase(vocab, threshold=threshold)


def test_retrieve_finds_stored_concept():
    """Une requête = canonique du concept i retrouve i (confiance élevée)."""
    kb = _kb()
    for i in [0, 10, 39]:
        idx, conf = kb.retrieve(kb.vocab.canonical(i))
        assert idx == i
        assert conf > 0.9


def test_retrieve_abstains_on_unknown():
    """Une requête aléatoire (sans rapport avec E) => abstention (None)."""
    kb = _kb(threshold=0.5)
    torch.manual_seed(1)
    abstained = 0
    for _ in range(20):
        idx, _ = kb.retrieve(torch.randn(64))
        abstained += (idx is None)
    assert abstained >= 18, f"trop de requêtes aléatoires acceptées: {20-abstained}/20"


def test_threshold_controls_abstention():
    """Seuil bas => retourne le plus proche ; seuil haut => abstention."""
    kb = _kb()
    q = kb.vocab.canonical(5) * 0.5 + kb.vocab.canonical(6) * 0.5   # entre 5 et 6
    idx_lo, _ = kb.retrieve(q, threshold=0.0)
    assert idx_lo is not None                  # seuil bas => répond
    idx_hi, _ = kb.retrieve(q, threshold=0.99)
    assert idx_hi is None                      # seuil haut => abstention


def test_store_then_answer_returns_value():
    """store (apprentissage) => answer renvoie la valeur stockée."""
    kb = _kb()
    kb.store(7, "Paris est la capitale de la France")
    val, conf = kb.answer(kb.vocab.canonical(7))
    assert val == "Paris est la capitale de la France"
    assert conf > 0.9


def test_answer_abstention_returns_none():
    """answer sur OOD => (None, conf basse) = signal 'je ne sais pas'."""
    kb = _kb(threshold=0.5)
    torch.manual_seed(2)
    val, conf = kb.answer(torch.randn(64))
    assert val is None
    assert conf < 0.5


def test_knows_distinguishes_known_from_unknown():
    """knows() = True pour un concept du vocabulaire, False pour OOD."""
    kb = _kb(threshold=0.5)
    assert kb.knows(kb.vocab.canonical(3)) is True
    torch.manual_seed(3)
    assert kb.knows(torch.randn(64)) is False
