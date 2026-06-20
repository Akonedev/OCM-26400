"""Tests TDD — CompositionalVocabulary (scalabilité par composition, OCM-26400).

Démontre que l'espace adressable est EXPONENTIEL (P^L, >> slot 64-dim) et que le
retrieval distingue un lexicon composé.
"""
import torch

from ocm26400.learned_vocab import LearnedVocab
from ocm26400.compositional_vocab import CompositionalVocabulary


def _cv(P=12, max_len=3):
    prim = LearnedVocab(n=P, init="random", seed=0).freeze()
    return CompositionalVocabulary(prim, max_len=max_len)


def test_addressable_space_exponential():
    """addressable = P^max_len, exponentiel >> slot 64-dim."""
    cv = _cv(P=12, max_len=3)
    assert cv.addressable_space() == 12 ** 3          # 1728 adressables
    assert cv.addressable_space() > 64                 # >> slot ent


def test_word_vector_unit_norm():
    cv = _cv()
    v = cv.word_vector([0, 3, 7])
    assert abs(float(v.norm()) - 1.0) < 1e-5


def test_distinct_sequences_distinct_vectors():
    """Séquences de morphèmes différentes => vecteurs peu corrélés (distincts)."""
    cv = _cv(P=12, max_len=3)
    v1 = cv.word_vector([0, 1, 2])
    v2 = cv.word_vector([3, 4, 5])
    cos = float((v1 @ v2) / (v1.norm() * v2.norm() + 1e-8))
    assert cos < 0.5, f"séquences distinctes trop corrélées: cos={cos:.2f}"


def test_retrieve_roundtrip():
    """retrieve(word_vector(w)) retrouve w dans le lexicon."""
    cv = _cv(P=10, max_len=3)
    lex = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]
    M, _ = cv.build_index(lex)
    for w in lex:
        found, conf = cv.retrieve(cv.word_vector(w), M, lex, threshold=0.5)
        assert found == w, f"roundtrip échoué: {w} -> {found}"
        assert conf > 0.9


def test_retrieval_precision_on_lexicon():
    """Un lexicon de 150 mots composés : precision@1 élevée."""
    cv = _cv(P=8, max_len=3)
    torch.manual_seed(0)
    lex = []
    seen = set()
    while len(lex) < 150:
        w = tuple(torch.randint(0, 8, (3,)).tolist())
        if w not in seen:
            seen.add(w); lex.append(list(w))
    M, _ = cv.build_index(lex)
    correct = sum(1 for w in lex if cv.retrieve(cv.word_vector(w), M, lex, threshold=0.0)[0] == w)
    assert correct / len(lex) > 0.9, f"precision@1 trop basse: {correct}/{len(lex)}"


def test_retrieve_abstains_on_noise():
    """Requête aléatoire (hors lexicon) => abstention si seuil sérieux."""
    cv = _cv(P=10, max_len=3)
    lex = [[0, 1, 2], [3, 4, 5]]
    M, _ = cv.build_index(lex)
    torch.manual_seed(1)
    found, _ = cv.retrieve(torch.randn(64), M, lex, threshold=0.9)
    assert found is None     # abstention
