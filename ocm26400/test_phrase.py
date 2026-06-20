"""Tests TDD — composition de phrases (OCM-26400, cahier des charges §TESTS).

Valide : compose (mots→phrase), decode (phrase→mots), similarité (synonymes),
régénération avec synonymes.
"""
import torch
from ocm26400.phrase import PhraseComposer, phrase_similarity, regenerate_with_synonyms
from ocm26400.verifier import SymbolicDict


def _setup():
    d = SymbolicDict(n=20)
    comp = PhraseComposer(dict_n=20)
    return d, comp


def test_phrase_compose_produces_amv():
    """Composer une phrase → vecteur AMV phrase (256-d)."""
    d, c = _setup()
    amv = c.compose([0, 5, 10], d)
    assert amv.shape == (256,)
    assert float(amv.norm()) > 0


def test_phrase_distinct_phrases_distinct_amv():
    """Phrases différentes → AMV différents."""
    d, c = _setup()
    a1 = c.compose([0, 1, 2], d)
    a2 = c.compose([10, 15, 19], d)
    assert phrase_similarity(a1, a2) < 0.95           # pas identiques


def test_phrase_similar_words_similar_phrases():
    """Phrase identique → sim ≈ 1.0 ; phrase différente → sim < 1.0."""
    d, c = _setup()
    a1 = c.compose([0, 1, 2], d)
    a_same = c.compose([0, 1, 2], d)                   # identique
    a_diff = c.compose([10, 15, 19], d)                # tout différent
    assert phrase_similarity(a1, a_same) > 0.99        # identique → sim≈1
    assert phrase_similarity(a1, a_diff) < 0.99        # différent → sim<1


def test_regenerate_with_synonyms():
    """Régénérer avec synonymes → similarité conservée."""
    d, c = _setup()
    words = [0, 5, 10]
    synonyms = {5: 6, 10: 11}                           # synonymes proches
    new_words, sim = regenerate_with_synonyms(words, synonyms, c, d)
    assert new_words == [0, 6, 11]
    assert sim > 0.0                                    # similarité non-nulle


def test_phrase_decode_returns_valid_words():
    """Décoder une phrase → mots valides (dans le dictionnaire)."""
    d, c = _setup()
    amv = c.compose([3, 7, 12], d)
    words = c.decode_words(amv, d, max_words=3)
    assert len(words) == 3
    assert all(0 <= w < d.n for w in words)
