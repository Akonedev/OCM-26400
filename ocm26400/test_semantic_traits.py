"""Tests sèmes / traits sémantiques (OCM-26400)."""
from ocm26400.semantic_traits import (
    semantic_traits, disambiguate, are_synonyms, are_antonyms, semantic_similarity,
)


def test_semantic_traits():
    assert "félin" in semantic_traits("chat")
    assert "animal" in semantic_traits("chien")


def test_disambiguate_vol():
    sense, _ = disambiguate("vol", "l'oiseau vole")
    assert "voler" in sense or "déplacement" in str(_)


def test_disambiguate_mine():
    sense, _ = disambiguate("mine", "extraction de charbon dans la mine")
    assert "exploitation" in sense or "extraction" in str(_)


def test_synonyms():
    assert semantic_similarity("chat", "lion") > 0.3  # félin commun
    assert not are_synonyms("chat", "voiture")


def test_antonyms():
    assert are_antonyms("amour", "haine")
    assert are_antonyms("joie", "tristesse")
    assert not are_antonyms("chat", "chien")


def test_similarity():
    assert semantic_similarity("voiture", "vélo") > 0.2
    assert semantic_similarity("chat", "voiture") < 0.2
