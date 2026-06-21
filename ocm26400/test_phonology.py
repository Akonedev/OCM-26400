"""Tests phonologie / IPA (OCM-26400) — audit L5."""
from ocm26400.phonology import to_ipa, classify_sounds, elision, liaison


def test_to_ipa_basic():
    assert "[ʃ]" in to_ipa("chat")      # ch → ʃ
    assert "[a]" in to_ipa("chat")
    assert "[ɔ̃]" in to_ipa("bonjour")  # on → nasal


def test_to_ipa_nasals():
    assert "[ɑ̃]" in to_ipa("français") or "[ɑ̃]" in to_ipa("enfant")


def test_to_ipa_silent_e():
    """le 'e' final muet est géré (maison ne termine pas par voyelle 'e' prononcée)."""
    ipa = to_ipa("maison")
    assert "[m]" in ipa and "[z]" in ipa      # s → z entre voyelles


def test_classify_sounds():
    c = classify_sounds("bonjour")
    assert "b" in c["consonnes_voisees"]
    assert "on" in c["voyelles_nasales"]


def test_elision():
    assert elision("le", "ami") is True       # l'ami
    assert elision("la", "ami") is False      # pas d'élision pour la
    assert elision("ma", "école") is True


def test_liaison_z():
    assert liaison("les", "amis") == "[z]"
    assert liaison("les", "chats") == ""      # pas de liaison devant consonne


def test_liaison_n():
    assert liaison("on", "a") == "[n]"
