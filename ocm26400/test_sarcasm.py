"""Tests sarcasme (OCM-26400) — audit M3."""
from ocm26400.sarcasm import detect_sarcasm


def test_sarcastic_positive_on_negative():
    r = detect_sarcasm("Quelle magnifique journée pour une panne !")
    assert r["label"] == "sarcastique"
    assert r["score"] >= 0.5


def test_literal_no_sarcasm():
    r = detect_sarcasm("Le chat dort sur le canapé")
    assert r["label"] == "littéral"
    assert r["score"] < 0.25


def test_sarcasm_has_reasons():
    r = detect_sarcasm("Bravo, génial, encore un échec total !")
    assert r["label"] == "sarcastique"
    assert len(r["reasons"]) >= 1


def test_sarcasm_quotes():
    r = detect_sarcasm("Son « aide » a tout cassé.")
    assert r["score"] > 0      # guillemets détectés


def test_literal_positive():
    """Sentiment positif SANS contexte négatif = pas sarcastique."""
    r = detect_sarcasm("Ce film est vraiment merveilleux et joyeux")
    assert r["label"] == "littéral"
