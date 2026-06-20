"""Tests NLP (traduction/sentiment/résumé) — OCM-26400."""
from ocm26400.nlp_tools import translate, translate_word, sentiment, summarize


def test_translate_word_known():
    assert translate_word("chat") == "cat"
    assert translate_word("cat", to_en=False) == "chat"


def test_translate_word_unknown_kept():
    """Mot inconnu → gardé tel quel (abstention honnête)."""
    assert translate_word("xyzqwerty") == "xyzqwerty"


def test_translate_sentence():
    # mots à forme exacte dans le dictionnaire
    out = translate("le grand chat")
    assert "cat" in out and "big" in out


def test_sentiment_positive():
    s = sentiment("c'est génial et merveilleux, super")
    assert s["label"] == "positif" and s["score"] > 0


def test_sentiment_negative():
    s = sentiment("c'est terrible et horrible")
    assert s["label"] == "négatif" and s["score"] < 0


def test_sentiment_neutral():
    s = sentiment("le chat dort sur le canapé")
    assert s["label"] == "neutre" and s["score"] == 0.0


def test_summarize_shorter():
    txt = ("Le chat dort. Le chien mange. L'oiseau vole. Le poisson nage. " * 2)
    summ = summarize(txt, 2)
    assert len(summ) < len(txt)


def test_summarize_preserves_short():
    txt = "Une seule phrase ici."
    assert summarize(txt, 2) == txt
