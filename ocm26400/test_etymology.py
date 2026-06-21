"""Tests étymologie / morphèmes / lexèmes (OCM-26400)."""
from ocm26400.etymology import etymology, etymological_family, morphemes, lexeme


def test_etymology_known():
    e = etymology("lumière")
    assert e["root"] == "luc-/lux-"
    assert "lumineux" in e["family"]


def test_etymological_family():
    fam = etymological_family("conduire")
    assert "éduquer" in fam


def test_morphemes_decomposition():
    m = morphemes("transporter")
    assert m["radical"] in ("port", "porter")
    assert any(p in ("trans",) for p in m["prefixes"])


def test_lexeme_forms():
    l = lexeme("porter")
    assert l["lexeme"] == "porter"
    assert l["n_forms"] > 1  # au moins le lemme + formes conjuguées


def test_etymology_unknown():
    e = etymology("xyzqwerty")
    assert e["root"] is None
