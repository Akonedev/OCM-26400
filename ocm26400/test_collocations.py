"""Tests collocations / locutions (OCM-26400) — audit L13."""
from ocm26400.collocations import detect_collocations, is_fixed_expression, detect_compounds


def test_collocation_verbe_nom():
    c = detect_collocations(["prendre", "une", "décision"])
    assert ("prendre", "décision", "verbe+nom") in c


def test_collocation_adj_nom():
    c = detect_collocations(["erreur", "grave"])
    assert any(t == "adj+nom" for _, _, t in c)


def test_fixed_expression():
    ok, sens = is_fixed_expression("il faut avoir raison")
    assert ok and "vrai" in sens.lower()


def test_compound_detection():
    comps = detect_compounds("je cuisine une pomme de terre")
    assert "pomme de terre" in comps


def test_no_collocation():
    assert detect_collocations(["le", "chat"]) == []
