"""Tests abstraction / catégorisation (OCM-26400)."""
from ocm26400.abstraction import categorize, abstract, hierarchy, INSTANCE_TRAITS


def test_categorize_animals():
    cat, conf = categorize("chien")
    assert cat in ("mammifère", "animal")
    assert conf > 0.3


def test_categorize_vehicle():
    cat, _ = categorize("voiture")
    assert cat == "véhicule"


def test_categorize_unknown():
    cat, conf = categorize("xyzqwerty")
    assert cat == "inconnu" and conf == 0.0


def test_abstract_animals():
    result = abstract(["chien", "chat", "aigle"])
    assert "vivant" in result["traits_communs"]
    assert result["concept"] in ("animal", "mammifère", "oiseau")


def test_abstract_common_traits():
    result = abstract(["chien", "chat"])
    assert "poils" in result["traits_communs"]
    assert "allaiter" in result["traits_communs"]


def test_hierarchy():
    h = hierarchy("mammifère")
    assert "animal" in h and "mammifère" in h
