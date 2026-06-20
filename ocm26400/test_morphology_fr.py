"""Tests conjugaison française (OCM-26400) — audit C4."""
from ocm26400.morphology_fr import (
    conjugate, verify_conjugation, fr_conjugation_rules, coverage_report,
    G1_ENDINGS, G2_ENDINGS, IRREGULAR,
)


# ---- 1er groupe (régulier -er) ----

def test_g1_present():
    assert conjugate("parler", "présent", 0) == "parle"      # je
    assert conjugate("parler", "présent", 5) == "parlent"    # ils
    assert conjugate("chanter", "présent", 3) == "chantons"  # nous


def test_g1_futur_imparfait():
    assert conjugate("parler", "futur", 5) == "parleront"
    assert conjugate("parler", "imparfait", 0) == "parlais"
    assert conjugate("manger", "futur", 0) == "mangerai"


def test_g1_all_six_persons_present():
    forms = [conjugate("aimer", "présent", i) for i in range(6)]
    assert forms == ["aime", "aimes", "aime", "aimons", "aimez", "aiment"]


# ---- 2e groupe (-ir) ----

def test_g2_present():
    assert conjugate("finir", "présent", 0) == "finis"
    assert conjugate("finir", "présent", 3) == "finissons"   # -iss-
    assert conjugate("finir", "présent", 5) == "finissent"


def test_g2_futur():
    assert conjugate("réussir", "futur", 2) == "réussira"


# ---- irréguliers ----

def test_irregular_etre_avoir():
    assert conjugate("être", "présent", 5) == "sont"
    assert conjugate("être", "présent", 0) == "suis"
    assert conjugate("avoir", "présent", 5) == "ont"
    assert conjugate("aller", "futur", 0) == "irai"
    assert conjugate("faire", "présent", 5) == "font"


def test_irregular_all_present_etre():
    forms = [conjugate("être", "présent", i) for i in range(6)]
    assert forms == ["suis", "es", "est", "sommes", "êtes", "sont"]


# ---- verify (rejette le faux) ----

def test_verify_accepts_true_rejects_false():
    assert verify_conjugation(("parler", "présent", 0), "parle") is True
    assert verify_conjugation(("parler", "présent", 0), "parles") is False  # je parle pas parles
    assert verify_conjugation(("être", "présent", 5), "sont") is True
    assert verify_conjugation(("être", "présent", 5), "est") is False


def test_unknown_tense_returns_none():
    """Honnête : un temps non couvert → None (pas d'invention)."""
    assert conjugate("parler", "plus-que-parfait", 0) is None
    # verbe du 3e groupe NON mémorisé dans IRREGULAR → None (honnête, pas d'invention)
    assert conjugate("battre", "présent", 0) is None
    # mais un -er régulier (jamais codé) se conjugue (composition)
    assert conjugate("danser", "présent", 0) == "danse"


# ---- intégration RuleLibrary ----

def test_fr_rules_built():
    rules = fr_conjugation_rules()
    assert len(rules) >= 20          # 6 g1 + 6 g2 + irréguliers
    domains = {r.domain for r in rules}
    assert domains == {"grammar_fr"}


def test_coverage_report():
    cov = coverage_report()
    assert cov["g1_tenses"] == 6
    assert cov["g2_tenses"] == 6
    assert cov["irregular_verbs"] >= 10
    assert cov["irregular_forms"] > 100


def test_compositional_generalization():
    """N'IMPORTE quel verbe -er régulier (jamais vu) se conjugue (composition radical+terminaison)."""
    # verbes jamais codés explicitement
    assert conjugate("danser", "présent", 0) == "danse"
    assert conjugate("tomber", "futur", 3) == "tomberons"
    assert conjugate("écouter", "imparfait", 2) == "écoutait"
