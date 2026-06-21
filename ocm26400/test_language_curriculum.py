"""Tests curriculum de langage (OCM-26400)."""
from ocm26400.language_curriculum import (
    run_language_curriculum, _eval_conjugation_rule, _eval_generalization_to_unseen,
)


def test_conjugation_rule_grokks():
    """Une règle de conjugaison (groupe 1) grok à 1.0 (symbolique, exact)."""
    tr, te = _eval_conjugation_rule("présent", group=1)
    assert te >= 0.99


def test_generalization_to_unseen():
    """Crown-jewel linguistique : verbes inédits (-er) se conjuguent (composition)."""
    gen = _eval_generalization_to_unseen("présent")
    assert gen >= 0.95     # règle régulière → 100% sur verbes inédits


def test_curriculum_runs():
    rep = run_language_curriculum()
    assert rep["n_rules"] > 0
    assert rep["n_grokked"] >= 3
    assert all(v >= 0.9 for v in rep["generalization_unseen"].values())
