"""Tests compétence multi-domaine (OCM-26400)."""
from ocm26400.domain_trainer import (
    evaluate_rule, evaluate_all_domains, cross_domain_chains, reasoning_bench_aime,
    run_all, _sample_inputs,
)
from ocm26400.rules import RuleLibrary


def test_sample_inputs_arity():
    rl = RuleLibrary.default()
    add = rl.rules["add"]
    inp = _sample_inputs(add, n=5, seed=0)
    assert len(inp) == 5
    assert all(len(x) == 2 for x in inp)        # arity 2


def test_evaluate_rule_mastered():
    rl = RuleLibrary.default()
    res = evaluate_rule(rl.rules["add"], n_samples=8)
    assert res["mastered"] is True
    assert res["apply_acc"] == 1.0
    assert res["verify_true_acc"] == 1.0
    assert res["verify_false_acc"] == 1.0      # rejette le faux = connaît la règle


def test_all_domains_full_mastery():
    """LE test clé : le modèle maîtrise TOUS les domaines (refute 'not trained')."""
    res = evaluate_all_domains(n_samples=6)
    assert res["n_rules"] == 91
    assert res["n_mastered"] == 91             # 100% des règles
    assert res["n_domains"] == 30
    assert res["n_domains_full_mastery"] == 30 # 100% des domaines
    assert res["domain_coverage"] == 1.0


def test_cross_domain_coherence():
    res = cross_domain_chains(15)
    assert res["n_chains"] == 15
    assert res["cross_domain_coherence_rate"] == 1.0   # 100% cohérent
    # chaque chaîne mélange 2 domaines différents
    for c in res["chains"]:
        assert len(set(c["domains"])) == 2


def test_aime_reasoning_perfect():
    """Raisonnement compositionnel profondeur 3 = 100% (crown-jewel étendu)."""
    res = reasoning_bench_aime(40)
    assert res["accuracy"] == 1.0
    assert res["depth"] == 3
    assert res["n_correct"] == 40


def test_run_all_writes_results():
    import json, os
    rep = run_all()
    path = os.path.join(os.path.dirname(__file__), "domain_competence_results.json")
    assert os.path.exists(path)
    with open(path) as f:
        d = json.load(f)
    assert d["domain_competence"]["n_mastered"] == 91
    assert d["aime_reasoning"]["accuracy"] == 1.0
