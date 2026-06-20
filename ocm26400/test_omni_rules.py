"""Tests TDD — OmniModel multi-règles conjoint + génération inter-règles (OCM-26400).

Valide : règles hétérogènes (add/mul/linop), composition inter-règles (gt), et qu'UN
noyau entraîné conjointement COMPREND chaque règle + GÉNÈRE des chaînes mixtes neuves.
"""
import random
import torch

from ocm26400.verifier import SymbolicDict, P_MOD
from ocm26400.omni_rules import (
    RULES, RULE_NAMES, train_omni_rules, comprehend, inter_rule_gt, generate_chain,
)


def test_rules_are_distinct():
    """add/mul/linop donnent des résultats différents (mul bilinéaire = vraiment distinct)."""
    a, b = 4, 7
    vals = {name: RULES[name](a, b) for name in RULE_NAMES}
    assert len(set(vals.values())) >= 2          # au moins 2 règles distinctes sur ce couple


def test_inter_rule_gt_composes_functions():
    """Vérité : compose les fonctions règles. [add(a,b), mul(.,c)] = mul(add(a,b), c)."""
    a, c = 3, 6
    chain = [("add", 5), ("mul", c)]
    assert inter_rule_gt(chain, a) == RULES["mul"](RULES["add"](a, 5), c)


def test_omni_rules_comprehend_and_generate():
    """UN noyau entraîné conjointement : COMPREND chaque règle + GÉNÈRE chaînes inter-règles neuves."""
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=P_MOD)
    blk = train_omni_rules(d, n_steps=2000)
    # 1) compréhension : chaque règle grokkée
    comp = comprehend(blk, d, n_test=40)
    for name, acc in comp.items():
        assert acc > 0.9, f"règle {name} non comprise: {acc:.2f}"
    # 2) GÉNÉRATION inter-règles : chaînes mixtes neuves (add puis mul puis linop)
    chains = [[("add", random.randrange(P_MOD)), ("mul", random.randrange(P_MOD)),
               ("linop", random.randrange(P_MOD))] for _ in range(60)]
    ok = 0
    for c in chains:
        init = random.randrange(P_MOD)
        ok += (generate_chain(blk, d, c, init) == inter_rule_gt(c, init))
    assert ok / len(chains) > 0.85, f"génération inter-règles trop basse: {ok}/{len(chains)}"
