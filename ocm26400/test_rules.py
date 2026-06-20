"""Tests TDD — bibliothèque de règles multi-domaines (OCM-26400).

Valide : règles exactes (math/physique/grammaire), compréhension (verify), routage par
domaine, génération par composition de règles comprises.
"""
from ocm26400.rules import RuleLibrary


def test_math_rules_apply_and_verify():
    lib = RuleLibrary.default(n=11)
    assert lib.apply("add", (2, 3)) == 5
    assert lib.verify("add", (2, 3), 5) is True          # compris
    assert lib.verify("add", (2, 3), 9) is False         # mal appliqué -> pas compris


def test_physics_rules():
    lib = RuleLibrary.default()
    assert lib.apply("force", (2, 3)) == 6               # F=ma
    assert lib.apply("velocity", (10, 2)) == 5.0          # v=d/t
    assert lib.apply("kinetic", (2, 3)) == 9.0            # ½mv² = 0.5*2*9
    assert lib.apply("momentum", (2, 3)) == 6             # p=mv


def test_grammar_rules():
    lib = RuleLibrary.default()
    assert lib.apply("past", ("walk",)) == "walked"
    assert lib.apply("plural", ("cat",)) == "cats"
    assert lib.apply("gerund", ("walk",)) == "walking"   # règle naive s+ing (verbe régulier)


def test_library_domains_and_routing():
    lib = RuleLibrary.default()
    assert len(lib.domains()) >= 13              # 13+ domaines (extensible)
    assert "math" in lib.domains() and "neuroscience" in lib.domains()
    assert len(lib.by_domain("math")) == 4               # add/mul/linop/neg
    assert len(lib.by_domain("physics")) == 4


def test_understands_all_rules():
    """Compréhension : toutes les applications vérifiées correctement."""
    lib = RuleLibrary.default()
    apps = [("add", (4, 5)), ("force", (2, 9)), ("past", ("jump",)), ("kinetic", (1, 2))]
    assert lib.understands_all(apps) is True
    # si on corrompt une règle (mal appliquer), la compréhension échoue
    bad = lib.verify("add", (4, 5), 999)
    assert bad is False


def test_compose_generation_math_and_grammar():
    """GÉNÉRATION par composition de règles comprises."""
    lib = RuleLibrary.default()
    # math (mod 11) : 4 -> add(.,3)=7 -> mul(.,2)=14%11=3
    chain = lib.compose([("add", (3,)), ("mul", (2,))], init=4)
    assert chain == [4, 7, 3]
    # grammaire : walk -> past -> plural
    gchain = lib.compose([("past", ()), ("plural", ())], init="walk")
    assert gchain == ["walk", "walked", "walkeds"]
