"""Tests TDD — sommeil / consolidation (extraction de règle, OCM-26400).

Valide l'extraction de règle depuis des faits épisodiques (r = αa+βb mod n) et la
consolidation (compression + généralisation). Spec 'phases de sommeil'.
"""
from ocm26400.verifier import P_MOD
from ocm26400.sleep import modinv, extract_rule, rule_predicts, consolidate, consolidation_stats


def _op(a, b, alpha=3, beta=5, n=P_MOD):
    return (alpha * a + beta * b) % n


def test_modinv_basic():
    assert modinv(3, 11) == 4          # 3*4=12≡1 mod 11
    assert modinv(0, 11) is None
    assert modinv(2, 4) is None        # gcd(2,4)=2 non inversible mod 4


def test_extract_rule_finds_op():
    """Faits issus de op=(3a+5b) mod 11 -> extraction retrouve (3,5)."""
    facts = [(a, b, _op(a, b)) for a in range(P_MOD) for b in range(P_MOD)]
    rule = extract_rule(facts[:20], P_MOD)         # 20 faits suffisent
    assert rule == (3, 5)


def test_extract_rule_generalizes_to_all_pairs():
    """La règle extraite de quelques faits prédit TOUTES les paires (généralisation)."""
    facts = [(a, b, _op(a, b)) for a in range(4) for b in range(4)]
    rule = extract_rule(facts, P_MOD)
    assert rule is not None
    # vérifie sur les 121 paires (dont beaucoup jamais vues dans facts)
    for a in range(P_MOD):
        for b in range(P_MOD):
            assert rule_predicts(rule, a, b, P_MOD) == _op(a, b)


def test_extract_rule_returns_none_on_inconsistent():
    """Faits ne suivant AUCUNE règle linéaire -> None."""
    facts = [(0, 0, 1), (1, 0, 2), (0, 1, 3), (1, 1, 5)]   # pas de (α,β) cohérent
    rule = extract_rule(facts, P_MOD)
    assert rule is None


class _FakeAgent:
    """Agent minimal avec une mémoire épisodique (pour tester consolidate)."""
    def __init__(self, memory):
        self.memory = memory


def test_consolidate_compresses_memory():
    """L'agent a appris N faits ; le sommeil les compacte en 1 règle (compression)."""
    mem = {(a, b): _op(a, b) for a in range(5) for b in range(5)}  # 25 faits
    agent = _FakeAgent(mem)
    rule = consolidate(agent, P_MOD)
    assert rule == (3, 5)
    stats = consolidation_stats(agent, rule, P_MOD)
    assert stats["rule_found"] is True
    assert stats["episodic_facts"] == 25
    assert stats["compressed_to"] == 1
    assert stats["compression_ratio"] == 25
    assert stats["generalizes_to_all_pairs"] is True


def test_consolidate_no_rule_when_inconsistent():
    """Mémoire incohérente -> pas de règle (pas de consolidation)."""
    mem = {(0, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 5}
    agent = _FakeAgent(mem)
    assert consolidate(agent, P_MOD) is None
