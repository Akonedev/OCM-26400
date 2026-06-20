"""Tests TDD — récurrence fenêtrée profonde (OCM-26400).

Valide op_chain_gt (ground-truth op^k) : la récursion correcte du vérifieur.
recursive_decompose (block appliqué k fois) est validé empiriquement par
experiment_recursion (100% aux profondeurs 2-5).
"""
from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.experiment_recursion import op_chain_gt


def _v():
    return Verifier(SymbolicDict(n=P_MOD))


def test_op_chain_gt_depth2_matches_manual():
    """op_chain_gt([a,b,c]) == compose(compose(a,b),c)."""
    ver = _v()
    for (a, b, c) in [(0, 1, 2), (5, 5, 5), (10, 3, 7)]:
        assert op_chain_gt(ver, [a, b, c]) == ver.compose(ver.compose(a, b), c)


def test_op_chain_gt_depth4_recurses_left():
    """op^4 = op(op(op(op(a,b),c),d),e) — récursion strictement à gauche."""
    ver = _v()
    chain = [2, 7, 1, 9, 4]
    manual = ver.compose(ver.compose(ver.compose(ver.compose(2, 7), 1), 9), 4)
    assert op_chain_gt(ver, chain) == manual


def test_op_chain_gt_single_element_identity():
    """Chaîne de 1 élément -> l'élément lui-même (pas d'application)."""
    ver = _v()
    assert op_chain_gt(ver, [6]) == 6
