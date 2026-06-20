"""Tests TDD — MorphologyVerifier (dispatch op_id, OCM-26400)."""
import torch

from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.morphology import (
    MorphologyVerifier, CONJUGATE_PAST, CONJUGATE_GERUND, CONJUGATE_THIRD,
)


def _make():
    """Verbe -> forme par tense. rules[op_id](verb, b) = forme_id."""
    d = SymbolicDict(n=30)
    # formes : 6 verbes x 3 temps = 18 formes, ids 6..23 (verbes 0..5)
    def past(a, b):     return 6 + a                  # walked, talked, ...
    def gerund(a, b):   return 12 + a                 # walking, ...
    def third(a, b):    return 18 + a                 # walks, ...
    rules = [past, gerund, third]                     # op_id 0, 1, 2
    return d, MorphologyVerifier(d, rules)


def test_dispatches_by_op_id():
    """compose(a,b,0) != compose(a,b,1) != compose(a,b,2) : le dispatch sélectionne la règle."""
    d, ver = _make()
    a = 2
    assert ver.compose(a, 0, op_id=CONJUGATE_PAST) == 8      # 6+2
    assert ver.compose(a, 0, op_id=CONJUGATE_GERUND) == 14   # 12+2
    assert ver.compose(a, 0, op_id=CONJUGATE_THIRD) == 20    # 18+2
    assert ver.compose(a, 0, 0) != ver.compose(a, 0, 1)


def test_V_accepts_op_id_range():
    """V légal pour op_id dans [0, n_ops), faux sinon."""
    d, ver = _make()
    assert ver.V(0, 0, 0) is True
    assert ver.V(0, 0, 2) is True
    assert ver.V(0, 0, 3) is False     # hors range (n_ops=3)
    assert ver.V(0, 0, 99) is False


def test_is_valid_intermediate_uses_dispatch():
    """is_valid_intermediate vérifie la bonne forme SELON op_id."""
    d, ver = _make()
    assert ver.is_valid_intermediate(2, 0, 8, op_id=CONJUGATE_PAST) is True
    assert ver.is_valid_intermediate(2, 0, 14, op_id=CONJUGATE_PAST) is False   # 14 = gerund, pas past
    assert ver.is_valid_intermediate(2, 0, 14, op_id=CONJUGATE_GERUND) is True


def test_base_verifier_compose_fn_contract_preserved():
    """MorphologyVerifier ne casse PAS la base : Verifier(compose_fn) ignore op_id."""
    d = SymbolicDict(n=20)
    base = Verifier(d, compose_fn=lambda a, b: (a + b) % 20)
    # le chemin compose_fn ignore op_id (contract test_contracts.py)
    assert base.compose(2, 5, op_id=0) == base.compose(2, 5, op_id=7) == 7


def test_morphology_verifier_out_of_range_is_noop():
    """op_id hors range -> no-op (retourne a), pas de crash."""
    d, ver = _make()
    assert ver.compose(2, 0, op_id=99) == 2     # no-op
