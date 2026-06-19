"""Tests RED — Contrats partagés OCM (juge: fixer UNE FOIS avant composition).

(1) compose(a,b,op_id=0) + is_valid_intermediate(a,b,m,op_id=0) : dispatch par op_id,
    rétrocompatible (appels 2-arg existants inchangés).
(2) Partition meta(64) : meta[0]=confidence LSRA, meta[1]=source/bridge confidence,
    meta[2]=consist score cross-modale. Évite la contention confirmée par le juge.
"""
import pytest
import torch
from ocm26400.amv import AMVVector
from ocm26400.verifier import SymbolicDict, Verifier


@pytest.fixture
def dv():
    return SymbolicDict(), Verifier(SymbolicDict())


# ── (1) contrats compose / is_valid_intermediate avec op_id ──

def test_compose_accepts_op_id_default_zero(dv):
    d, ver = dv
    assert ver.compose(2, 5) == ver.compose(2, 5, op_id=0)  # rétrocompatible


def test_V_accepts_op_id(dv):
    d, ver = dv
    assert ver.V(2, 5, op_id=0) is True
    assert ver.V(2, 5, op_id=99) is False  # op inconnu


def test_is_valid_intermediate_accepts_op_id(dv):
    d, ver = dv
    m = ver.compose(2, 5)
    assert ver.is_valid_intermediate(2, 5, m) is True                 # rétrocompatible
    assert ver.is_valid_intermediate(2, 5, m, op_id=0) is True
    assert ver.is_valid_intermediate(2, 5, (m + 1) % 11, op_id=0) is False


def test_compose_fn_path_ignores_op_id_safely():
    # un Verifier pluggable (morphologie) garde compose_fn(a,b) intact
    d = SymbolicDict(n=20)
    fn = lambda a, b: (a + b) % 20
    ver = Verifier(d, compose_fn=fn, n_ops=1)
    assert ver.compose(3, 4, op_id=0) == 7


# ── (2) partition meta(64) ──

def test_amv_meta_partitions_three_roles():
    v = AMVVector(torch.zeros(256))
    v.meta[0] = 3.0   # LSRA confidence pre-sigmoid
    v.meta[1] = 2.0   # source/bridge confidence
    v.meta[2] = 0.8   # consist score
    assert torch.allclose(v.confidence(), torch.sigmoid(torch.tensor(3.0)))     # meta[0]
    assert torch.allclose(v.source_confidence(), torch.sigmoid(torch.tensor(2.0)))  # meta[1]
    assert v.consist_score().item() == pytest.approx(0.8)                       # meta[2] brute


def test_meta_roles_are_distinct_indices():
    v = AMVVector(torch.zeros(256))
    # écrire dans un rôle ne pollue pas les autres
    v.meta[0] = 1.0
    assert v.meta[1].item() == 0.0
    assert v.meta[2].item() == 0.0
