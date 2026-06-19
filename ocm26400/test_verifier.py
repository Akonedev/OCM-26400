"""Tests RED — Vérifieur symbolique + dictionnaire de primitives.

Spec Besoins_Maths.md §2.2: V(ent, prop, op) -> 1 si operation legale, 0 sinon.
Tache compositionnelle de validation (crown jewel): op(a,b) = (3a + 5b) mod 11,
NON-commutative et NON-associative -> la decomposition (calculer l'intermediaire)
est necessaire pour generaliser sur des triples jamais vus.
"""
import pytest
import torch
from ocm26400.amv import AMVVector
from ocm26400.verifier import SymbolicDict, Verifier, P_MOD, A_COEF, B_COEF


def test_dict_has_p_mod_primitives():
    d = SymbolicDict()
    assert d.n == P_MOD  # 11 primitives (Z_11)
    assert d.canonical(0).shape == (64,)


def test_dict_canonical_is_one_hot_in_first_n_dims():
    d = SymbolicDict()
    v0 = d.canonical(3)
    assert v0[3] == 1.0
    assert v0[:3].sum() == 0.0
    assert v0[4:11].sum() == 0.0  # seul l'index 3 est actif


def test_dict_decode_recognizes_canonical_primitive():
    d = SymbolicDict()
    idx, valid = d.decode(d.canonical(7))
    assert idx == 7
    assert valid is True


def test_dict_decode_rejects_garbage():
    d = SymbolicDict()
    garbage = torch.ones(64) * 0.5  # pas one-hot
    idx, valid = d.decode(garbage)
    assert valid is False


def test_verifier_compose_is_non_commutative():
    v = Verifier(SymbolicDict())
    # op(a,b) = (3a + 5b) mod 11
    assert v.compose(2, 5) == (A_COEF * 2 + B_COEF * 5) % P_MOD
    # non-commutatif : op(2,5) != op(5,2) en general
    assert v.compose(2, 5) != v.compose(5, 2)


def test_verifier_compose_is_non_associative():
    v = Verifier(SymbolicDict())
    # (a o b) o c != a o (b o c) en general
    left = v.compose(v.compose(2, 5), 7)
    right = v.compose(2, v.compose(5, 7))
    assert left != right


def test_verifier_V_legal_step_returns_true():
    # V(ent, prop, op) : ent & prop valides + op connu -> legal
    v = Verifier(SymbolicDict())
    assert v.V(d_ent=0, d_prop=1, op_id=0) is True


def test_verifier_V_rejects_invalid_primitive():
    v = Verifier(SymbolicDict())
    # ent hors dictionnaire (index trop grand) -> illegal
    assert v.V(d_ent=999, d_prop=1, op_id=0) is False


def test_verifier_check_intermediate_composition():
    # etant donne a,b,c et un intermediaire propose m: m == op(a,b) ?
    v = Verifier(SymbolicDict())
    a, b, c = 2, 5, 7
    m_correct = v.compose(a, b)
    m_wrong = (m_correct + 1) % P_MOD
    assert v.is_valid_intermediate(a, b, m_correct) is True
    assert v.is_valid_intermediate(a, b, m_wrong) is False
