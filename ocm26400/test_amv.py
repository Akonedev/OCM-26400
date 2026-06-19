"""Tests RED — AMV-256 (Amodal Mentalese Vector).

Spec Besoins_Maths.md §1:
  v = [v_ent(64) || v_prop(64) || v_op(64) || v_meta(64)]  in R^256
"""
import pytest
import torch
from ocm26400.amv import AMVVector, D_MODEL, PART


def test_amv_zeros_has_full_256_shape():
    v = AMVVector.zeros()
    assert v.tensor.shape == (D_MODEL,)
    assert D_MODEL == 256


def test_amv_has_four_partitions_of_64():
    v = AMVVector.zeros()
    assert PART == 64
    assert v.ent.shape == (64,)
    assert v.prop.shape == (64,)
    assert v.op.shape == (64,)
    assert v.meta.shape == (64,)


def test_partitions_are_contiguous_slices_of_tensor():
    # ent = [0:64], prop = [64:128], op = [128:192], meta = [192:256]
    v = AMVVector.zeros()
    v.ent.fill_(1.0)
    v.prop.fill_(2.0)
    v.op.fill_(3.0)
    v.meta.fill_(4.0)
    assert torch.all(v.tensor[0:64] == 1.0)
    assert torch.all(v.tensor[64:128] == 2.0)
    assert torch.all(v.tensor[128:192] == 3.0)
    assert torch.all(v.tensor[192:256] == 4.0)


def test_amv_from_tensor_shares_or_wraps_256_vector():
    raw = torch.randn(256)
    v = AMVVector(raw)
    assert v.tensor.shape == (256,)
    # les partitions pointent vers les bonnes tranches
    assert torch.allclose(v.ent, raw[0:64])


def test_amv_confidence_is_first_meta_dim():
    # Spec §1: meta encode Confidence c in [0,1]. On le met sur meta[0] via sigmoid.
    v = AMVVector.zeros()
    assert v.confidence().shape == ()
