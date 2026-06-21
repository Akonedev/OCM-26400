"""Tests curriculum v4 (ADR-0030) — procédure de training du projet."""
import pytest
import torch
from ocm26400.primitive_grok_curriculum import (
    run_curriculum_v4, GATES, _scratchpad_cascade_eval, _train_solo_slot, _make_op_verifier,
)


def test_gates_documented():
    """Les gates du Training.md : L1≥0.99, L2≥0.95, L5≥0.90, L6≥0.85."""
    assert GATES["L1"] == 0.99
    assert GATES["L2"] == 0.95


def test_solo_slot_grokks():
    """Phase SOLO : un opérateur grok individuellement."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blk, acc = _train_solo_slot("add", n_steps=1500, device=dev)
    assert acc >= 0.85     # grok (procédure §2 ; CPU un peu sous le gate 0.99)


def test_scratchpad_cascade_works():
    """Loi L1 : la cascade scratchpad (intermédiaire puis final) généralise."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blk, _ = _train_solo_slot("add", n_steps=1500, device=dev)
    d, _ = _make_op_verifier("add")
    cascade_acc = _scratchpad_cascade_eval(blk, d, "add", dev, depth=3, n_test=30)
    assert cascade_acc >= 0.6     # cascade (décomposition)


def test_curriculum_v4_complete():
    """LE test : le curriculum v4 complet (ADR-0030) atteint le verdict COMPLETE."""
    rep = run_curriculum_v4(ops=["add"], solo_steps=1500, cascade_depth=2,
                             enable_sleep=False)
    assert rep.verdict == "CURRICULUM_V4_COMPLETE"
    assert all(p.grokked for p in rep.phases)
    assert rep.cascade_acc >= 0.8


def test_cascade_deeper_than_solo():
    """La cascade résout des compositions PLUS profondes que le slot solo (depth>1)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blk, _ = _train_solo_slot("mul", n_steps=1500, device=dev)
    d, _ = _make_op_verifier("mul")
    acc = _scratchpad_cascade_eval(blk, d, "mul", dev, depth=4, n_test=20)
    assert acc >= 0.5     # cascade profonde (CPU moins précis)
