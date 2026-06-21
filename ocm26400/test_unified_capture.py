"""Tests unified capture training (OCM-26400)."""
import pytest
from ocm26400.unified_capture_train import sanity_check_p1, train_multi_domain_simultaneous


def test_sanity_check():
    """P1 : 100 steps, grok immédiat."""
    res = sanity_check_p1(device="cpu")
    assert res["grokked"] is True
    assert res["acc"] >= 0.5


def test_multi_domain_simultaneous():
    """Training SIMULTANÉ multi-domaine — tous grokkent."""
    res = train_multi_domain_simultaneous(n_steps=600, device="cpu")
    assert res["n_domains"] >= 3
    assert all(v >= 0.3 for v in res["final_accs"].values())
