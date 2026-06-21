"""Tests vague 3 — stats / théorie jeux / crypto / information / optimisation."""
import pytest
from ocm26400.statistics import mean, median, variance, correlation, bayes_update, linear_regression
from ocm26400.game_theory import minimax, nash_equilibria, PRISONERS_DILEMMA, MATCHING_PENNIES
from ocm26400.cryptography import caesar_encrypt, caesar_decrypt, vigenere_encrypt, vigenere_decrypt, rsa_keygen, rsa_encrypt, rsa_decrypt
from ocm26400.information import entropy, max_entropy, kl_divergence, mutual_information
from ocm26400.optimization import gradient_descent, minimize_1d, is_convex_1d


# stats
def test_stats_basic():
    assert mean([1, 2, 3]) == 2.0
    assert median([1, 2, 3]) == 2
    assert abs(variance([2, 4, 6], ddof=0) - 8 / 3) < 1e-6


def test_correlation_perfect():
    assert abs(correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-6


def test_bayes_vpp():
    r = bayes_update(0.01, 0.99, 0.95)     # test 99%, prévalence 1%
    assert 0.1 < r["vpp_P_malade_pos"] < 0.3   # ~17% (paradoxe)


# théorie jeux
def test_minimax():
    row, val = minimax([[3, 2], [1, 4], [5, 0]])
    assert row == 0 and val == 2


def test_nash_prisoners():
    assert nash_equilibria(PRISONERS_DILEMMA) == [(1, 1)]


def test_nash_matching_pennies_no_pure():
    assert nash_equilibria(MATCHING_PENNIES) == []


# crypto
def test_caesar_roundtrip():
    assert caesar_decrypt(caesar_encrypt("BONJOUR", 7), 7) == "BONJOUR"


def test_vigenere_roundtrip():
    assert vigenere_decrypt(vigenere_encrypt("ATTAQUEZ", "CLE"), "CLE") == "ATTAQUEZ"


def test_rsa_roundtrip():
    pub, priv = rsa_keygen(61, 53)
    assert rsa_decrypt(rsa_encrypt(42, pub), priv) == 42


def test_rsa_only_modular():
    pub, priv = rsa_keygen(11, 13)   # n=143
    c = rsa_encrypt(100, pub)
    assert c < 143     # chiffré < n


# information
def test_entropy_coin():
    assert abs(entropy([0.5, 0.5]) - 1.0) < 1e-9


def test_entropy_max():
    assert abs(max_entropy(8) - 3.0) < 1e-9


def test_kl_zero_if_equal():
    assert abs(kl_divergence([0.5, 0.5], [0.5, 0.5])) < 1e-9


def test_mutual_info_perfect():
    i = mutual_information([0.5, 0.5], [0.5, 0.5], [[0.5, 0], [0, 0.5]])
    assert abs(i - 1.0) < 1e-9


# optimisation
def test_gradient_descent_quadratic():
    xmin, fmin, _ = gradient_descent(lambda x: x[0] ** 2 + x[1] ** 2, [3.0, 4.0])
    assert abs(xmin[0]) < 0.01 and abs(fmin) < 0.01


def test_minimize_1d():
    x, f = minimize_1d(lambda t: (t - 3) ** 2, 0.0)
    assert abs(x - 3) < 0.01


def test_is_convex():
    assert is_convex_1d(lambda t: t ** 2) is True
