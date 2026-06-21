"""Tests programmation dynamique (OCM-26400)."""
from ocm26400.dynamic_programming import (knapsack, lcs, lcs_sequence, edit_distance,
    longest_increasing_subsequence, coin_change)


def test_knapsack():
    assert knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7     # items 2+3 (val 3+4=7) ou 5(val6)


def test_lcs():
    assert lcs("ABCBDAB", "BDCAB") == 4
    assert lcs("AGGTAB", "GXTXAYB") == 4   # GTAB


def test_lcs_sequence():
    seq = lcs_sequence("AGGTAB", "GXTXAYB")
    assert seq == "GTAB"


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0


def test_lis():
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4


def test_coin_change():
    assert coin_change([1, 5, 10], 27) == 5      # 10+10+5+1+1 = 5 pièces (optimal)
    assert coin_change([1, 5, 10], 11) == 2      # 10+1


def test_coin_change_impossible():
    assert coin_change([2], 3) == -1
