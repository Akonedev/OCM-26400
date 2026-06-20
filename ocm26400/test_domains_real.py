"""Tests domaines réels (chimie/génétique/finance) — OCM-26400."""
import pytest
from ocm26400.chemistry import parse_formula, molar_mass, is_balanced, balance_simple
from ocm26400.genetics import punnett_square, phenotype_ratios, mendelian_cross
from ocm26400.finance import compound_interest, loan_payment, total_paid, present_value, effective_rate


# ---- chimie ----
def test_parse_formula_simple():
    assert parse_formula("H2O") == {"H": 2, "O": 1}
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}


def test_parse_formula_parens():
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}


def test_molar_mass_water():
    assert abs(molar_mass("H2O") - 18.015) < 0.01


def test_balance_water():
    cl, cr = balance_simple(["H2", "O2"], ["H2O"])
    assert cl == [2, 1] and cr == [2]
    assert is_balanced(["H2", "O2"], ["H2O"], cl, cr)


def test_balance_combustion():
    cl, cr = balance_simple(["CH4", "O2"], ["CO2", "H2O"])
    assert is_balanced(["CH4", "O2"], ["CO2", "H2O"], cl, cr)


def test_unbalanced_detected():
    assert not is_balanced(["H2", "O2"], ["H2O"])  # coeffs=1 → non équilibré


# ---- génétique ----
def test_punnett_monohybrid():
    ratios = punnett_square("Aa", "Aa")
    assert abs(ratios["AA"] - 0.25) < 0.01
    assert abs(ratios["Aa"] - 0.5) < 0.01
    assert abs(ratios["aa"] - 0.25) < 0.01


def test_mendel_3_to_1():
    """LE test : Aa×Aa donne 75% phénotype dominant / 25% récessif (loi de Mendel)."""
    dom = {"A": "dominant", "a": "récessif"}
    ph = phenotype_ratios("Aa", "Aa", dom)
    assert abs(ph["dominant"] - 0.75) < 0.01
    assert abs(ph["récessif"] - 0.25) < 0.01


def test_pure_breeding():
    ratios = punnett_square("AA", "aa")
    assert list(ratios.keys()) == ["Aa"]
    assert ratios["Aa"] == 1.0


# ---- finance ----
def test_compound_interest():
    assert abs(compound_interest(1000, 0.05, 10) - 1628.89) < 0.1


def test_loan_payment_positive():
    pmt = loan_payment(200000, 0.03, 300)
    assert 900 < pmt < 1000


def test_total_paid_interest():
    t = total_paid(200000, 0.03, 300)
    assert t["interest_paid"] > 0
    assert abs(t["total"] - t["monthly"] * 300) < 1


def test_present_value():
    assert abs(present_value(1000, 0.05, 10) - 613.91) < 0.1


def test_effective_rate_higher_than_nominal():
    assert effective_rate(0.12) > 12.0
