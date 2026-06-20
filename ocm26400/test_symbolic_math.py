"""Tests math symboliques (OCM-26400) — étend la compétence olympiade."""
from ocm26400.symbolic_math import (
    poly_add, poly_mul, poly_eval, poly_deriv, poly_integ,
    gcd, lcm, is_prime, factorize, modexp, quad_roots, linear_solve,
    symbolic_math_rules,
)


# ---- polynômes ----

def test_poly_add_mul_eval():
    p = [1, 2, 3]                      # 1 + 2x + 3x²
    assert poly_eval(p, 2) == 17       # 1 + 4 + 12
    assert poly_add([1, 1], [2, 0, 1]) == [3, 1, 1]
    assert poly_mul([1, 1], [1, -1]) == [1, 0, -1]   # (1+x)(1-x)=1-x²


def test_poly_deriv():
    assert poly_deriv([5, 6, 7]) == [6, 14]      # 5+6x+7x² → 6+14x
    assert poly_deriv([3]) == [0]                 # constante → 0
    assert poly_deriv([0, 4]) == [4]              # 4x → 4


def test_poly_deriv_verify_rejects_wrong():
    """verify rejette le faux (le modèle CONNAÎT la dérivée)."""
    gold = poly_deriv([5, 6, 7])
    assert gold == [6, 14]
    assert poly_deriv([5, 6, 7]) != [6, 13]       # faux rejeté


# ---- théorie des nombres ----

def test_gcd_lcm():
    assert gcd(12, 18) == 6
    assert gcd(17, 5) == 1
    assert lcm(4, 6) == 12
    assert lcm(0, 5) == 0


def test_is_prime():
    assert is_prime(2) and is_prime(97) and is_prime(101)
    assert not is_prime(1) and not is_prime(0) and not is_prime(91)  # 7*13


def test_factorize():
    assert factorize(60) == [2, 2, 3, 5]
    assert factorize(13) == [13]
    assert factorize(1) == []


def test_modexp():
    assert modexp(2, 10, 1000) == 24          # 1024 mod 1000
    assert modexp(3, 5, 7) == pow(3, 5, 7)    # 243 mod 7 = 5
    assert modexp(5, 0, 13) == 1


def test_modexp_matches_python_pow():
    import random
    rng = random.Random(0)
    for _ in range(50):
        b, e, m = rng.randint(1, 50), rng.randint(0, 20), rng.randint(1, 100)
        assert modexp(b, e, m) == pow(b, e, m)


# ---- algèbre ----

def test_quad_roots():
    # x² - 5x + 6 = 0 → (2, 3)
    assert sorted(quad_roots(1, -5, 6)) == [2, 3]
    # pas de racines réelles
    assert quad_roots(1, 0, 1) is None


def test_linear_solve():
    assert linear_solve(3, 2, 11) == 3         # 3x+2=11 → x=3
    assert linear_solve(2, 0, 10) == 5


# ---- intégration RuleLibrary ----

def test_symbolic_rules_verify_true_reject_false():
    """Chaque règle symbolique : verify accepte le vrai, rejette le faux."""
    rules = {r.name: r for r in symbolic_math_rules()}
    # gcd
    g = rules["gcd"]
    assert g.verify((12, 18), 6) is True
    assert g.verify((12, 18), 5) is False          # faux rejeté
    # poly_deriv (arity 1)
    pd = rules["poly_deriv"]
    assert pd.verify(([5, 6, 7],), [6, 14]) is True
    assert pd.verify(([5, 6, 7],), [6, 13]) is False
    # modexp (arity 3)
    me = rules["modexp"]
    assert me.verify((2, 10, 1000), 24) is True
    assert me.verify((2, 10, 1000), 25) is False


def test_symbolic_rules_domains_covered():
    rules = symbolic_math_rules()
    domains = {r.domain for r in rules}
    assert "algebra" in domains
    assert "calculus" in domains
    assert "number_theory" in domains
    assert len(rules) == 10
