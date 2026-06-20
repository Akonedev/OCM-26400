"""Tests génération de code vérifiée (OCM-26400) — audit H3."""
import pytest
from ocm26400.code_generator import (
    generate, verify_code, generate_and_verify, coverage, ALGO_TEMPLATES,
)


def test_generate_returns_source():
    code = generate("factorial")
    assert "def factorial" in code
    assert "return" in code


def test_generate_unknown_spec_raises():
    with pytest.raises(KeyError):
        generate("spec_inexistant")


def test_verify_correct_code_passes():
    code = generate("fibonacci")
    cases = [((0,), 0), ((1,), 1), ((10,), 55)]
    assert verify_code(code, cases) is True


def test_verify_buggy_code_fails():
    """verify_code attrape le code FAUX (zéro code incorrect validé)."""
    buggy = "def f(x):\n    return x + 1\n"   # devrait doubler
    assert verify_code(buggy, [((5,), 10)]) is False


def test_generate_and_verify_all_templates():
    """TOUS les templates génèrent du code vérifié correct par exécution."""
    for spec in ALGO_TEMPLATES:
        code, passed, results = generate_and_verify(spec)
        assert passed, f"{spec} a échoué: {results}"
        assert len(results) > 0


def test_coverage_all_correct():
    cov = coverage()
    assert all(cov.values()), f"échecs: {[k for k,v in cov.items() if not v]}"
    assert len(cov) >= 12


def test_factorial_execution():
    code, passed, results = generate_and_verify("factorial")
    assert passed
    assert 120 in [r["got"] for r in results]   # 5! = 120


def test_is_prime_execution():
    code, passed, _ = generate_and_verify("is_prime")
    assert passed


def test_code_is_runnable_python():
    """Le code généré est du Python valide (compile sans erreur)."""
    for spec in ALGO_TEMPLATES:
        code = generate(spec)
        compile(code, f"<{spec}>", "exec")   # lève SyntaxError si invalide


def test_arbitrary_code_rejected_without_execution():
    """SÉCURITÉ : verify_code refuse le code non-template SANS l'exécuter."""
    malicious = "def f():\n    return __import__('os').system('echo pwned')\n"
    # retourne False (rejeté) sans exécuter l'import malveillant
    assert verify_code(malicious, [((), None)]) is False


def test_generate_and_verify_only_trusted():
    """generate_and_verify n'exécute que du code template de confiance."""
    code, passed, results = generate_and_verify("gcd")
    assert code in __import__("ocm26400").code_generator.ALGO_TEMPLATES.values()
    assert passed is True
