"""Tests scanner OWASP réel (AST) — OCM-26400 — audit H15."""
from ocm26400.owasp_scanner import scan_code, scan_report, VULNERABLE_SAMPLE, SAFE_SAMPLE


def test_detects_command_injection():
    findings = scan_code("import os\nos.system('echo ' + x)\n")
    rules = [f.rule for f in findings]
    assert any("command_injection" in r for r in rules)


def test_detects_eval():
    findings = scan_code("eval(user)\n")
    assert any("arbitrary_code_exec" in f.rule for f in findings)


def test_detects_sql_injection_fstring():
    code = "def f(uid):\n    sql = f\"SELECT * FROM u WHERE id={uid}\"\n"
    findings = scan_code(code)
    assert any("sql_injection" in f.rule for f in findings)


def test_detects_pickle_deserialization():
    code = "import pickle\npickle.loads(data)\n"
    findings = scan_code(code)
    assert any("pickle" in f.rule for f in findings)


def test_detects_yaml_unsafe():
    findings = scan_code("import yaml\nyaml.load(d)\n")
    assert any("yaml" in f.rule for f in findings)


def test_detects_hardcoded_secret():
    code = 'API_KEY = "sk-1234567890abcdef"\n'
    findings = scan_code(code)
    assert any("secret" in f.rule for f in findings)


def test_detects_weak_hash():
    code = "import hashlib\nhashlib.md5(pw)\n"
    findings = scan_code(code)
    assert any("weak_hash" in f.rule for f in findings)


def test_safe_code_clean():
    """LE test : le code sûr → 0 faux-positif."""
    rep = scan_report(SAFE_SAMPLE)
    assert rep["verdict"] == "CLEAN"
    assert rep["n_findings"] == 0


def test_vulnerable_sample_finds_many():
    """Le fixture vulnérable déclenche plusieurs détections réelles."""
    rep = scan_report(VULNERABLE_SAMPLE)
    assert rep["verdict"] == "VULNERABLE"
    assert rep["n_findings"] >= 6
    assert rep["critical"] >= 4      # command inj + eval + sqli + secret au moins


def test_syntax_error_returns_empty():
    assert scan_code("def (broken") == []


def test_findings_have_remediation():
    findings = scan_code("eval(x)\n")
    assert all(f.remediation for f in findings)
