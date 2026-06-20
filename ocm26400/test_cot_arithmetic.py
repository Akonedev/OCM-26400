"""Tests CoT arithmétique vérifié (OCM-26400) — vérification de la claim NL↔exact."""
from ocm26400.cot_arithmetic import (
    eval_expr, step, ReasoningStep, CotTrace, parse_step, parse_trace,
    solve_word_problem, verify_claim,
)


# ---- moteur exact ----

def test_eval_expr_exact_integers():
    assert eval_expr("3*4") == 12
    assert eval_expr("12+5") == 17
    assert eval_expr("17-2") == 15
    assert eval_expr("2**10") in (1024, 1024.0) or eval_expr("2**10") == 1024
    assert eval_expr("(3+4)*2") == 14


def test_eval_expr_none_on_garbage():
    assert eval_expr("") is None
    assert eval_expr("hello world") is None
    assert eval_expr(None) is None


def test_eval_expr_rejects_arbitrary_code():
    """SÉCURITÉ : eval_expr ne doit JAMAIS exécuter du code arbitraire."""
    # toutes ces tentatives doivent retourner None (rejet), pas s'exécuter
    assert eval_expr("__import__('os').system('echo pwned')") is None
    assert eval_expr("open('/etc/passwd').read()") is None
    assert eval_expr("exec('x=1')") is None


# ---- étape + vérification exacte ----

def test_step_computes_exact_value():
    s = step(1, "3 étages de 4 pommes", "3*4")
    assert s.val == 12
    assert s.context == "3 étages de 4 pommes"
    assert s.expr == "3*4"
    assert s.is_exact() is True


def test_step_render_format():
    s = step(1, "calcul", "3*4")
    assert s.render() == "Step 1: calcul [3*4=12]"


def test_is_exact_rejects_wrong_value():
    """verify rejette une valeur fausse (zéro hallucination arithmétique)."""
    wrong = ReasoningStep(n=1, context="test", expr="3*4", val=11)   # 12 ≠ 11
    assert wrong.is_exact() is False
    correct = ReasoningStep(n=1, context="test", expr="3*4", val=12)
    assert correct.is_exact() is True


# ---- trace complète (pont NL → expr → résultat) ----

def test_solve_word_problem_all_exact():
    trace = solve_word_problem(
        problem="3 boîtes de 4, plus 5, moins 2.",
        steps_spec=[("3×4", "3*4"), ("+5", "12+5"), ("-2", "17-2")],
        final_expr="3*4+5-2",
    )
    assert trace.n_steps() == 3
    assert trace.all_exact() is True
    assert trace.final_answer == 15


# ---- parsing / round-trip (format apprenable) ----

def test_parse_step_roundtrip():
    s = parse_step("Step 2: ajout de 5 [12+5=17]")
    assert s is not None
    assert s.n == 2 and s.context == "ajout de 5" and s.expr == "12+5" and s.val == 17


def test_parse_trace_roundtrip_preserves_nl_and_exact():
    trace = solve_word_problem(
        problem="démo",
        steps_spec=[("a", "3*4"), ("b", "12+5")],
        final_expr="17",
    )
    rendered = trace.render()
    reparsed = parse_trace(rendered)
    assert reparsed.n_steps() == 2
    assert reparsed.all_exact() is True            # NL + exact préservés au round-trip
    assert reparsed.problem == "démo"


# ---- vérification de la claim (le test global) ----

def test_claim_verified():
    """LE test : la claim 'NL + arithmétique exacte préservées + pont apprenable' tient."""
    out = verify_claim()
    assert out["verdict"] == "CLAIM_VERIFIED"
    assert out["claim_1_nl_preserved"] is True
    assert out["claim_2_exact_arithmetic"] is True
    assert out["verify_rejects_wrong_value"] is True
    assert out["claim_3_format_learnable_roundtrip"] is True
    assert out["final_answer"] == 15


def test_format_is_regular_learnable_pattern():
    """Le format est régulier (même structure à chaque étape) → pattern apprenable par le modèle."""
    trace = solve_word_problem(
        problem="x",
        steps_spec=[("c1", "2+2"), ("c2", "4*3"), ("c3", "12-1")],
        final_expr="2+2",
    )
    pattern = [s.render().split(":", 1)[1].strip().startswith(s.context) for s in trace.steps]
    assert all(pattern)            # chaque étape suit 'contexte [expr=val]'
    # toutes les valeurs sont des entiers exacts (le moteur garantit)
    assert all(isinstance(s.val, int) for s in trace.steps)
