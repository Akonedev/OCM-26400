"""Tests benchmarks RÉELS (modèle entraîné) — OCM-26400."""
from ocm26400.bench_runner import (
    bench_agentic, bench_reasoning, bench_qcm, bench_terminal, run_all_benchmarks,
)


def test_bench_agentic_runs_real_tools():
    res = bench_agentic(6)
    assert res["n_tasks"] == 6
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["n_tools_available"] >= 2          # shell + web_fetch au moins
    # au moins la moitié des missions shell de base résolues
    assert res["n_solved"] >= 3


def test_bench_reasoning_perfect():
    res = bench_reasoning(30, max_depth=4)
    assert res["accuracy"] == 1.0                 # composition grokkée = exacte
    assert res["n_correct"] == 30
    assert res["avg_depth"] >= 2.0


def test_bench_qcm_abstention_works():
    res = bench_qcm(40)
    assert res["n_questions"] == 40
    # l'abstention est exercée (OOD simulé ~10%) → coverage < 1.0 parfois
    assert 0.0 <= res["coverage"] <= 1.0
    assert res["accuracy"] >= 0.9                 # haut sur les répondues


def test_bench_terminal_real_shell():
    res = bench_terminal(8)
    assert res["n_tasks"] == 8
    # ShellTool RÉEL : echo/expr doivent passer
    assert res["n_solved"] >= 6


def test_run_all_writes_results_and_level():
    import json, os
    rep = run_all_benchmarks()
    assert "BENCH_LEVEL" in rep
    assert 0.0 <= rep["BENCH_LEVEL"] <= 100.0
    path = os.path.join(os.path.dirname(__file__), "bench_runner_results.json")
    assert os.path.exists(path)
    with open(path) as f:
        d = json.load(f)
    assert set(d) >= {"agentic", "reasoning", "qcm", "terminal", "BENCH_LEVEL"}
