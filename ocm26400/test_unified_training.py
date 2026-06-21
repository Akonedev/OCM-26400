"""Tests pipeline unifié (OCM-26400)."""
from ocm26400.unified_training import run_unified_pipeline


def test_pipeline_runs():
    """LE test : le pipeline d'entraînement unifié s'exécute complètement."""
    rep = run_unified_pipeline()
    assert rep["overall_verdict"] == "PIPELINE_COMPLETE"
    assert "curriculum_v4_arithmetic" in rep["stages"]
    assert "curriculum_v4_language" in rep["stages"]
    assert "domain_cascade" in rep["stages"]
    assert "evaluation" in rep["stages"]
    assert rep["stages"]["evaluation"]["real_bench_accuracy"] >= 0.9
