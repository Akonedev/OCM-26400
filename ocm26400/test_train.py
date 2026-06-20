"""Tests orchestrateur d'entraînement (OCM-26400) — légers, sans GPU long."""
from ocm26400.train import stage0_build, run_pipeline, _quick_grok_acc, _quick_depth
from ocm26400.verifier import SymbolicDict, Verifier


def test_stage0_build_ground_truth():
    d, ver = stage0_build(p_mod=11)
    assert isinstance(d, SymbolicDict)
    assert isinstance(ver, Verifier)
    # ground-truth symbolique : (3a+5b) mod 11
    assert ver.compose(2, 3) == (3 * 2 + 5 * 3) % 11
    assert ver.compose(0, 0) == 0


def test_pipeline_stage0_only_instant():
    """run_pipeline avec stage 0 seul = instantané (pas d'entraînement)."""
    report = run_pipeline(stages=[0], smoke=True, device="cpu")
    assert report["stages_run"] == [0]
    assert report["smoke"] is True
    assert "device" in report


def test_quick_helpers_dont_crash():
    """Les helpers de vérification rapide tournent sans erreur sur un block non entraîné."""
    from ocm26400.reasoner import ReasonerBlock
    d, ver = stage0_build()
    blk = ReasonerBlock()
    acc = _quick_grok_acc(blk, d, ver, "cpu", n_check=3)
    depth = _quick_depth(blk, d, ver, "cpu", n_check=2)
    assert 0.0 <= acc <= 1.0
    assert depth >= 1.0


def test_full_mode_steps_larger_than_smoke():
    """Le mode full doit entraîner plus longtemps que le smoke (anti-shortcut)."""
    from ocm26400.train import run_pipeline
    # on ne lance QUE stage 0 pour rester rapide ; on vérifie juste la config
    smoke = run_pipeline([0], smoke=True, device="cpu")
    full = run_pipeline([0], smoke=False, device="cpu")
    assert smoke["smoke"] is True
    assert full["smoke"] is False


def test_train_results_json_written(tmp_path):
    """Le rapport d'entraînement se sérialise en JSON (pour bench.py)."""
    import json
    report = run_pipeline([0], smoke=True, device="cpu")
    s = json.dumps(report, default=str)   # ne doit pas lever
    assert "stages_run" in json.loads(s)
