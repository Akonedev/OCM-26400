"""Tests TDD — benchmark honnête LEVEL (OCM-26400, plan E0).

Valide : collecte des résultats, sonde packing (révèle séparabilité), calcul LEVEL qualifié.
"""
from ocm26400.bench import collect_results, packing_probe, level, run_bench


def test_collect_results_loads_jsons():
    res = collect_results()
    assert isinstance(res, dict) and len(res) > 0      # au moins crown_jewel_results


def test_packing_probe_reveals_packing():
    """La sonde packing retourne (V, retrieval@1, cos proche voisin) par taille."""
    pk = packing_probe(Vs=[100, 1000])
    assert len(pk) == 2
    for V, r1, nn in pk:
        assert isinstance(V, int) and 0.0 <= r1 <= 1.0 and -1.0 <= nn <= 1.0


def test_level_computes_qualified_score():
    res = collect_results()
    pk = packing_probe(Vs=[100, 1000])
    lv = level(res, pk)
    assert 0.0 <= lv["LEVEL"] <= 100.0
    assert "SOTA" in lv["qualification"]               # qualifié honnêtement
    assert "composition_crown_jewel_pt" in lv["subscores"]


def test_run_bench_end_to_end():
    rep = run_bench()
    assert "LEVEL" in rep and "results_files" in rep
    assert isinstance(rep["LEVEL"], float)
