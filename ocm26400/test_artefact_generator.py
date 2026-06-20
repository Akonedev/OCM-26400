"""Tests génération d'artefacts réels (OCM-26400) — audit H14."""
import os
import pytest
from ocm26400.artefact_generator import (
    generate_chart, generate_slides, generate_table, available_generators,
)


def test_available_generators_shape():
    av = available_generators()
    assert set(av) >= {"chart", "table", "slides"}


@pytest.mark.skipif(not available_generators()["chart"], reason="matplotlib absent")
def test_generate_chart_creates_png(tmp_path):
    r = generate_chart([("A", [1, 3, 2])], str(tmp_path / "c.png"), title="t")
    assert r["ok"] and os.path.exists(r["path"])
    assert os.path.getsize(r["path"]) > 100      # fichier non vide


@pytest.mark.skipif(not available_generators()["chart"], reason="matplotlib absent")
def test_generate_chart_kinds(tmp_path):
    for kind in ["line", "bar", "scatter"]:
        r = generate_chart([("x", [1, 2, 3])], str(tmp_path / f"c_{kind}.png"), kind=kind)
        assert r["ok"], f"{kind} a échoué"


@pytest.mark.skipif(not available_generators()["slides"], reason="python-pptx absent")
def test_generate_slides_creates_pptx(tmp_path):
    r = generate_slides("Titre", [{"title": "S1", "bullets": ["a", "b"]}],
                        str(tmp_path / "s.pptx"))
    assert r["ok"] and r["n_slides"] == 2
    assert os.path.getsize(r["path"]) > 1000     # .pptx non trivial


@pytest.mark.skipif(not available_generators()["table"], reason="matplotlib absent")
def test_generate_table_creates_png(tmp_path):
    r = generate_table(["A", "B"], [["1", "2"], ["3", "4"]], str(tmp_path / "t.png"))
    assert r["ok"] and os.path.exists(r["path"])
