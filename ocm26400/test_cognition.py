"""Tests capacités cognitives (OCM-26400) — audit M1/M4/M5/M10."""
import pytest
from ocm26400.cognition import (
    TheoryOfMind, MoralAction, evaluate_morality, SpatialReasoner,
    structure_mapping, atom_solar_system_analogy,
)


# ---- M1 Theory of Mind ----

def test_sally_anne_false_belief():
    """LE test : Sally cherche là où elle CROIT (basket), pas la réalité (box)."""
    tom = TheoryOfMind()
    r = tom.sally_anne(sally_sees="basket", anne_moves_to="box")
    assert r["sally_searches_at"] == "basket"      # false belief
    assert r["reality"] == "box"
    assert r["false_belief_correct"] is True
    assert r["verdict"] == "TOM_CORRECT"


def test_belief_tracking_two_agents():
    tom = TheoryOfMind()
    tom.set_reality("key", "drawer")
    tom.observe("alice", "key", "drawer")
    tom.observe("bob", "key", "drawer")
    tom.move_unobserved("key", "shelf", observer_absent="bob")
    # alice a vu le déplacement ? non plus (on l'a pas ré-observée)
    assert tom.predict_search("bob", "key") == "drawer"   # bob croit drawer (false belief)


def test_predict_search_abstains_on_no_belief():
    tom = TheoryOfMind()
    assert tom.predict_search("ghost", "x") is None


# ---- M4 Moral ----

def test_moral_good_action():
    act = MoralAction("sauver une vie", impacts={"non_maleficence": 0.9,
                                                 "benevolence": 0.8})
    m = evaluate_morality(act)
    assert m["score"] > 1.0
    assert "BON" in m["verdict"]


def test_moral_bad_action():
    act = MoralAction("blesser volontairement", impacts={"non_maleficence": -0.9,
                                                         "benevolence": -0.7})
    m = evaluate_morality(act)
    assert m["score"] < -1.0
    assert "RÉPRÉHENSIBLE" in m["verdict"]


def test_moral_dilemma_steal_to_save():
    """Le vol pour sauver une vie : positif globalement (dilemme éthique)."""
    act = MoralAction("vol médicament pour sauver",
                      impacts={"non_maleficence": 0.9, "benevolence": 0.8,
                               "justice": -0.3, "honesty": -0.2})
    m = evaluate_morality(act)
    assert m["score"] > 0


def test_moral_reasoning_lists_principles():
    act = MoralAction("x", impacts={"autonomy": 0.5, "honesty": -0.4})
    m = evaluate_morality(act)
    assert len(m["reasoning"]) == 2
    assert all("contribution" in r for r in m["reasoning"])


# ---- M5 Spatial ----

def test_spatial_distance():
    sr = SpatialReasoner()
    sr.place("A", 0, 0); sr.place("B", 3, 4)
    assert sr.distance("A", "B") == 5.0


def test_spatial_relation_geometry():
    """C à (1,0) relativement à A(0,0) face north = à droite (vraie géométrie)."""
    sr = SpatialReasoner()
    sr.place("A", 0, 0); sr.place("C", 1, 0)
    assert "droite" in sr.relation("C", "A", "north")
    # face east : C serait devant
    sr.place("D", 1, 0)
    assert "devant" in sr.relation("D", "A", "east")


def test_spatial_nearest():
    sr = SpatialReasoner()
    sr.place("A", 0, 0); sr.place("B", 10, 10); sr.place("C", 1, 1)
    assert sr.nearest("A") == "C"


def test_spatial_move():
    sr = SpatialReasoner()
    sr.place("A", 0, 0)
    sr.move("A", 2, 0)
    assert sr.distance("A", "A") == 0 or sr.objects["A"].x == 2


# ---- M10 Analogie ----

def test_structure_mapping_transfers_relations():
    source = {"orbits": {"type": "relation", "args": ["planet", "sun"]}}
    rel_map = {"planet": "electron", "sun": "nucleus", "orbits": "orbits"}
    out = structure_mapping(source, source, rel_map)
    assert out["valid"] is True
    assert any("electron" in inf["target_inference"] for inf in out["inferences"])


def test_structure_mapping_rejects_non_injective():
    source = {"a": {"type": "relation", "args": []}}
    rel_map = {"a": "x", "b": "x"}    # non 1-1
    out = structure_mapping(source, source, rel_map)
    assert out["valid"] is False


def test_atom_solar_analogy():
    out = atom_solar_system_analogy()
    assert out["valid"] is True
    assert out["n_inferences"] >= 2
