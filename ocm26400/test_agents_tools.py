"""Tests TDD — tools/skills pour agents + PNJ (OCM-26400).

Valide : Skill, Toolkit (registre, use), Mission + execute_mission, équipement d'un PNJ.
"""
from ocm26400.agents_tools import Skill, Toolkit, Mission, execute_mission, default_toolkit, npc_with_mission
from ocm26400.world import NPC


def test_skill_use():
    s = Skill("calculate", lambda a, b: a + b)
    assert s.use(2, 3) == 5


def test_toolkit_add_and_use():
    tk = Toolkit().add(Skill("speak", lambda t: f"«{t}»")).add(Skill("move", lambda dx, dy: (dx, dy)))
    assert tk.has("speak") and tk.has("move")
    assert tk.use("move", 1, 2) == (1, 2)
    assert "inconnu" in tk.use("absente")


def test_execute_mission_plan():
    """Une mission = but + plan de skills ; exécutée via le toolkit."""
    tk = default_toolkit()
    m = Mission(goal="livrer colis",
                plan=[("move", (1, 0)), ("lookup", ("colis",)), ("speak", ("voilà",))])
    log = execute_mission(tk, m)
    assert log == [(1, 0), "info(colis)", "« voilà »"]


def test_default_toolkit_skills():
    tk = default_toolkit()
    assert set(["calculate", "lookup", "move", "speak"]).issubset(set(tk.names()))
    assert tk.use("calculate", 4, 5) == 9


def test_npc_equipped_with_toolkit_and_mission():
    """Un PNJ reçoit toolkit + mission (compétences + skills pour sa mission)."""
    npc = NPC("agent1", 0, 0, goal=(3, 3))
    tk = default_toolkit()
    m = Mission(goal="explorer", plan=[("move", (2, 0)), ("speak", ("vu",))])
    npc_with_mission(npc, m, tk)
    assert npc.toolkit is tk and npc.mission is m
    log = execute_mission(npc.toolkit, npc.mission)
    assert log[1] == "« vu »"
