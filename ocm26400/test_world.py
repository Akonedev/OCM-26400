"""Tests TDD — monde interactif + PNJ (OCM-26400).

Valide : poursuite de but, routine planifiée, évolution des habitudes, interactions
entre PNJ, continuation cohérente, et contrôle user injecté.
"""
import random

from ocm26400.world import Entity, NPC, World


def test_npc_moves_toward_goal():
    """Le PNJ se rapproche de son but à chaque tick."""
    npc = NPC("a", 0, 0, goal=(5, 5), rng=random.Random(0))
    w = World().add(npc)
    w.step()
    assert (npc.x, npc.y) != (0, 0)
    assert abs(npc.x - 5) + abs(npc.y - 5) < 10            # plus proche


def test_world_step_advances_and_logs():
    npc = NPC("a", 0, 0, goal=(3, 3), rng=random.Random(0))
    w = World().add(npc)
    snap = w.step()
    assert w.tick == 1
    assert len(w.history) == 1
    assert "a" in snap["positions"]


def test_npc_routine_scheduled():
    """Routine planifiée (tick % period) exécutée au lieu de la poursuite de but."""
    npc = NPC("a", 5, 5, goal=(0, 0), routine={0: "haut"}, rng=random.Random(0))
    w = World().add(npc)
    a0 = w.step()["actions"]["a"]
    assert a0 == "haut"
    assert npc.y == 4                                       # a bougé vers le haut


def test_npc_habits_evolve():
    """Les habitudes évoluent : le but change périodiquement (routines qui varient)."""
    npc = NPC("a", 0, 0, goal=(9, 9), habit_period=2, rng=random.Random(1))
    w = World().add(npc)
    g0 = npc.goal
    for _ in range(3):            # ticks 0,1,2 -> évolution à tick 2 (2%2==0, tick>0)
        w.step()
    assert npc.goal != g0                                   # le but a évolué


def test_npc_interaction_detected():
    """Deux PNJ adjacents -> interaction enregistrée."""
    a = NPC("a", 0, 0, goal=(1, 0), rng=random.Random(0))
    b = NPC("b", 2, 0, goal=(1, 0), rng=random.Random(0))
    w = World().add(a).add(b)
    w.run(3)
    assert any("a<->b" in i for i in w.interactions)


def test_world_run_with_user_control():
    """Contrôle user injecté à chaque tick (le user déplace une entité)."""
    npc = NPC("a", 0, 0, goal=(9, 9), rng=random.Random(0))
    w = World().add(npc)

    def ctrl(world):
        npc.move(0, 0, world.w, world.h)          # user: réinitialise pos (contrôle)
    w.run(2, user_control=ctrl)
    assert w.tick == 2


def test_coherent_continuation():
    """run(N) -> N états rejoués, positions dans les bornes, pas de crash."""
    w = World(w=8, h=8)
    for i in range(5):
        w.add(NPC(f"n{i}", i, i, goal=(7 - i, 7 - i), rng=random.Random(i)))
    w.run(10)
    assert len(w.history) == 10
    for n in w.npcs:
        assert 0 <= n.x < 8 and 0 <= n.y < 8
