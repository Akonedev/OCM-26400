"""Tests algorithmes de graphes (OCM-26400) — audit G-ALGO."""
from ocm26400.graph_algorithms import dijkstra, astar, bfs, dfs, has_cycle


def test_dijkstra_shortest():
    g = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}
    path, dist = dijkstra(g, "A", "D")
    assert path == ["A", "B", "C", "D"]
    assert dist == 4     # 1+2+1


def test_dijkstra_no_path():
    g = {"A": {"B": 1}, "B": {}, "C": {}}
    path, dist = dijkstra(g, "A", "C")
    assert path is None and dist == float("inf")


def test_astar_matches_dijkstra():
    g = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}
    h = {"A": 4, "B": 2, "C": 1, "D": 0}
    p, d = astar(g, "A", "D", h)
    dp, dd = dijkstra(g, "A", "D")
    assert p == dp and d == dd     # A* admissible = même optimal


def test_bfs_shortest_hops():
    g = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    assert bfs(g, "A", "D") == ["A", "B", "D"]


def test_dfs_visits_all():
    g = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
    order = dfs(g, "A")
    assert set(order) == {"A", "B", "C", "D"}
    assert order[0] == "A"


def test_has_cycle():
    assert has_cycle({"A": ["B"], "B": ["C"], "C": ["A"]}) is True
    assert has_cycle({"A": ["B"], "B": [], "C": []}) is False


def test_dijkstra_optimal_vs_greedy():
    """Dijkstra ne se fait pas piéger par le greedy (poids qui remonte)."""
    g = {"A": {"B": 1, "C": 5}, "B": {"C": 1}, "C": {}}
    path, dist = dijkstra(g, "A", "C")
    assert path == ["A", "B", "C"] and dist == 2   # pas A→C direct (5)
