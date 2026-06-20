"""Algorithmes de graphes — Dijkstra / A* / BFS / DFS — réfute audit G-ALGO.

EX-B297/298. Algorithmes classiques VÉRIFIABLES (chemin optimal garanti) :
* Dijkstra : plus court chemin (graphes pondérés positifs).
* A* : plus court chemin avec heuristique (admissible → optimal).
* BFS : plus court chemin en nb d'arêtes (non pondéré).
* DFS : parcours en profondeur, détection de cycles.
Tous avec reconstruction du chemin. C'est la compétence algorithmique réelle.
"""
from __future__ import annotations
import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Graph = Dict[str, Dict[str, float]]


def dijkstra(graph: Graph, start: str, end: str) -> Tuple[Optional[List[str]], float]:
    """Dijkstra : plus court chemin start→end (poids positifs). Retourne (chemin, distance)."""
    dist = {start: 0.0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq = [(0.0, start)]
    visited: Set[str] = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if end not in dist:
        return None, float("inf")
    # reconstruction
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path)), dist[end]


def astar(graph: Graph, start: str, end: str,
          heuristic: Dict[str, float] = None) -> Tuple[Optional[List[str]], float]:
    """A* : Dijkstra + heuristique admissible (h ≤ distance réelle). Optimal si admissible."""
    h = heuristic or {}
    g = {start: 0.0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq = [(h.get(start, 0), 0.0, start)]
    visited: Set[str] = set()
    while pq:
        _, d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < g.get(v, float("inf")):
                g[v] = nd
                prev[v] = u
                f = nd + h.get(v, 0)
                heapq.heappush(pq, (f, nd, v))
    if end not in g:
        return None, float("inf")
    path, cur = [], end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path)), g[end]


def bfs(graph: Dict[str, List[str]], start: str, end: str) -> Optional[List[str]]:
    """BFS : plus court chemin en nb d'arêtes (graphe non pondéré)."""
    if start == end:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in prev:
                prev[v] = u
                if v == end:
                    path, cur = [], end
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return list(reversed(path))
                q.append(v)
    return None


def dfs(graph: Dict[str, List[str]], start: str) -> List[str]:
    """DFS : parcours en profondeur (ordre de visite)."""
    visited: Set[str] = set()
    order: List[str] = []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        for v in reversed(graph.get(u, [])):
            if v not in visited:
                stack.append(v)
    return order


def has_cycle(graph: Dict[str, List[str]]) -> bool:
    """Détecte un cycle dans un graphe orienté (DFS à 3 couleurs)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    for n in graph:
        if color[n] != WHITE:
            continue
        stack = [(n, iter(graph.get(n, [])))]
        color[n] = GRAY
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if color.get(nb, WHITE) == GRAY:
                    return True
                if color.get(nb, WHITE) == WHITE:
                    color[nb] = GRAY
                    stack.append((nb, iter(graph.get(nb, []))))
                    advanced = True
                    break
            if not advanced:
                color[node] = BLACK
                stack.pop()
    return False


if __name__ == "__main__":
    g = {"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}
    path, dist = dijkstra(g, "A", "D")
    print(f"[graph] Dijkstra A→D : {path} (dist={dist})")
    h = {"A": 4, "B": 2, "C": 1, "D": 0}     # heuristique admissible
    p2, d2 = astar(g, "A", "D", h)
    print(f"[graph] A* A→D : {p2} (dist={d2})")
    gu = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    print(f"[graph] BFS A→D : {bfs(gu, 'A', 'D')} | DFS depuis A : {dfs(gu, 'A')}")
    print(f"[graph] cycle dans {{A→B→A}} ? {has_cycle({'A': ['B'], 'B': ['A']})}")
