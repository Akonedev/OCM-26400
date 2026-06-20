"""Planification MCTS (Monte Carlo Tree Search) — réfute audit MCTS CRITIQUE.

EX-M (maths), EX-B205. MCTS = test-time compute pour la PLANIFICATION et le raisonnement.
L'arbre grandit par 4 phases : sélection (UCB1), expansion, simulation (rollout),
rétropropagation. C'est l'instantiation du test-time compute (le modèle "réfléchit"
plus longtemps = meilleure solution).

Générique : on branche un Environment (états/actions/récompense). MCTS trouve la
meilleure action par rollouts. Utilisable pour : jeux, raisonnement multi-étapes,
exploration d'arbre de preuve.

* MCTSNode : nœud (état, parent, visites, valeur, enfants).
* MCTS.search(root_state, n_iter) → meilleure action (par rollouts).
* UCB1 : sélection exploration/exploitation.
Vérifiable : converge vers l'optimal sur un env simple (ex : trouver le chemin gagnant).
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class MCTSNode:
    state: Any
    parent: Optional["MCTSNode"] = None
    action: Optional[Any] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried: List[Any] = field(default_factory=list)

    def ucb1(self, explore: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value / self.visits
        return exploit + explore * math.sqrt(math.log(self.parent.visits + 1) / self.visits)

    def best_child(self, explore: float = 1.414) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(explore))


class MCTS:
    """Monte Carlo Tree Search. Branche un environnement (state, actions, step, reward, terminal)."""

    def __init__(self, env_actions: Callable, env_step: Callable, env_reward: Callable,
                 env_terminal: Callable, rollout_policy: Callable = None,
                 max_depth: int = 20, seed: int = 0):
        self.actions = env_actions          # state → [actions possibles]
        self.step = env_step                # (state, action) → new_state
        self.reward = env_reward            # state → récompense (gagnant=1)
        self.terminal = env_terminal        # state → bool (état terminal)
        self.rollout_policy = rollout_policy or (lambda s, acts: random.choice(acts))
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.untried:
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried:
            action = node.untried.pop()
            child = MCTSNode(state=self.step(node.state, action), parent=node, action=action,
                             untried=self.actions(self.step(node.state, action)))
            node.children.append(child)
            return child
        return node

    def _rollout(self, state) -> float:
        for _ in range(self.max_depth):
            if self.terminal(state):
                return self.reward(state)
            acts = self.actions(state)
            if not acts:
                return self.reward(state)
            state = self.step(state, self.rollout_policy(state, acts))
        return self.reward(state)

    def _backprop(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, root_state: Any, n_iter: int = 500) -> Any:
        """Retourne la meilleure action depuis root_state (après n_iter rollouts)."""
        root = MCTSNode(state=root_state, untried=self.actions(root_state))
        for _ in range(n_iter):
            leaf = self._select(root)
            if not self.terminal(leaf.state):
                leaf = self._expand(leaf)
            reward = self._rollout(leaf.state)
            self._backprop(leaf, reward)
        if not root.children:
            return None
        # meilleure action = enfant le + visité (robuste)
        best = max(root.children, key=lambda c: c.visits)
        return best.action


# ---- env de démo : grille 1D, atteindre la position 5 ----
def demo_env():
    """Atteindre la position 5 depuis 0 (actions +1/-1). MCTS doit trouver +1,+1,+1,+1,+1."""
    def actions(s):
        return [] if s == 5 else ([1] if s >= 5 else [1, -1] if s > 0 else [1])
    def step(s, a):
        return s + a
    def reward(s):
        return 1.0 if s == 5 else 0.0
    def terminal(s):
        return s == 5
    return actions, step, reward, terminal


if __name__ == "__main__":
    acts, step, rew, term = demo_env()
    mcts = MCTS(acts, step, rew, term, max_depth=15, seed=0)
    state, path = 0, []
    for _ in range(6):
        if term(state):
            break
        a = mcts.search(state, n_iter=200)
        path.append(a)
        state = step(state, a)
    print(f"[mcts] planification 0→5 : actions={path} état_final={state} "
          f"récompense={rew(state)} {'✓ GAGNÉ' if term(state) else '✗'}")
