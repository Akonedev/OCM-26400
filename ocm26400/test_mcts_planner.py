"""Tests MCTS (OCM-26400) — audit MCTS."""
from ocm26400.mcts_planner import MCTS, MCTSNode


def _demo_env():
    def actions(s): return [] if s == 5 else ([1] if s >= 5 else [1, -1] if s > 0 else [1])
    def step(s, a): return s + a
    def reward(s): return 1.0 if s == 5 else 0.0
    def terminal(s): return s == 5
    return actions, step, reward, terminal


def test_mcts_finds_goal():
    """MCTS planifie 0→5 et atteint le but."""
    acts, step, rew, term = _demo_env()
    mcts = MCTS(acts, step, rew, term, max_depth=15, seed=0)
    state = 0
    for _ in range(8):
        if term(state):
            break
        a = mcts.search(state, n_iter=200)
        state = step(state, a)
    assert state == 5


def test_mcts_prefers_winning_action():
    """À l'état 4 (1 pas du but), MCTS choisit +1 (gagnant) pas -1."""
    acts, step, rew, term = _demo_env()
    mcts = MCTS(acts, step, rew, term, seed=0)
    a = mcts.search(4, n_iter=300)
    assert a == 1


def test_ucb1_exploration():
    """UCB1 = inf pour nœud non visité (explore)."""
    parent = MCTSNode(state=0, visits=10)
    child = MCTSNode(state=1, parent=parent, visits=0)
    assert child.ucb1() == float("inf")


def test_ucb1_balance():
    """UCB1 équilibre exploitation (value/visits) + exploration."""
    parent = MCTSNode(state=0, visits=100)
    high_value = MCTSNode(state=1, parent=parent, visits=10, value=8)
    low_value = MCTSNode(state=2, parent=parent, visits=10, value=2)
    assert high_value.ucb1() > low_value.ucb1()


def test_mcts_returns_none_on_terminal():
    acts, step, rew, term = _demo_env()
    mcts = MCTS(acts, step, rew, term, seed=0)
    # état 5 = terminal, aucune action
    assert mcts.actions(5) == []
