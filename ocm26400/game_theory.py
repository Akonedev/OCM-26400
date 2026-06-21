"""Théorie des jeux — équilibre de Nash, minimax, dilemme — vague 3.

* Équilibre de Nash : stratégie où aucun joueur n'a intérêt à dévier unilatéralement.
* Minimax : stratégie optimale en jeu à somme nulle (minimiser le pire cas).
* Dilemme du prisonnier, dominance.
Vérifiable : minimax exact, Nash par énumération sur matrices de paiement.
"""
from __future__ import annotations
from typing import List, Tuple

PayoffMatrix = List[List[Tuple[float, float]]]   # [ligne][col] = (gain_ligne, gain_col)


def minimax(matrix: List[List[float]]) -> Tuple[int, float]:
    """Minimax (jeu à somme nulle, joueur ligne) : maximise le gain minimum garanti.
    Retourne (ligne_optimale, valeur_du_jeu)."""
    best_row, best_val = 0, float("-inf")
    for i, row in enumerate(matrix):
        worst = min(row)            # pire cas pour cette ligne (l'adversaire minimise)
        if worst > best_val:
            best_val, best_row = worst, i
    return best_row, best_val


def nash_equilibria(matrix: PayoffMatrix) -> List[Tuple[int, int]]:
    """Trouve les équilibres de Nash purs : aucune déviation unilatérale profitable.
    Énumération (matrice finie)."""
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    equilibria = []
    for i in range(rows):
        for j in range(cols):
            payoff_i, payoff_j = matrix[i][j]
            # i peut-il améliorer en changeant de ligne ?
            best_i = max(matrix[r][j][0] for r in range(rows))
            # j peut-il améliorer en changeant de colonne ?
            best_j = max(matrix[i][c][1] for c in range(cols))
            if payoff_i == best_i and payoff_j == best_j:
                equilibria.append((i, j))
    return equilibria


def is_dominant(matrix: PayoffMatrix, player: int) -> bool:
    """Y a-t-il une stratégie strictement dominante pour 'player' (0=ligne, 1=col) ?"""
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    if player == 0:
        for i1 in range(rows):
            if all(all(matrix[i1][c][0] > matrix[i2][c][0] for c in range(cols))
                   for i2 in range(rows) if i2 != i1):
                return True
    return False


# Matrices classiques
PRISONERS_DILEMMA = [[(-1, -1), (-3, 0)], [(0, -3), (-2, -2)]]   # (Coopère, Defect) × 2
MATCHING_PENNIES = [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]]      # somme nulle


if __name__ == "__main__":
    print("[jeux] minimax somme nulle :", minimax([[3, 2], [1, 4], [5, 0]]))
    print("[jeux] Nash dilemme du prisonnier :", nash_equilibria(PRISONERS_DILEMMA))
    print("[jeux] Nash matching pennies :", nash_equilibria(MATCHING_PENNIES),
          "(aucun pur → mixte)")
    print("[jeux] stratégie dominante (dilemme) :", is_dominant(PRISONERS_DILEMMA, 0))
