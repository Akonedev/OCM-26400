"""Programmation dynamique — algorithmes DP vérifiables — vague 3.

DP : sous-problèmes + mémoïsation/tabulation. Algorithmes classiques EXACTS :
* Sac à dos (0/1 knapsack) : valeur max sous contrainte de poids.
* Plus longue sous-séquence commune (LCS) : alignement de séquences.
* Distance d'édition (Levenshtein) : nb minimal d'opérations.
* Fibonacci mémoïsé, plus court chemin (Bellman-Ford), sous-séquence croissante (LIS).
Vérifiable : résultats exacts (ground truth connu).
"""
from __future__ import annotations
from typing import List, Tuple


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """Sac à dos 0/1 : valeur max. DP tabulaire. O(n·capacity)."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]


def lcs(a: str, b: str) -> int:
    """Longueur de la plus longue sous-séquence commune."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_sequence(a: str, b: str) -> str:
    """La sous-séquence commune elle-même (reconstruction)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # reconstruction
    i, j, res = m, n, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            res.append(a[i - 1]); i -= 1; j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(res))


def edit_distance(a: str, b: str) -> int:
    """Distance de Levenshtein (insertion/suppression/substitution)."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def longest_increasing_subsequence(arr: List[int]) -> int:
    """Plus longue sous-séquence strictement croissante."""
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def coin_change(coins: List[int], amount: int) -> int:
    """Nb minimal de pièces pour faire 'amount'. -1 si impossible."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != float("inf") else -1


if __name__ == "__main__":
    print("[dp] knapsack([2,3,4,5],[3,4,5,6],cap=5) =", knapsack([2, 3, 4, 5], [3, 4, 5, 6], 5))
    print("[dp] LCS('ABCBDAB','BDCAB') =", lcs("ABCBDAB", "BDCAB"),
          "| séquence:", lcs_sequence("ABCBDAB", "BDCAB"))
    print("[dp] edit_distance('kitten','sitting') =", edit_distance("kitten", "sitting"), "(=3)")
    print("[dp] LIS([10,9,2,5,3,7,101,18]) =", longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]))
    print("[dp] coin_change([1,5,10],27) =", coin_change([1, 5, 10], 27))
