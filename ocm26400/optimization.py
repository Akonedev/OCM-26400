"""Optimisation — descente de gradient, minima — vague 3.

* Gradient descent : minimise f(x) itérativement (x ← x − η·∇f).
* Recherche linéaire, point stationnaire.
* Convexité : pour f convexe, le minimum global est trouvé.
Vérifiable : converge vers le minimum connu (ex : f=x² → min en 0).
"""
from __future__ import annotations
import math
from typing import Callable, List, Tuple


def numerical_gradient(f: Callable[[List[float]], float], x: List[float],
                       h: float = 1e-5) -> List[float]:
    """Gradient numérique par différences finies centrées."""
    grad = []
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        grad.append((f(xp) - f(xm)) / (2 * h))
    return grad


def gradient_descent(f: Callable[[List[float]], float], x0: List[float],
                     lr: float = 0.1, n_iter: int = 1000, tol: float = 1e-8
                     ) -> Tuple[List[float], float, int]:
    """Descente de gradient. Retourne (xmin, fmin, itérations)."""
    x = x0.copy()
    for it in range(n_iter):
        grad = numerical_gradient(f, x)
        step = [lr * g for g in grad]
        x = [xi - s for xi, s in zip(x, step)]
        if max(abs(s) for s in step) < tol:
            return x, f(x), it
    return x, f(x), n_iter


def minimize_1d(f: Callable[[float], float], x0: float, lr: float = 0.1,
                n_iter: int = 200) -> Tuple[float, float]:
    """Minimise f:R→R. Retourne (xmin, fmin)."""
    x = x0
    for _ in range(n_iter):
        h = 1e-5
        grad = (f(x + h) - f(x - h)) / (2 * h)
        step = lr * grad
        x -= step
        if abs(step) < 1e-10:
            break
    return x, f(x)


def is_convex_1d(f: Callable[[float], float], lo: float = -5, hi: float = 5,
                 n: int = 50) -> bool:
    """Vérifie la convexité 1D (inégalité Jensen : f(tx+(1-t)y) ≤ tf(x)+(1-t)f(y))."""
    xs = [lo + i * (hi - lo) / n for i in range(n + 1)]
    for i in range(len(xs) - 2):
        mid = (xs[i] + xs[i + 2]) / 2
        if f(mid) > (f(xs[i]) + f(xs[i + 2])) / 2 + 1e-6:
            return False
    return True


if __name__ == "__main__":
    f = lambda x: x[0] ** 2 + 3 * x[1] ** 2          # convexe, min en (0,0)
    xmin, fmin, it = gradient_descent(f, [5.0, 5.0], lr=0.1)
    print(f"[optim] min x²+3y² : x={xmin} f={fmin:.6f} en {it} iters (attendu 0,0)")
    xmin1, fmin1 = minimize_1d(lambda x: (x - 3) ** 2, 0.0)      # min en 3
    print(f"[optim] min (x-3)² : x={xmin1:.4f} f={fmin1:.6f} (attendu x=3)")
    print(f"[optim] x² est convexe ? {is_convex_1d(lambda x: x ** 2)}")
