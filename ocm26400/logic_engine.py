"""Logique propositionnelle formelle — raisonnement vérifiable.

Moteur de logique propositionnelle (formelle, exacte) :
* Évaluation d'expressions (AND ∧, OR ∨, NOT ¬, IMPLIES →, IFF ↔) pour une assignation.
* Tables de vérité complètes.
* Tautologie / contradiction / satisfiabilité (par énumération).
* Modus ponens, équivalences.

C'est le raisonnement formel EXACT (pas neuro) — vérifiable. Complément du raisonnement
neural pour les déductions logiques garanties.
"""
from __future__ import annotations
import re
from itertools import product
from typing import Dict, List, Set

# ---- Parseur recursive-descent pour la logique propositionnelle ----
# Grammaire (précédence croissante) : équiv ↔ > implique → > ou ∨ > et ∧ > non ¬ > atome
# Sécurité : ZÉRO eval, parse manuel → impossible d'exécuter du code arbitraire.

def _tokenize(s: str) -> List[str]:
    s = s.replace("∧", " & ").replace("&&", " & ")
    s = s.replace("∨", " | ").replace("||", " | ")
    s = s.replace("¬", "~").replace("!", "~")
    s = s.replace("→", "->").replace("⟶", "->")
    s = s.replace("↔", "<->").replace("⟷", "<->")
    s = s.replace("TRUE", "TRUE").replace("FALSE", "FALSE")
    # insère des espaces autour des opérateurs et parenthèses
    s = re.sub(r"([()&|~])", r" \1 ", s)
    s = s.replace("->", " -> ").replace("<->", " <-> ")
    toks = [t for t in s.split() if t]
    return toks


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        node = self.equiv()
        return node

    def equiv(self):           # <-> (associatif gauche)
        node = self.implies()
        while self.peek() == "<->":
            self.next()
            node = ("iff", node, self.implies())
        return node

    def implies(self):         # -> (associatif droit)
        node = self.orexp()
        if self.peek() == "->":
            self.next()
            node = ("imp", node, self.implies())
        return node

    def orexp(self):           # |
        node = self.andexp()
        while self.peek() == "|":
            self.next()
            node = ("or", node, self.andexp())
        return node

    def andexp(self):          # &
        node = self.notexp()
        while self.peek() == "&":
            self.next()
            node = ("and", node, self.notexp())
        return node

    def notexp(self):          # ~
        if self.peek() == "~":
            self.next()
            return ("not", self.notexp())
        return self.atom()

    def atom(self):
        t = self.next()
        if t == "(":
            node = self.parse()
            if self.peek() == ")":
                self.next()
            return node
        return ("var", t)


def _parse(expr: str):
    return _Parser(_tokenize(expr)).parse()


def _ev(node, env) -> bool:
    tag = node[0]
    if tag == "var":
        name = node[1]
        if name in ("TRUE", "True"):
            return True
        if name in ("FALSE", "False"):
            return False
        return bool(env.get(name, False))
    if tag == "not":
        return not _ev(node[1], env)
    if tag == "and":
        return _ev(node[1], env) and _ev(node[2], env)
    if tag == "or":
        return _ev(node[1], env) or _ev(node[2], env)
    if tag == "imp":               # A → B = ¬A ∨ B
        return (not _ev(node[1], env)) or _ev(node[2], env)
    if tag == "iff":               # A ↔ B = (A∧B) ∨ (¬A∧¬B)
        a, b = _ev(node[1], env), _ev(node[2], env)
        return (a and b) or (not a and not b)
    return False


def extract_vars(expr: str) -> List[str]:
    """Variables propositionnelles d'une expression (lettres)."""
    return sorted(set(re.findall(r"\b[a-z]\b", expr)))



def evaluate(expr: str, assignment: Dict[str, bool]) -> bool:
    """Évalue une expression logique pour une assignation de variables.
    Sécurité : parseur recursive-descent manuel (ZÉRO eval). Retourne False si invalide."""
    env = {v: assignment.get(v, False) for v in extract_vars(expr)}
    try:
        return bool(_ev(_parse(expr), env))
    except Exception:
        return False


def truth_table(expr: str) -> List[Dict[str, bool]]:
    """Table de vérité complète : liste de {var: val, ..., 'result': bool}."""
    vars_ = extract_vars(expr)
    rows = []
    for combo in product([False, True], repeat=len(vars_)):
        assign = dict(zip(vars_, combo))
        res = evaluate(expr, assign)
        row = dict(assign)
        row["result"] = res
        rows.append(row)
    return rows


def is_tautology(expr: str) -> bool:
    """L'expression est-elle vraie pour TOUTE assignation ?"""
    return all(r["result"] for r in truth_table(expr))


def is_contradiction(expr: str) -> bool:
    return all(not r["result"] for r in truth_table(expr))


def is_satisfiable(expr: str) -> bool:
    return any(r["result"] for r in truth_table(expr))


def modus_ponens(p: bool, implies_q: bool) -> bool:
    """Si P est vrai et (P→Q) est vrai, alors Q est vrai. Forme fondamentale."""
    return (not p) or implies_q


def valid_argument(premises: List[str], conclusion: str) -> bool:
    """Un argument est VALIDE ssi : (conjonction des prémisses) → conclusion est une
    tautologie. C'est la définition formelle de la validité d'un argument."""
    conj = " ∧ ".join(f"({p})" for p in premises)
    impl_expr = f"({conj}) → ({conclusion})"
    return is_tautology(impl_expr)


if __name__ == "__main__":
    print("[logic] modus ponens (P→Q, P) :", is_tautology("(p → q) ∧ p → q"))
    print("[logic] non-contradiction (¬(p∧¬p)) :", is_tautology("¬(p ∧ ¬p)"))
    print("[logic] 'p ∨ q → p' tautologie ?", is_tautology("p ∨ q → p"), "(non)")
    print("[logic] argument valide (P→Q, P ⊢ Q) :", valid_argument(["p → q", "p"], "q"))
    print("[logic] table de vérité p→q :")
    for r in truth_table("p → q"):
        print(f"    p={r['p']} q={r['q']} → {r['result']}")
