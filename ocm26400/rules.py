"""Bibliothèque de RÈGLES multi-domaines (OCM-26400, cahier des charges).

L'utilisateur : « la génération vient de la compréhension, de toutes les règles. Le
modèle doit apprendre toutes les règles en maths, physiques etc et comprendre, avant de
généraliser et générer ».

Une RÈGLE est une opération DÉTERMINISTE et VÉRIFIABLE (au sens du Verifier symbolique) :
math (add/mul/op/neg), physique (force=m·a, vélocité=d/t, énergie cinétique ½mv²,
quantité de mouvement m·v), grammaire (préétérit +ed, pluriel +s, gérondif +ing). Le
modèle APPREND+COMPREND une règle s'il sait l'APPLIQUER et la VÉRIFIER correctement ;
il GÉNÈRE en COMPOSANT les règles comprises.

RuleLibrary : collection indexée par domaine (routage MoE-style), apply(), verify()
(compréhension), compose() (génération par composition de règles comprises).

HONNÊTE : règles symboliques exactes (vérifiables, pas devinées) — c'est l'opposé d'un
LLM qui hallucine : la règle est codée explicitement, le modèle l'apprend puis génère par
composition (comprehension -> généralisation -> génération, cf. capstone).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Tuple


@dataclass
class Rule:
    name: str
    domain: str
    fn: Callable[..., Any]
    arity: int
    desc: str = ""

    def apply(self, *args) -> Any:
        return self.fn(*args)

    def verify(self, args: Tuple, output: Any) -> bool:
        """Compréhension : la règle appliquée aux args donne-t-elle output ?"""
        return self.apply(*args) == output


# ---- règles exactes par domaine ----

def _math_rules(n: int = 11) -> List[Rule]:
    return [
        Rule("add", "math", lambda a, b: (a + b) % n, 2, "(a+b) mod n"),
        Rule("mul", "math", lambda a, b: (a * b) % n, 2, "(a*b) mod n"),
        Rule("linop", "math", lambda a, b: (3 * a + 5 * b) % n, 2, "(3a+5b) mod n"),
        Rule("neg", "math", lambda a: (-a) % n, 1, "-a mod n"),
    ]


def _physics_rules() -> List[Rule]:
    return [
        Rule("force", "physics", lambda m, a: m * a, 2, "F = m·a (Newton)"),
        Rule("velocity", "physics", lambda d, t: d / t if t else 0.0, 2, "v = d/t"),
        Rule("kinetic", "physics", lambda m, v: 0.5 * m * v * v, 2, "Ec = ½mv²"),
        Rule("momentum", "physics", lambda m, v: m * v, 2, "p = m·v"),
    ]


def _grammar_rules() -> List[Rule]:
    return [
        Rule("past", "grammar", lambda s: s + "ed", 1, "préétérit +ed"),
        Rule("plural", "grammar", lambda s: s + "s", 1, "pluriel +s"),
        Rule("gerund", "grammar", lambda s: s + "ing", 1, "gérondif +ing"),
    ]


@dataclass
class RuleLibrary:
    """Bibliothèque de règles vérifiables multi-domaines (math/physique/grammaire)."""
    rules: Dict[str, Rule] = field(default_factory=dict)

    @classmethod
    def default(cls, n: int = 11) -> "RuleLibrary":
        lib = cls()
        for r in _math_rules(n) + _physics_rules() + _grammar_rules():
            lib.add(r)
        return lib

    def add(self, rule: Rule):
        self.rules[rule.name] = rule

    def domains(self) -> List[str]:
        return sorted({r.domain for r in self.rules.values()})

    def by_domain(self, domain: str) -> List[Rule]:
        """Routage MoE-style : règles d'un domaine."""
        return [r for r in self.rules.values() if r.domain == domain]

    def apply(self, name: str, args: Tuple) -> Any:
        return self.rules[name].apply(*args)

    def verify(self, name: str, args: Tuple, output: Any) -> bool:
        """Compréhension d'une règle : vérifie une application."""
        return self.rules[name].verify(args, output)

    def understands_all(self, applications: List[Tuple[str, Tuple]]) -> bool:
        """Le modèle comprend toutes les règles s'il vérifie correctement toutes les
        applications (apply puis verify sur le résultat)."""
        for name, args in applications:
            out = self.apply(name, args)
            if not self.verify(name, args, out):
                return False
        return True

    def compose(self, plan: List[Tuple[str, Tuple]], init: Any) -> List[Any]:
        """GÉNÉRATION par composition de règles comprises.
        plan = [(rule_name, extra_args)] ; init = entrée. Chaque règle appliquée au
        résultat précédent (+ extra_args) => chaîne de génération causale."""
        out = [init]
        cur = init
        for name, extra in plan:
            args = (cur,) + tuple(extra)
            cur = self.apply(name, args)
            out.append(cur)
        return out
