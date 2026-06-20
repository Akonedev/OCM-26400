"""Explication structurée — réfute audit M9.

EX-B191, M9. « Pourquoi cette réponse ? » Le modèle JUSTIFIE ses conclusions en
remontant la chaîne de raisonnement. Une explication = (réponse, prémisses, étapes,
règles_appliquées, confiance). C'est l'interprétabilité exigée (le modèle ne répond pas
boîte noire).

* explain(query, answer, evidence) → Explication structurée (citable, vérifiable).
* explain_trace(cot_trace) → explique une chaîne CoT (chaque étape + sa règle).
* Distingue : déduction (règle→conclusion) vs induction (exemples→règle) vs abstention.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Explanation:
    """Explication structurée et vérifiable d'une réponse."""
    query: str
    answer: Any
    reasoning_type: str               # déduction / induction / analogie / abstention
    premises: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)
    confidence: float = 1.0
    citations: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Question : {self.query}", f"Réponse : {self.answer}",
                 f"Type de raisonnement : {self.reasoning_type}",
                 f"Confiance : {self.confidence:.0%}"]
        if self.premises:
            lines.append("Prémisses :")
            lines += [f"  - {p}" for p in self.premises]
        if self.steps:
            lines.append("Étapes :")
            for i, s in enumerate(self.steps, 1):
                lines.append(f"  {i}. {s}")
        if self.rules_applied:
            lines.append("Règles appliquées : " + ", ".join(self.rules_applied))
        if self.citations:
            lines.append("Sources : " + ", ".join(self.citations))
        return "\n".join(lines)


def explain_deduction(query: str, answer: Any, premises: List[str], rule: str,
                      steps: List[str] = None, confidence: float = 0.95) -> Explanation:
    """Explication par DÉDUCTION : prémisses + règle → conclusion."""
    return Explanation(query=query, answer=answer, reasoning_type="déduction",
                       premises=premises, rules_applied=[rule],
                       steps=steps or [], confidence=confidence)


def explain_abstention(query: str, reason: str) -> Explanation:
    """Explication d'ABSTENTION : pourquoi le modèle refuse de répondre."""
    return Explanation(query=query, answer=None, reasoning_type="abstention",
                       premises=[f"incertain: {reason}"], confidence=0.0,
                       steps=["pas assez d'information fiable → 'je ne sais pas'"])


def explain_trace(steps_with_rules: List[tuple], query: str, final_answer: Any
                  ) -> Explanation:
    """Explique une chaîne CoT : chaque étape + la règle/évidence qui la justifie.
    steps_with_rules = [(étape_texte, règle_ou_évidence), ...]"""
    premises = [e for _, e in steps_with_rules if e]
    steps = [s for s, _ in steps_with_rules]
    rules = list({e for _, e in steps_with_rules if e})
    return Explanation(query=query, answer=final_answer, reasoning_type="déduction (CoT)",
                       premises=premises, steps=steps, rules_applied=rules,
                       confidence=1.0 if steps else 0.0)


def explain_with_cot_arithmetic(steps, query: str, final_answer: Any) -> Explanation:
    """Explique une résolution CoT arithmétique (cot_arithmetic.ReasoningStep)."""
    return explain_trace([(s.context, s.expr) for s in steps], query, final_answer)


if __name__ == "__main__":
    # déduction : Socrate est mortel
    e = explain_deduction("Socrate est-il mortel ?", "oui",
                          premises=["Tous les hommes sont mortels", "Socrate est un homme"],
                          rule="modus ponens",
                          steps=["Socrate ∈ hommes", "hommes → mortels", "donc Socrate mortel"])
    print(e.render())
    print()
    print(explain_abstention("Y a-t-il de la vie sur Mars ?", "aucune preuve concluante").render())
