"""In-context learning (few-shot) — réfute audit H11.

L'audit H11 : « Few-shot / in-context learning par le core manquant (test 302 hardcodé) ».
Le cahier des charges + paradigme utilisateur : « notre modèle comprend, réfléchit, il
n'a pas besoin de milliards d'exemples ». L'in-context learning (ICL) est PRECISÉMENT
cela : donner k exemples (input→output) en contexte, le modèle infère la règle et
l'applique à une nouvelle entrée — SANS réentraînement.

Mécanisme (compositionnel, fidèle à OCM) :
1. On extrait la règle des k exemples (réutilise sleep.extract_rule : trouve (α,β)
   tel que output = (α·input_a + β·input_b) mod n cohérent sur tous les exemples).
2. On applique la règle extraite au query → prédiction.
3. Si aucune règle ne tient sur les exemples → ABSTENTION (« je ne vois pas le pattern »).

C'est du VRAI in-context learning : apprentissage au moment de l'inférence depuis le
contexte, pas des poids. Généralise à toute règle linéaire mod n (add, mul, linop,
et toutes les variantes (α,β)). Abstention honnête si pattern non reconnu.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from .sleep import extract_rule, rule_predicts


def learn_rule_from_context(examples: List[Tuple[Tuple[int, int], int]],
                            n: int = 11) -> Optional[Tuple[int, int]]:
    """Extrait la règle (α,β) cohérente avec TOUS les exemples du contexte.
    None si aucune règle linéaire mod n ne tient (→ abstention)."""
    facts = [(a, b, r) for (a, b), r in examples]
    return extract_rule(facts, n)


def predict_from_context(examples: List[Tuple[Tuple[int, int], int]],
                         query: Tuple[int, int], n: int = 11
                         ) -> Tuple[Optional[int], Optional[Tuple[int, int]], bool]:
    """In-context learning : apprend la règle des exemples, l'applique au query.
    Retourne (prédiction, règle_apprise, confiant).
    confiant=False → abstention (pattern non reconnu dans le contexte)."""
    rule = learn_rule_from_context(examples, n)
    if rule is None:
        return None, None, False          # abstention : pattern non reconnu
    a, b = query
    return rule_predicts(rule, a, b, n), rule, True


def in_context_accuracy(test_rules: List[Tuple[int, int]], n_examples: int = 5,
                        n_queries: int = 10, n: int = 11, seed: int = 0) -> dict:
    """Évalue l'ICL : pour chaque règle test, génère n_examples de contexte, puis
    n_queries à prédire. Mesure accuracy + taux d'abstention."""
    import random
    rng = random.Random(seed)
    n_correct = n_abstained = n_total = 0
    for rule in test_rules:
        # génère exemples de contexte pour CETTE règle
        exs = []
        for _ in range(n_examples):
            a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
            exs.append(((a, b), rule_predicts(rule, a, b, n)))
        # requêtes (autres entrées)
        for _ in range(n_queries):
            a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
            gold = rule_predicts(rule, a, b, n)
            pred, learned, confiant = predict_from_context(exs, (a, b), n)
            n_total += 1
            if not confiant:
                n_abstained += 1
            elif pred == gold:
                n_correct += 1
    answered = n_total - n_abstained
    return {
        "n_rules": len(test_rules), "n_total": n_total,
        "n_correct": n_correct, "n_abstained": n_abstained,
        "accuracy_on_answered": n_correct / answered if answered else 0.0,
        "coverage": answered / n_total if n_total else 0.0,
        "verdict": ("ICL_WORKS" if n_correct / max(answered, 1) >= 0.95
                    and answered > 0 else "ICL_WEAK"),
    }


if __name__ == "__main__":
    # démo : ICL sur plusieurs règles jamais codées explicitement
    print("[in_context] in-context learning — apprend la règle depuis le contexte")
    test_rules = [(1, 1), (3, 5), (2, 7), (1, 0), (0, 1), (4, 9)]  # (α,β) variés
    for rule in test_rules[:3]:
        # 4 exemples de contexte
        exs = [((a, b), rule_predicts(rule, a, b, 11))
               for a, b in [(1, 2), (3, 1), (0, 5), (2, 2)]]
        pred, learned, conf = predict_from_context(exs, (7, 3), 11)
        gold = rule_predicts(rule, 7, 3, 11)
        print(f"  règle vraie (α,β)={rule} | contexte→appris {learned} | "
              f"query(7,3)→prédit {pred} (or {gold}) {'✓' if pred == gold else '✗'}")
    res = in_context_accuracy(test_rules, n_examples=5, n_queries=10)
    print(f"\n  accuracy ICL : {res['accuracy_on_answered']*100:.1f}% sur répondu | "
          f"couverture {res['coverage']*100:.1f}% | {res['verdict']}")
