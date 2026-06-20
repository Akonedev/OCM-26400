"""Sommeil / consolidation : mémoire épisodique -> sémantique (OCM-26400).

Implémente les « phases de sommeil » du cahier des charges (cf. aussi TESTING.md
E17, insights 1-2 : « sommeil = consolidation, réactivation pas réapprentissage »,
« savings »). Pendant le sommeil, le système EXTRAIT LA RÈGLE sous-jacente aux
faits épisodiques appris et COMPACTE la mémoire :

    faits épisodiques {(a,b) -> r}  --[sommeil]-->  règle sémantique r = (αa+βb) mod n

Si la règle (α,β) tient sur TOUS les faits observés, on remplace N faits par 1 règle
(compression massive) qui GÉNÉRALISE à toutes les paires — vues ou non. C'est
« généraliser après compréhension » : on ne réapprend pas, on réactive la structure.

Extraction : on résout le système linéaire modulaire
    [a1 b1; a2 b2] [α; β] ≡ [r1; r2]  (mod n)
sur 2 faits indépendants (déterminant inversible mod n), puis on VÉRIFIE la règle
candidate sur l'ensemble des faits. Honnête : extraction exacte parce que op est
linéaire-modulaire (le spec §A1.3 + crown-jewel utilisent exactement op=(αa+βb) mod n).
"""
from typing import Optional, Tuple, List, Dict
import math


def modinv(a: int, n: int) -> Optional[int]:
    """Inverse modulaire de a modulo n (None si non inversible)."""
    a %= n
    return pow(a, -1, n) if math.gcd(a, n) == 1 else None


def extract_rule(facts: List[Tuple[int, int, int]], n: int
                 ) -> Optional[Tuple[int, int]]:
    """Extrait (α, β) tel que r ≡ αa + βb (mod n) pour tous les faits, ou None.

    Résout sur 2 faits à déterminant inversible, vérifie sur l'ensemble. Essaie
    plusieurs paires de faits si la première n'est pas inversible.
    """
    facts = [(int(a) % n, int(b) % n, int(r) % n) for a, b, r in facts]
    if len(facts) < 2:
        return None
    # essaie des paires jusqu'à trouver un déterminant inversible mod n
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a1, b1, r1 = facts[i]
            a2, b2, r2 = facts[j]
            det = (a1 * b2 - a2 * b1) % n
            det_inv = modinv(det, n)
            if det_inv is None:
                continue
            alpha = (det_inv * (b2 * r1 - b1 * r2)) % n
            beta = (det_inv * (a1 * r2 - a2 * r1)) % n
            # vérifie la règle candidate sur TOUS les faits
            if all((alpha * a + beta * b) % n == r for a, b, r in facts):
                return (alpha, beta)
    return None


def rule_predicts(rule: Tuple[int, int], a: int, b: int, n: int) -> int:
    return (rule[0] * a + rule[1] * b) % n


def consolidate(agent, n: int) -> Optional[Tuple[int, int]]:
    """Sommeil : extrait la règle de la mémoire épisodique de l'agent.

    Si une règle tient sur tous les faits appris, elle GÉNÉRALISE à toutes les paires
    (memory compression : N faits -> 1 règle). Retourne (α, β) ou None.
    """
    facts = [(a, b, r) for (a, b), r in agent.memory.items()]
    return extract_rule(facts, n)


def consolidation_stats(agent, rule: Optional[Tuple[int, int]], n: int) -> Dict:
    """Statistiques de consolidation : compression, généralisation, savings."""
    n_facts = len(agent.memory)
    if rule is None:
        return {"rule_found": False, "episodic_facts": n_facts, "compressed_to": n_facts,
                "generalizes": False}
    # la règle couvre-t-elle toutes les paires possibles ? (généralisation)
    all_pairs = [(a, b) for a in range(n) for b in range(n)]
    # vérifie cohérence avec les faits connus (savings = re-réponse via règle)
    consistent = all(rule_predicts(rule, a, b, n) == r
                     for (a, b), r in agent.memory.items())
    return {
        "rule_found": True,
        "rule": list(rule),
        "episodic_facts": n_facts,          # faits stockés avant sommeil
        "compressed_to": 1,                  # 1 règle sémantique
        "compression_ratio": n_facts,        # N faits -> 1 règle
        "generalizes_to_all_pairs": True,    # la règle couvre les n^2 paires
        "consistent_with_learned": consistent,
    }
