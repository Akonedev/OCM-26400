"""Abstraction / catégorisation — M7, R15, EX-B181 (audit final HAUTE).

Capacité cognitive : regrouper des instances sous des concepts abstraits.
* categorize(item, categories) : classe un item dans la meilleure catégorie (par traits).
* abstract(items) : extrait le concept commun (abstraction inductive).
* hierarchy : organiser les catégories en hiérarchie (animal → mammifère → chien).

Loi L6 (association) : l'abstraction = associer des instances à un concept.
Vérifiable : les catégories sont exactes (traits communs identifiés).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# Traits par catégorie (taxonomie)
CATEGORY_TRAITS: Dict[str, Set[str]] = {
    "animal": {"vivant", "mobile", "respire"},
    "mammifère": {"vivant", "mobile", "respire", "poils", "allaiter"},
    "oiseau": {"vivant", "mobile", "respire", "plumes", "voler", "œufs"},
    "poisson": {"vivant", "mobile", "respire", "écailles", "nager", "branchies"},
    "plante": {"vivant", "immobile", "photosynthèse", "racines"},
    "minéral": {"non-vivant", "solide", "inorganique"},
    "outil": {"non-vivant", "fabriqué", "fonction"},
    "véhicule": {"non-vivant", "fabriqué", "mobile", "transport"},
    "nourriture": {"comestible", "énergie"},
    "bâtiment": {"non-vivant", "fabriqué", "immobile", "abri"},
}

# Instances → traits (pour catégorisation)
INSTANCE_TRAITS: Dict[str, Set[str]] = {
    "chien": {"vivant", "mobile", "respire", "poils", "allaiter", "aboie"},
    "chat": {"vivant", "mobile", "respire", "poils", "allaiter", "miaule"},
    "aigle": {"vivant", "mobile", "respire", "plumes", "voler", "œufs"},
    "requin": {"vivant", "mobile", "respire", "écailles", "nager", "branchies"},
    "chêne": {"vivant", "immobile", "photosynthèse", "racines", "écorce"},
    "granite": {"non-vivant", "solide", "inorganique", "cristallin"},
    "marteau": {"non-vivant", "fabriqué", "fonction", "frapper"},
    "voiture": {"non-vivant", "fabriqué", "mobile", "transport", "moteur"},
    "pomme": {"comestible", "énergie", "fruit", "rouge"},
    "maison": {"non-vivant", "fabriqué", "immobile", "abri", "murs"},
}


def categorize(item: str) -> Tuple[str, float]:
    """Catégorise un item par ses traits → (meilleure catégorie, confiance)."""
    traits = INSTANCE_TRAITS.get(item.lower(), set())
    if not traits:
        return ("inconnu", 0.0)
    best_cat, best_score = "inconnu", 0.0
    for cat, cat_traits in CATEGORY_TRAITS.items():
        if not cat_traits:
            continue
        # score de Jaccard (intersection / union)
        intersection = traits & cat_traits
        union = traits | cat_traits
        score = len(intersection) / len(union) if union else 0
        if score > best_score:
            best_cat, best_score = cat, score
    return (best_cat, round(best_score, 3))


def abstract(items: List[str]) -> Dict:
    """Abstraction inductive : extrait le concept commun à plusieurs items.
    'chien', 'chat', 'aigle' → traits communs = {vivant, mobile, respire} → 'animal'."""
    all_traits = [INSTANCE_TRAITS.get(i.lower(), set()) for i in items]
    if not all_traits or not all(all_traits):
        return {"concept": "inconnu", "traits_communs": set(), "catégorie": "inconnu"}
    common = set.intersection(*all_traits)
    # catégorise par les traits communs
    best_cat, best_score = "inconnu", 0.0
    for cat, cat_traits in CATEGORY_TRAITS.items():
        intersection = common & cat_traits
        score = len(intersection) / len(cat_traits) if cat_traits else 0
        if score > best_score:
            best_cat, best_score = cat, score
    return {"concept": best_cat, "traits_communs": sorted(common),
            "catégorie": best_cat, "confiance": round(best_score, 3)}


def hierarchy(category: str) -> List[str]:
    """Hiérarchie taxonomique d'une catégorie (du général au spécifique)."""
    HIERARCHY = {
        "mammifère": ["animal", "mammifère"],
        "oiseau": ["animal", "oiseau"],
        "poisson": ["animal", "poisson"],
        "chien": ["animal", "mammifère", "chien"],
        "chat": ["animal", "mammifère", "chat"],
    }
    return HIERARCHY.get(category.lower(), [category])


if __name__ == "__main__":
    for item in ["chien", "aigle", "requin", "chêne", "voiture"]:
        cat, conf = categorize(item)
        print(f"[abstraction] {item:10s} → {cat} (conf={conf})")
    print()
    abs_result = abstract(["chien", "chat", "aigle"])
    print(f"[abstraction] chien+chat+aigle → {abs_result['concept']} "
          f"(traits: {abs_result['traits_communs']})")
