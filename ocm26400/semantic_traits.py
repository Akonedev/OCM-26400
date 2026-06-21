"""Sèmes / traits sémantiques / polysémie — EX-T8, L11 (audit final HAUTE).

Les sèmes sont les plus petites unités de sens d'un mot. La polysémie = un mot
a plusieurs sens selon le contexte. L'homonymie = mots identiques, sens différents.

* semantic_traits(word) : retourne les sèmes (traits de sens) d'un mot.
* disambiguate(word, context) : désambiguïse la polysémie selon le contexte.
* are_synonyms(w1, w2) : 2 mots sont-ils synonymes (sèmes communs) ?
* are_antonyms(w1, w2) : 2 mots sont-ils antonymes ?
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple

# Sèmes par mot (traits sémantiques minimaux)
SEMES: Dict[str, Set[str]] = {
    # animaux
    "chat": {"animal", "félin", "domestique", "vivant", "petit"},
    "chien": {"animal", "canin", "domestique", "vivant", "fidèle"},
    "lion": {"animal", "félin", "sauvage", "vivant", "féroce"},
    "aigle": {"animal", "oiseau", "sauvage", "vol", "rapace"},
    # objets
    "marteau": {"outil", "frapper", "métal", "main"},
    "couteau": {"outil", "couper", "tranchant", "métal"},
    "voiture": {"véhicule", "transport", "moteur", "roues"},
    "vélo": {"véhicule", "transport", "pédales", "écologique"},
    # abstraits
    "amour": {"sentiment", "positif", "affectif", "fort"},
    "haine": {"sentiment", "négatif", "affectif", "fort"},
    "joie": {"émotion", "positif", "passager"},
    "tristesse": {"émotion", "négatif", "passager"},
    "liberté": {"abstrait", "positif", "politique", "droit"},
    "prison": {"lieu", "négatif", "contrainte", "punition"},
    # nourriture
    "pomme": {"fruit", "comestible", "rond", "rouge"},
    "pain": {"aliment", "comestible", "blé", "basique"},
    # couleurs
    "rouge": {"couleur", "chaleureux", "intense"},
    "bleu": {"couleur", "froid", "calme"},
}

# Polysémie : un mot, plusieurs sens selon le contexte
POLYSEMY: Dict[str, Dict[str, Set[str]]] = {
    "vol": {
        "action_de_voler": {"action", "déplacement", "air"},
        "délit": {"crime", "délit", "appropriation"},
    },
    "mine": {
        "exploitation": {"lieu", "extraction", "ressource"},
        "apparence": {"visage", "aspect", "physique"},
        "explosif": {"arme", "explosion", "maritime"},
    },
    "tour": {
        "édifice": {"bâtiment", "haut", "architecture"},
        "rotation": {"mouvement", "rotation", "axe"},
        "visite": {"action", "promenade", "découverte"},
    },
}

# Antonymes
ANTONYMS: List[Tuple[str, str]] = [
    ("amour", "haine"), ("joie", "tristesse"), ("liberté", "prison"),
    ("chaud", "froid"), ("grand", "petit"), ("ouvrir", "fermer"),
    ("monter", "descendre"), ("vivre", "mourir"), ("acheter", "vendre"),
]


def semantic_traits(word: str) -> Set[str]:
    """Retourne les sèmes (traits de sens) d'un mot."""
    return SEMES.get(word.lower(), set())


def disambiguate(word: str, context: str) -> Tuple[str, Set[str]]:
    """Désambiguïse la polysémie d'un mot selon le contexte.
    Retourne (sens_choisi, sèmes_du_sens)."""
    w = word.lower()
    if w not in POLYSEMY:
        return (w, semantic_traits(w))
    senses = POLYSEMY[w]
    ctx = context.lower()
    best_sense, best_score = w, set()
    for sense_name, semes in senses.items():
        score = sum(1 for s in semes if any(s in c for c in ctx.split()))
        if score >= len(best_score):
            best_sense, best_score = sense_name, semes
    return (best_sense, best_score)


def are_synonyms(w1: str, w2: str) -> bool:
    """2 mots sont-ils synonymes (≥50% de sèmes communs) ?"""
    s1, s2 = semantic_traits(w1), semantic_traits(w2)
    if not s1 or not s2:
        return False
    common = s1 & s2
    return len(common) >= len(s1 | s2) * 0.5


def are_antonyms(w1: str, w2: str) -> bool:
    """2 mots sont-ils antonymes (connus) ?"""
    return (w1.lower(), w2.lower()) in ANTONYMS or (w2.lower(), w1.lower()) in ANTONYMS


def semantic_similarity(w1: str, w2: str) -> float:
    """Similarité sémantique [0,1] basée sur les sèmes (Jaccard)."""
    s1, s2 = semantic_traits(w1), semantic_traits(w2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


if __name__ == "__main__":
    print("[sèmes] chat:", semantic_traits("chat"))
    print("[sèmes] disambiguate 'vol' (contexte oiseau):", disambiguate("vol", "l'oiseau vole dans le ciel"))
    print("[sèmes] disambiguate 'vol' (contexte crime):", disambiguate("vol", "il a été arrêté pour vol à main armée"))
    print("[sèmes] synonyms(chat, lion):", are_synonyms("chat", "lion"))
    print("[sèmes] antonyms(amour, haine):", are_antonyms("amour", "haine"))
    print("[sèmes] similarity(voiture, vélo):", round(semantic_similarity("voiture", "vélo"), 2))
