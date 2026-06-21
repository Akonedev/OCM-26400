"""Collocations / locutions figées / mots composés — réfute audit L13.

EX-B197, L13. Collocations = mots qui vont ensemble (forte association statistique),
locutions figées (sens ≠ somme des mots), mots composés.
* Collocations : verbe+nom (prendre décision), adj+nom (erreur grave), nom+nom.
* Locutions figées : « avoir raison », « donner lieu », « coup d'œil » (sens global).
* Mots composés : « grand-mère », « pomme de terre ».
Vérifiable : détection + liste. Base lexicale du langage.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# Collocations FR courantes (verbe+nom, adj+nom) — cooccurrences figées
COLLOCATIONS: Dict[str, List[str]] = {
    "prendre": ["décision", "risque", "soin", "conscience", "fuite", "position", "rendez-vous"],
    "faire": ["attention", "confiance", "face", "preuve", "référence", "signe", "erreur"],
    "avoir": ["raison", "tort", "besoin", "envie", "faim", "peur", "honte", "lieu"],
    "donner": ["lieu", "raison", "ordre", "conseil", "accord", "suite"],
    "erreur": ["grave", "fatale", "commune", "grossière", "frequente"],
    "décision": ["difficile", "important", "stratégique", "rapide"],
    "problème": ["majeur", "complexe", "récurrent", "grave"],
}

# Locutions figées (sens global ≠ somme)
FIXED_EXPRESSIONS: Dict[str, str] = {
    "avoir raison": "être dans le vrai",
    "donner lieu": "provoquer/causer",
    "coup d'oeil": "regard rapide",
    "pomme de terre": "tubercule comestible (patate)",
    "grand_mère": "mère d'un parent",
    "chemin de fer": "réseau ferroviaire",
    "raison du plus fort": "la force prime le droit",
    "pied à terre": "logement secondaire",
}

# Mots composés (à fusionner)
COMPOUND_WORDS = {
    "pomme de terre", "chemin de fer", "coup d'oeil", "pied à terre",
    "grand_mère", "grand_père", "café au lait", "arc en ciel",
}


def detect_collocations(tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Détecte les collocations dans une séquence de tokens. Retourne (mot1, mot2, type)."""
    found = []
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i].lower(), tokens[i + 1].lower()
        # verbe + nom (prendre décision)
        if w1 in COLLOCATIONS and w2 in COLLOCATIONS[w1]:
            found.append((tokens[i], tokens[i + 1], "verbe+nom"))
        # adj + nom inversé (erreur grave)
        for adj_host, adjs in COLLOCATIONS.items():
            if w2 == adj_host and w1 in adjs:
                found.append((tokens[i], tokens[i + 1], "adj+nom"))
    return found


def is_fixed_expression(phrase: str) -> Tuple[bool, str]:
    """La phrase est-elle une locution figée ? Retourne (oui, sens)."""
    p = phrase.lower().strip()
    for expr, sens in FIXED_EXPRESSIONS.items():
        if expr.replace("_", " ") in p or expr in p:
            return True, sens
    return False, ""


def detect_compounds(text: str) -> List[str]:
    """Détecte les mots composés dans un texte."""
    t = text.lower()
    return [c for c in COMPOUND_WORDS if c in t]


if __name__ == "__main__":
    print("[collocations] 'prendre une décision importante' :",
          detect_collocations(["prendre", "une", "décision", "importante"]))
    print("[collocations] 'erreur grave' :", detect_collocations(["erreur", "grave"]))
    print("[locutions] 'avoir raison' :", is_fixed_expression("il a avoir raison"))
    print("[composés] 'pomme de terre' :", detect_compounds("je mange une pomme de terre"))
