"""Sarcasme / ironie — réfute audit M3.

M3. Détection de sarcasme/ironie. Honnête : rule-based (pas un classifieur entraîné,
qui nécessiterait un corpus annoté). Signaux de sarcasme :
* Contraste : sentiment lexical positif + contexte négatif (ou inverse) = ironie.
* Marqueurs : « bien sûr », « évidemment », « bravo », guillemets, points d'exclamation
  en contradiction avec le sens.
* Exagération positive sur un contexte clairement négatif.
Vérifiable : détecte les cas canoniques (« Quelle journée magnifique » après une catastrophe).
"""
from __future__ import annotations
import re
from typing import Dict
from .nlp_tools import sentiment


SARCASM_MARKERS = {"bien sûr", "évidemment", "bravo", "génial", "super", "magnifique",
                   "génialissime", "merci beaucoup", "oh joie", "quelle merveille",
                   "fantastique", "incroyable"}
NEGATIVE_CONTEXT = {"pluie", "pleut", "panne", "accident", "mort", "maladie", "perdu",
                    "échec", "catastrophe", "problème", "douleur", "triste", "colère",
                    "guerre", "tomber", "casser", "cassé", "vacances gâchées", "gâch"}


def detect_sarcasm(text: str) -> Dict:
    """Score de sarcasme ∈ [0,1] + label. Combine : marqueurs + contraste sentiment/contexte."""
    t = text.lower()
    sent = sentiment(text)
    has_marker = any(m in t for m in SARCASM_MARKERS)
    has_negative_ctx = any(w in t for w in NEGATIVE_CONTEXT)
    has_quotes = '"' in text or "«" in text or "»" in text
    exclam = text.count("!") >= 2

    score = 0.0
    reasons = []
    # contraste ironique : sentiment positif MAIS contexte négatif
    if sent["label"] == "positif" and has_negative_ctx:
        score += 0.5
        reasons.append("contraste: sentiment positif + contexte négatif")
    # marqueur exclamatif positif (magnifique/bravo/génial) SUR contexte négatif = sarcasme fort
    if has_marker and has_negative_ctx:
        score += 0.5
        reasons.append("marqueur exclamatif positif sur contexte négatif")
    if has_quotes:
        score += 0.15
        reasons.append("guillemets (distanciation ironique)")
    if exclam and has_negative_ctx:
        score += 0.1
        reasons.append("exclamations sur contexte négatif")
    score = min(score, 1.0)
    label = "sarcastique" if score >= 0.5 else ("peut-être ironique" if score >= 0.25
                                                else "littéral")
    return {"score": round(score, 2), "label": label, "reasons": reasons,
            "sentiment": sent["label"]}


if __name__ == "__main__":
    tests = ["Quelle magnifique journée pour une panne de voiture !",
             "Bravo, encore un échec total, génial !",
             "Le chat dort sur le canapé",                # littéral
             "Oh joie, il pleut encore sur nos vacances"]  # sarcastique
    for t in tests:
        r = detect_sarcasm(t)
        print(f"  [{r['label']:16s}] {t}")
        print(f"     score={r['score']} raisons={r['reasons']}")
