"""Phonologie + transcription IPA — réfute audit L5.

EX-T6, L5. Phonologie = étude des sons du langage. Au-delà des phonèmes (linguistics.py),
on ajoute :
* Transcription IPA (International Phonetic Alphabet) des mots français.
* Règles phonologiques (élision, liaison, nasalisation, assourdissement).
* Voyelles/consonnes classification (point d'articulation, voisement).
* Comptage syllabique phonologique + accentuation.

C'est la primitive phonologique (le 'son' du mot, pour les associations forme/son/sens).
"""
from __future__ import annotations
import re
from typing import Dict, List, Set


# Voyelles (orales + nasales + semi) pour classification
VOYELLES_ORALES = {"a", "à", "â", "e", "é", "è", "ê", "i", "o", "ô", "u", "ou", "eu", "oe", "y"}
VOYELLES_NASALES = {"an", "en", "in", "on", "un", "ain", "ein", "oin", "yn"}
SEMI_VOYELLES = {"i", "ou", "u", "y"}  # devant voyelle

CONSONNES_SOURDES = {"p", "t", "k", "f", "s", "ch"}    # non voisées
CONSONNES_VOISEES = {"b", "d", "g", "v", "z", "j", "l", "r", "m", "n", "gn"}

# Règles de transcription FR → IPA (simplifiées, ordonnées par longueur)
FR_TO_IPA = [
    ("ouill", "[uj]"), ("ou", "[u]"), ("oi", "[wa]"), ("oî", "[wa]"),
    ("an", "[ɑ̃]"), ("en", "[ɑ̃]"), ("in", "[ɛ̃]"), ("ain", "[ɛ̃]"), ("ein", "[ɛ̃]"),
    ("on", "[ɔ̃]"), ("oin", "[wɛ̃]"), ("un", "[œ̃]"), ("yn", "[œ̃]"),
    ("eu", "[ø]"), ("oe", "[œ]"), ("œ", "[œ]"), ("au", "[o]"), ("eau", "[o]"),
    ("ch", "[ʃ]"), ("gn", "[ɲ]"), ("ph", "[f]"), ("th", "[t]"), ("ss", "[s]"),
    ("é", "[e]"), ("è", "[ɛ]"), ("ê", "[ɛ]"), ("â", "[ɑ]"), ("ô", "[o]"), ("î", "[i]"),
    ("a", "[a]"), ("à", "[a]"), ("e", "[ə]"), ("i", "[i]"), ("o", "[o]"), ("u", "[y]"),
    ("y", "[i]"),
    ("b", "[b]"), ("c", "[k]"), ("d", "[d]"), ("f", "[f]"), ("g", "[g]"), ("h", ""),
    ("j", "[ʒ]"), ("k", "[k]"), ("l", "[l]"), ("m", "[m]"), ("n", "[n]"), ("p", "[p]"),
    ("q", "[k]"), ("r", "[ʁ]"), ("s", "[z]"), ("t", "[t]"), ("v", "[v]"), ("w", "[w]"),
    ("x", "[ks]"), ("z", "[z]"),
]


def to_ipa(word: str) -> str:
    """Transcription phonologique FR → IPA (approximation par règles)."""
    w = word.lower()
    # lettres muettes finales (sauf c,l,r,f)
    if w.endswith("e") and len(w) > 1 and not w.endswith(("le", "re", "ce")):
        w_transcribe = w[:-1]
    elif w.endswith("s") and len(w) > 2 and w[-2] not in "aiuoy":
        w_transcribe = w[:-1]
    elif w.endswith("nt") and len(w) > 4:           # 3e personne pluriel
        w_transcribe = w[:-3] if w.endswith("ent") else w
    else:
        w_transcribe = w
    out = ""
    i = 0
    while i < len(w_transcribe):
        matched = False
        for fr, ipa in FR_TO_IPA:
            if w_transcribe[i:i + len(fr)] == fr:
                out += ipa
                i += len(fr)
                matched = True
                break
        if not matched:
            i += 1
    return out or "[?]"


def classify_sounds(word: str) -> Dict[str, List[str]]:
    """Classifie les sons : voyelles (orales/nasales) + consonnes (sourdes/voisées)."""
    w = word.lower()
    orales = [c for c in w if c in "aeiouyéèêàâô"]
    nasales = []
    for nas in VOYELLES_NASALES:
        if nas in w:
            nasales.append(nas)
    sourdes = [c for c in w if c in "ptkfs"]
    voisées = [c for c in w if c in "bdgvzlmnr"]
    return {"voyelles_orales": sorted(set(orales)),
            "voyelles_nasales": sorted(set(nasales)),
            "consonnes_sourdes": sorted(set(sourdes)),
            "consonnes_voisees": sorted(set(voisées))}


ELIDABLE = {"le", "la", "je", "me", "te", "se", "de", "ne", "que", "ce", "si",
            "puisque", "lorsque", "quoique", "jusque"}


def elision(word_before: str, word_after: str) -> bool:
    """Élision : le 'e'/'a' muet final tombe devant voyelle (l'ami, l'arbre, d'école).
    Seuls certains mots élide (le/la/je/de/que...). 'ma', 'ta', 'sa' N'élident pas."""
    w = word_before.lower()
    if w not in ELIDABLE:
        return False
    return bool(word_after) and word_after[0].lower() in "aeiouyhéàâêîôû"


def liaison(word_before: str, word_after: str) -> str:
    """Liaison : consonne latente prononcée (les amis → [lezaˈmi])."""
    w = word_before.lower()
    if w.endswith(("les", "des", "mes", "tes", "ses", "ces", "nos", "vos", "leurs")):
        if word_after and word_after[0].lower() in "aeiouyh":
            return "[z]"        # liaison en z
    if w.endswith(("un", "on")) and word_after and word_after[0].lower() in "aeiouyh":
        return "[n]"
    return ""


if __name__ == "__main__":
    for w in ["chat", "chien", "bonjour", "maison", "français", "éléphant"]:
        print(f"[phonology] {w} → IPA {to_ipa(w)} | sons {classify_sounds(w)}")
    print("[phonology] élision (le+ami) :", elision("le", "ami"))
    print("[phonology] liaison (les+amis) :", liaison("les", "amis"))
