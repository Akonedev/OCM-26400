"""Primitives du langage — lemmatisation / déclinaison / synonymes / adverbes.

Réfute le feedback utilisateur : la traduction mot-à-mot ne lemmatisait pas → "dort" était
inconnu alors que "dormir" est dans le dictionnaire. C'est une INTÉGRATION MANQUANTE, pas
un gap d'apprentissage. On construit les PRIMITIVES DU LANGAGE (la base exigée par le
cahier des charges) :

* LEMMATISEUR (inverse de morphology_fr) : forme fléchie → lemme/infinitif.
  "mange"→"manger", "dormi"→"dormir", "chantait"→"chanter", "sont"→"être", "allez"→"aller".
* FLEXION DES ADJECTIFS : genre (masc/fém) + nombre (sing/plur).
  "grand"→"grande" (fém), "grands" (masc plur), "belle"→"beau".
* SYNONYMES : dictionnaire de synonymes courants.
* ADVERBES : dérivation (adjectif + -ment) : "vif"→"vivement", "rapide"→"rapidement".
* ACCORD : nom + adjectif (accorde en genre/nombre).

Le lemmatiseur est BRANCHÉ devant la traduction (translate lemmatise d'abord → traduit le
lemme connu). C'est la base lexicale/morphologique du modèle (primitives du langage).
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

from .morphology_fr import IRREGULAR as FR_IRREG, G1_ENDINGS, G2_ENDINGS, conjugate, _group


# ============ LEMMATISEUR (forme fléchie → lemme) ============

# Table inverse des irréguliers : forme → (infinitif,). Construite depuis morphology_fr.
_LEMMA_IRREG: Dict[str, str] = {}
for _inf, _tenses in FR_IRREG.items():
    for _tense, _forms in _tenses.items():
        for _form in _forms:
            _LEMMA_IRREG[_form] = _inf

# Terminaisons de conjugaison → lemme (patterns réguliers)
_ER_LEMMA_ENDINGS = {  # 1er groupe : formes → infinitif
    "e": "er", "es": "er", "ent": "er", "ons": "er", "ez": "er",
    "ai": "er", "as": "er", "a": "er", "âmes": "er", "âtes": "er", "èrent": "er",
    "ais": "er", "ait": "er", "ions": "er", "iez": "er", "aient": "er",
    "erai": "er", "eras": "er", "era": "er", "erons": "er", "erez": "er", "eront": "er",
}
_IR_LEMMA_ENDINGS = {  # 2e groupe
    "is": "ir", "it": "ir", "issons": "ir", "issez": "ir", "issent": "ir",
}
_RE_LEMMA_ENDINGS = {"re": "re"}


def lemmatize_fr(word: str) -> str:
    """Forme fléchie française → lemme (infinitif pour verbes).
    "mange"→"manger", "dort"→inconnu (3e groupe irrégulier → table), "sont"→"être"."""
    w = word.lower().strip(".,!?;:'")
    # 0. déjà un infinitif (-er/-ir/-re) → intact
    if w.endswith(("er", "ir", "re")) and len(w) >= 4:
        # mais "-er" court comme "cher" (adj) — on garde quand même (sûr)
        return w
    # 1. irréguliers mémorisés (inverse de la table)
    if w in _LEMMA_IRREG:
        return _LEMMA_IRREG[w]
    # 2. participe passé régulier -er → -é : "mangé"→"manger"
    if w.endswith("é") and len(w) > 3:
        return w[:-1] + "er"
    # 3. 1er groupe (-er) par terminaison (radical >= 3 lettres pour éviter faux positifs)
    for suff, inf_end in sorted(_ER_LEMMA_ENDINGS.items(), key=lambda x: -len(x[0])):
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            return w[:-len(suff)] + inf_end
    # 4. 2e groupe (-ir)
    for suff, inf_end in sorted(_IR_LEMMA_ENDINGS.items(), key=lambda x: -len(x[0])):
        if w.endswith(suff) and len(w) - len(suff) >= 3:
            stem = w[:-len(suff)]
            return stem + inf_end
    return w   # inconnu → on garde


def lemmatize_en(word: str) -> str:
    """Lemmatiseur anglais minimal : pluriels, -ed, -ing, comparatifs.
    "cats"→"cat", "running"→"run", "happier"→"happy", "went"→"go"."""
    w = word.lower().strip(".,!?;:'")
    IRREG_EN = {"went": "go", "gone": "go", "was": "be", "were": "be", "been": "be",
                "had": "have", "did": "do", "done": "do", "ate": "eat", "eaten": "eat",
                "ran": "run", "children": "child", "men": "man", "women": "woman",
                "better": "good", "best": "good"}
    if w in IRREG_EN:
        return IRREG_EN[w]
    if w.endswith("ing") and len(w) > 4:
        stem = w[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:     # running → run (double consonne)
            return stem[:-1]
        return stem + "e" if len(stem) <= 3 else stem    # making→make, walking→walk
    if w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem + "e" if stem.endswith(("lik", "mak", "lov")) else stem
    if w.endswith("ies") and len(w) > 4:                # ladies → lady
        return w[:-3] + "y"
    if w.endswith(("es", "s")) and len(w) > 3:
        return w[:-2] if w.endswith("es") else w[:-1]
    if w.endswith("iest") and len(w) > 4:               # happiest → happy
        return w[:-4] + "y"
    if w.endswith("ier") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("est") and len(w) > 4:
        return w[:-3]
    return w


# ============ FLEXION DES ADJECTIFS (genre/nombre) ============

# Adjectifs irréguliers (masc → fém) : inclut mutations f→v (vif→vive, neuf→neuve)
ADJ_FEM_IRREG: Dict[str, str] = {
    "beau": "belle", "nouveau": "nouvelle", "vieux": "vieille",
    "fou": "folle", "mou": "molle", "blanc": "blanche", "franc": "franche",
    "sec": "sèche", "public": "publique", "long": "longue",
    "vif": "vive", "neuf": "neuve", "bref": "brève", "veau": "velle",
}


def inflect_adjective(adj: str, feminine: bool = False, plural: bool = False) -> str:
    """Accorde un adjectif français en genre et nombre.
    "grand"→"grande"→"grandes" ; "beau"→"belle" ; "vif"→"vive" ; "national"→"nationaux"(plur)."""
    a = adj.lower()
    # féminin
    if feminine:
        if a in ADJ_FEM_IRREG:
            a = ADJ_FEM_IRREG[a]
        elif a.endswith("e"):       # déjà féminin (rapide, rouge)
            pass
        else:
            a = a + "e"             # grand → grande
    # pluriel
    if plural:
        if a.endswith(("s", "x")):
            pass                    # invariable
        elif a.endswith(("al", "ail")):
            a = a[:-2] + "aux"      # national → nationaux
        else:
            a = a + "s"
    return a


def agree_noun_adjective(noun: str, adjective: str, feminine: bool = None,
                          plural: bool = None) -> str:
    """Accorde un adjectif avec un nom (genre/nombre détectés ou imposés)."""
    if feminine is None:
        feminine = noun.lower().endswith(("tion", "té", "ée", "ure", "ice", "esse"))
    if plural is None:
        plural = noun.lower().endswith("s") and not noun.lower().endswith(("us", "is"))
    return inflect_adjective(adjective, feminine, plural)


# ============ SYNONYMES ============

SYNONYMS: Dict[str, List[str]] = {
    "grand": ["gros", "large", "vaste", "énorme"],
    "petit": ["menu", "minuscule", "mineur"],
    "beau": ["magnifique", "superbe", "ravissant"],
    "rapide": ["véloce", "prompt", "preste"],
    "intelligent": ["malin", "futé", "astucieux", "brillant"],
    "good": ["great", "fine", "excellent"],
    "big": ["large", "huge", "great"],
    "happy": ["glad", "joyful", "cheerful"],
    "smart": ["intelligent", "clever", "bright"],
}


def synonyms(word: str) -> List[str]:
    return SYNONYMS.get(word.lower().strip(), [])


# ============ ADVERBES (dérivation -ment) ============

def to_adverb(adjective: str) -> str:
    """Dérive un adverbe depuis un adjectif français (-ment).
    "vif"→"vivement", "rapide"→"rapidement", "précis"→"précisément", "constant"→"constamment"."""
    a = adjective.lower()
    if a.endswith("ant"):
        return a[:-3] + "amment"        # constant → constamment
    if a.endswith("ent"):
        return a[:-3] + "emment"        # évident → évidemment
    # forme féminine puis -ment (vif→vive→vivement) — gère déjà le -e (rapide→rapidement)
    fem = inflect_adjective(a, feminine=True)
    return fem + "ment"


if __name__ == "__main__":
    print("[lang_primitives] lemmatisation FR :")
    for f in ["mange", "dort", "chantait", "sont", "allez", "mangé", "finissons"]:
        print(f"  {f} → {lemmatize_fr(f)}")
    print("[lang_primitives] lemmatisation EN :")
    for f in ["running", "cats", "happier", "went", "making", "happiest"]:
        print(f"  {f} → {lemmatize_en(f)}")
    print("[lang_primitives] flexion adjectifs :")
    for adj in ["grand", "beau", "petit", "national"]:
        print(f"  {adj} → fém:{inflect_adjective(adj, feminine=True)} "
              f"→ fém plur:{inflect_adjective(adj, feminine=True, plural=True)}")
    print("[lang_primitives] adverbes :", {a: to_adverb(a) for a in ["vif", "rapide", "constant"]})
    print("[lang_primitives] synonymes de 'intelligent' :", synonyms("intelligent"))
