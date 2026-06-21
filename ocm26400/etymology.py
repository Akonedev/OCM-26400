"""Étymologie / morphèmes / lexèmes — EX-T7, L7-L9 (CRITIQUE audit final).

Étude de l'origine et de la structure des mots :
* Étymologie : racine/radical d'un mot, famille dérivationnelle (mots de même origine).
* Morphèmes : plus petite unité de sens (radical + affixes).
* Lexèmes : unité lexicale (lemme + toutes ses formes fléchies).

Étend linguistics.py (qui fait capture_all) avec l'analyse étymologique dédiée.
Vérifiable : les familles étymologiques sont exactes (racines latines/grecques/germaniques).
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple

# Racines étymologiques (racine → famille de mots FR)
ETYMO_ROOTS: Dict[str, Tuple[str, List[str]]] = {
    "am-": ("latin amare", ["aimer", "amour", "amical", "amitié", "amoureux", "amante"]),
    "luc-/lux-": ("latin lux", ["lumière", "lumineux", "illustrer", "lucide", "élucider"]),
    "duc-": ("latin ducere", ["conduire", "conduit", "éduquer", "déduire", "produire", "séduire"]),
    "port-": ("latin portare", ["porter", "apporter", "support", "transport", "importer", "exporter"]),
    "scrib-/script-": ("latin scribere", ["écrire", "écrit", "description", "inscription", "prescrire"]),
    "vid-/vis-": ("latin videre", ["voir", "visible", "vision", "visage", "réviser", "visiter"]),
    "ject-": ("latin jacere", ["jeter", "projet", "objet", "sujet", "rejet", "trajectoire"]),
    "pend-": ("latin pendere", ["pendre", "pendant", "dépendre", "suspendre", "appendice"]),
    "fac-/fect-": ("latin facere", ["faire", "factory", "facile", "difficile", "parfait", "défaut"]),
    "spec-/spect-": ("latin specere", ["spectacle", "respecter", "inspecter", "aspects", "suspect"]),
    "phon-": ("grec phonê", ["téléphone", "phonétique", "symphonie", "microphone", "gramophone"]),
    "log-": ("grec logos", ["logique", "logiciel", "biologie", "géologie", "dialogue", "catalogue"]),
    "graph-": ("graphein grec", ["graphique", "photographie", "télégraphe", "biographie", "paragraphe"]),
    "therm-": ("grec thermê", ["thermomètre", "thermique", "thermostat", "hypothermie"]),
    "chron-": ("grec chronos", ["chronique", "chronomètre", "synchroniser", "anachronique"]),
}

# Inverse : mot → racine étymologique
_WORD_TO_ROOT: Dict[str, str] = {}
for root, (_, words) in ETYMO_ROOTS.items():
    for w in words:
        _WORD_TO_ROOT[w] = root


def etymology(word: str) -> Dict:
    """Retourne l'étymologie d'un mot : racine, origine, famille."""
    w = word.lower().strip()
    root = _WORD_TO_ROOT.get(w)
    if root:
        origin, family = ETYMO_ROOTS[root]
        return {"word": w, "root": root, "origin": origin, "family": family,
                "in_family": True}
    # recherche par racine dans le mot
    for root, (origin, family) in ETYMO_ROOTS.items():
        if root.rstrip("-") in w or any(w in f for f in family):
            return {"word": w, "root": root, "origin": origin, "family": family,
                    "in_family": w in family}
    return {"word": w, "root": None, "origin": "inconnue", "family": [], "in_family": False}


def etymological_family(word: str) -> List[str]:
    """Famille étymologique (mots de même racine)."""
    info = etymology(word)
    return info["family"]


def morphemes(word: str) -> Dict[str, List[str]]:
    """Décompose un mot en morphèmes : radical + préfixes + suffixes.
    'déshabiller' → dé+s+habill+er (préfixe 'dé', radical 'habill', suffixe 'er')."""
    from .language_primitives import PREFIXES, SUFFIXES
    w = word.lower()
    prefixes = []
    suffixes = []
    stem = w
    # retire préfixes
    changed = True
    while changed and len(stem) > 3:
        changed = False
        for p in sorted(PREFIXES, key=len, reverse=True):
            if stem.startswith(p) and len(stem) - len(p) >= 3:
                prefixes.append(p)
                stem = stem[len(p):]
                changed = True
                break
    # retire suffixes
    changed = True
    while changed and len(stem) > 3:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if stem.endswith(s) and len(stem) - len(s) >= 3:
                suffixes.insert(0, s)
                stem = stem[:-len(s)]
                changed = True
                break
    return {"word": w, "radical": stem, "prefixes": prefixes,
            "suffixes": suffixes, "morphemes": prefixes + [stem] + suffixes}


def lexeme(word: str) -> Dict:
    """Lexème : lemme + formes fléchies. Le lexème est l'unité lexicale."""
    from .language_primitives import lemmatize_fr
    from .morphology_fr import conjugate, G1_ENDINGS
    lemma = lemmatize_fr(word)
    forms = {lemma}  # forme canonique
    # si verbe, ajoute formes conjuguées (présent)
    if lemma.endswith(("er", "ir", "re")):
        for tense in ["présent", "futur"]:
            for p in range(min(3, 6)):  # 3 premières personnes
                f = conjugate(lemma, tense, p)
                if f:
                    forms.add(f)
    return {"word": word, "lexeme": lemma, "forms": sorted(forms),
            "n_forms": len(forms)}


if __name__ == "__main__":
    for w in ["lumière", "conduire", "biologie", "déshabiller", "porter"]:
        e = etymology(w)
        print(f"[étymologie] {w} : racine={e['root']} origine={e['origin']} famille={e['family'][:4]}")
    for w in ["déshabiller", "transporter", "illuminateur"]:
        m = morphemes(w)
        print(f"[morphèmes] {w} : radical={m['radical']} préfixes={m['prefixes']} suffixes={m['suffixes']}")
    l = lexeme("porter")
    print(f"[lexème] porter : formes={l['forms']}")
