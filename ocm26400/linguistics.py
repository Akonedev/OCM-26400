"""Primitives linguistiques COMPLÈTES — capture en une passe — la base du langage.

Réfute le besoin utilisateur : capturer TOUTES les primitives linguistiques en une fois
(grammaire, vocabulaire, phonèmes, lexique, phonologie, morphologie, étymologie, voyelles,
consonnes, affixes, morphèmes, radical, lexèmes, préfixe/suffixe, flexionnel/dérivationnel,
désinences, affixes grammaticaux/de classe/sémantiques/lexicaux/séparables/tmèse, traits
sémantiques, syntaxe, conjugaison, synonymes, nuances). C'est la BASE du langage.

* capture_all(word) : extrait EN UNE PASSE toutes les caractéristiques linguistiques d'un
  mot (phonèmes, morphèmes, radical, lexème, affixes, étymologie, traits, catégorie) →
  représentation unifiée pour les ASSOCIATIONS (lier forme/son/sens).
* Phonologie : voyelles/consonnes, phonèmes, syllabation.
* Morphologie : morphèmes, radical, lexème, préfixe/suffixe, flexionnel vs dérivationnel.
* Affixes (taxonomie complète) : grammaticaux, de classe, sémantiques, lexicaux,
  séparables, tmèse.
* Étymologie : radical/racine, familles dérivationnelles.
* Désinences : terminaisons flexionnelles.
* Traits sémantiques (sèmes).
* Syntaxe : fonctions (sujet/verbe/objet).

Le paradigme : une fois ces primitives capturées, le modèle ASSOCIE (forme↔son↔sens↔étymo)
→ compréhension profonde. "Tout apprendre en une passe" = capture_all.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .language_primitives import lemmatize_fr, lemmatize_en, inflect_adjective, synonyms


# ============ PHONOLOGIE ============

VOYELLES = set("aeiouyàâéèêëîïôöùûüœæ")
VOYELLES_ORALES = set("aeiouy")
VOYELLES_NASALES = {"an", "in", "on", "un", "en", "ain", "ein"}
CONSONNES = set("bcdfgjklmnpqrstvxz")
CONSONNES_DOUBLES = {"ch", "ph", "th", "gn", "ll", "ss", "tt", "nn", "mm", "rr"}

# Phonèmes français (approximation IPA-lite)
PHONEMES_FR: Dict[str, str] = {
    "a": "/a/", "à": "/a/", "â": "/ɑ/", "e": "/ə/", "é": "/e/", "è": "/ɛ/",
    "ê": "/ɛ/", "i": "/i/", "o": "/o/", "ô": "/o/", "u": "/y/", "ou": "/u/",
    "an": "/ɑ̃/", "in": "/ɛ̃/", "on": "/ɔ̃/", "un": "/œ̃/", "eu": "/ø/",
    "ch": "/ʃ/", "gn": "/ɲ/", "ll": "/j/", "ph": "/f/", "th": "/t/",
    "b": "/b/", "c": "/k/", "d": "/d/", "f": "/f/", "g": "/g/", "k": "/k/",
    "m": "/m/", "n": "/n/", "p": "/p/", "r": "/ʁ/", "s": "/s/", "t": "/t/",
    "v": "/v/", "z": "/z/", "j": "/ʒ/", "w": "/w/", "x": "/ks/", "y": "/j/",
}


def is_vowel(c: str) -> bool:
    return c.lower() in VOYELLES


def phonemes(word: str) -> List[str]:
    """Décompose un mot en phonèmes (approximation). 'chat' → ['/ʃ/','/a/']."""
    w = word.lower()
    out = []
    i = 0
    while i < len(w):
        if not w[i].isalpha():
            i += 1
            continue
        # digramme/trigramme ?
        matched = False
        for n in (3, 2):
            chunk = w[i:i + n]
            if chunk in PHONEMES_FR:
                out.append(PHONEMES_FR[chunk])
                i += n
                matched = True
                break
        if not matched:
            if w[i] in PHONEMES_FR:
                out.append(PHONEMES_FR[w[i]])
            i += 1
    return out


def syllables(word: str) -> List[str]:
    """Syllabation française (approximation : coupe devant voyelle après consonne)."""
    w = word.lower()
    if len(w) <= 2:
        return [w]
    syls = re.findall(r"[^aeiouyàâéèêëîïôöùûüoe]*[aeiouyàâéèêëîïôöùûüoe]+", w)
    return [s for s in syls if s]


def vowel_consonant_count(word: str) -> Dict[str, int]:
    w = word.lower()
    return {"voyelles": sum(1 for c in w if c in VOYELLES),
            "consonnes": sum(1 for c in w if c in CONSONNES)}


# ============ MORPHOLOGIE (morphèmes, radical, lexème, affixes) ============

# Affixes français courants (préfixes / suffixes)
PREFIXES: Dict[str, str] = {
    "dé": "négation/retire", "re": "répétition", "in": "négation", "im": "négation",
    "anti": "opposition", "auto": "soi-même", "co": "ensemble", " pré": "avant",
    "sur": "excès", "sous": "manque", "super": "au-dessus", "trans": "à travers",
    "tri": "trois", "bi": "deux", "uni": "un", "multi": "plusieurs",
}
SUFFIXES: Dict[str, str] = {
    "tion": "action (dériv.", "ment": "manière (adv)", "able": "possibilité (adj)",
    "ité": "qualité (nom)", "eux": "pourvu de (adj)", "iste": "adepte (nom)",
    "iser": "rendre (verbe)", "esse": "état (nom)", "age": "action (nom)",
    "oire": "lieu/relatif", "al": "relatif à (adj)", "ique": "relatif à (adj)",
    "ation": "action", "ement": "adv", "erie": "métier/coll.",
}
# Affixes flexionnels (désinences) vs dérivationnels
INFLECTIONAL_SUFFIXES = {"e", "es", "ent", "ons", "ez", "ai", "as", "a", "ais", "ait",
                         "s", "x", "ment", "er", "ir", "re", "é", "ée"}  # désinences


@dataclass
class MorphologicalBreakdown:
    """Décomposition morphologique d'un mot."""
    word: str
    radical: str                       # radical (lexème)
    prefixes: List[str] = field(default_factory=list)
    suffixes: List[str] = field(default_factory=list)
    lexeme: str = ""                   # lemme/lexème
    is_inflected: bool = False
    derivation_type: str = ""          # flexionnel / dérivationnel / none

    def morphemes(self) -> List[str]:
        return self.prefixes + [self.radical] + self.suffixes


def morphemes_of(word: str) -> MorphologicalBreakdown:
    """Décompose un mot en morphèmes : préfixe(s) + radical + suffixe(s).
    'défaire' → dé + fair(e). 'rapidement' → rapide + ment."""
    w = word.lower()
    prefixes = []
    suffixes = []
    # préfixes (le plus long d'abord)
    rest = w
    changed = True
    while changed and len(rest) > 4:
        changed = False
        for p in sorted(PREFIXES, key=len, reverse=True):
            if rest.startswith(p) and len(rest) - len(p) >= 3:
                prefixes.append(p)
                rest = rest[len(p):]
                changed = True
                break
    # suffixes (le plus long d'abord)
    stem = rest
    changed = True
    while changed and len(stem) > 4:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if stem.endswith(s) and len(stem) - len(s) >= 3:
                suffixes.insert(0, s)
                stem = stem[:-len(s)]
                changed = True
                break
    lexeme = lemmatize_fr(word) if word[0:1].islower() else word.lower()
    # type de dérivation
    der_type = "none"
    if suffixes:
        der_type = "flexionnel" if any(s in INFLECTIONAL_SUFFIXES for s in suffixes) else "dérivationnel"
    return MorphologicalBreakdown(
        word=w, radical=stem, prefixes=prefixes, suffixes=suffixes,
        lexeme=lexeme, is_inflected=(lexeme != w), derivation_type=der_type)


# ============ AFFIXES — taxonomie complète ============

AFFIX_TYPES = {
    "grammatical": {"s", "x", "e", "es", "ent", "ons", "ez"},  # marquent la grammaire
    "de_classe": {"able", "ité", "iser", "tion"},               # changent la catégorie
    "sémantique": {"re", "dé", "in", "anti", "sur"},            # changent le sens
    "lexical": {"iste", "erie", "age"},                         # créent un nouveau mot
    "séparable": {"", },                                        # (allemand/néerlandais, conceptuel)
}


def classify_affix(affix: str) -> List[str]:
    """Classifie un affixe selon la taxonomie (grammatical/de classe/sémantique/lexical)."""
    types = []
    for t, members in AFFIX_TYPES.items():
        if affix in members:
            types.append(t)
    return types or ["non_classé"]


def tmesis_analysis(word: str) -> Dict:
    """Tmèse : insertion d'un élément à l'intérieur d'un mot (ex FR 'aujourd'hui' = à+le+jour+hui,
    ou 'extraordinaire'). Détecte les mots composés par apostrophe."""
    if "'" in word:
        parts = word.split("'")
        return {"has_tmesis": True, "parts": parts,
                "note": "tmèse (séparation par apostrophe)"}
    return {"has_tmesis": False, "parts": [word]}


# ============ ÉTYMOLOGIE ============

# Familles dérivationnelles (racine → famille de mots)
ETYMOLOGY_FAMILIES: Dict[str, List[str]] = {
    "aim": ["aimer", "amour", "amical", "amitié", "amoureux"],
    "luc/lux": ["lumière", "lumineux", "illustrer", "lucide"],
    "duc": ["conduire", "conduit", "éduquer", "deduire"],
    "port": ["porter", "apporter", "support", "transport", "importer"],
    "joc/jou": ["jeu", "jouer", "joyeux", "joie"],
    "vid": ["vide", "vider", "évider", "viduité"],
}


def etymology_family(word: str) -> List[str]:
    """Retourne la famille étymologique (mots de même racine)."""
    w = word.lower()
    for root, family in ETYMOLOGY_FAMILIES.items():
        if root.split("/")[0] in w or any(w in f for f in family):
            return family
    return []


# ============ TRAITS SÉMANTIQUES (sèmes) ============

SEMANTIC_TRAITS: Dict[str, Set[str]] = {
    "chat": {"animal", "félin", "domestique", "vivant"},
    "chien": {"animal", "canin", "domestique", "vivant"},
    "robot": {"machine", "artificiel", "intelligent"},
    "amour": {"sentiment", "abstrait", "positif"},
    "guerre": {"conflit", "abstrait", "négatif"},
}


def semantic_traits(word: str) -> Set[str]:
    return SEMANTIC_TRAITS.get(word.lower(), set())


# ============ SYNTAXE (fonctions) ============

def syntactic_role(word: str, lexeme: str = "") -> str:
    """Rôle syntaxique approximatif d'un mot (verbe/nom/adj/adv/déterminant)."""
    w = word.lower()
    l = lexeme or w
    if l in {"le", "la", "les", "un", "une", "des", "du", "ce", "cet", "cette", "mon", "ton", "son"}:
        return "déterminant"
    if l in {"et", "ou", "mais", "donc", "car", "avec", "pour", "dans", "sur"}:
        return "conjonction/préposition"
    if l.endswith(("er", "ir", "re")) and len(l) > 3:
        return "verbe"
    if l.endswith(("ment",)):
        return "adverbe"
    if l in {"bon", "grand", "petit", "beau", "vieux", "rouge"} or l.endswith(("eux", "able", "al", "if")):
        return "adjectif"
    return "nom"


# ============ CAPTURE EN UNE PASSE (représentation unifiée pour associations) ============

@dataclass
class WordCapture:
    """TOUTES les primitives linguistiques d'un mot, capturées en UNE passe.
    Sert de base aux ASSOCIATIONS (forme ↔ son ↔ sens ↔ étymo ↔ morpho)."""
    word: str
    lexeme: str
    category: str                       # rôle syntaxique
    phonemes: List[str]
    syllables: List[str]
    vowel_consonant: Dict[str, int]
    morphemes: MorphologicalBreakdown
    derivation_type: str
    affix_classes: List[str]
    etymology: List[str]
    semantic_traits: Set[str]
    synonyms: List[str]
    tmesis: Dict

    def to_dict(self) -> dict:
        return {
            "word": self.word, "lexeme": self.lexeme, "category": self.category,
            "phonemes": self.phonemes, "syllables": self.syllables,
            "vowel_consonant": self.vowel_consonant,
            "morphemes": [self.morphemes.morphemes()],
            "radical": self.morphemes.radical, "prefixes": self.morphemes.prefixes,
            "suffixes": self.morphemes.suffixes, "derivation_type": self.derivation_type,
            "affix_classes": self.affix_classes, "etymology": self.etymology,
            "semantic_traits": sorted(self.semantic_traits),
            "synonyms": self.synonyms, "tmesis": self.tmesis,
        }


def capture_all(word: str) -> WordCapture:
    """CAPTURE EN UNE PASSE : extrait toutes les primitives linguistiques d'un mot.
    C'est la base pour les associations (lier forme/son/sens/étymo/morpho)."""
    lex = lemmatize_fr(word) if word and word[0].islower() else word.lower()
    morpho = morphemes_of(word)
    aff_classes = []
    for s in morpho.prefixes + morpho.suffixes:
        aff_classes.extend(classify_affix(s))
    return WordCapture(
        word=word.lower(), lexeme=lex, category=syntactic_role(word, lex),
        phonemes=phonemes(word), syllables=syllables(word),
        vowel_consonant=vowel_consonant_count(word),
        morphemes=morpho, derivation_type=morpho.derivation_type,
        affix_classes=sorted(set(aff_classes)),
        etymology=etymology_family(word),
        semantic_traits=semantic_traits(lex),
        synonyms=synonyms(lex), tmesis=tmesis_analysis(word),
    )


if __name__ == "__main__":
    import json
    for w in ["rapidement", "défaire", "aujourd'hui", "chat"]:
        print(f"\n[capture_all] '{w}' :")
        c = capture_all(w)
        d = c.to_dict()
        for k in ["lexeme", "category", "phonemes", "syllables", "radical",
                  "prefixes", "suffixes", "derivation_type", "affix_classes",
                  "etymology", "semantic_traits", "tmesis"]:
            print(f"  {k}: {d[k]}")
