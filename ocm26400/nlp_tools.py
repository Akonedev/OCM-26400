"""Outils NLP — traduction / sentiment / résumé — capacités NLP classiques.

* M23 Traduction FR↔EN : dictionnaire de phrases + mots fréquents. Traduction mot-à-mot
  + expressions courantes. Honnête : phrase-based (pas un NMT entraîné, qui nécessiterait
  un corpus parallèle) — couvre le vocabulaire courant.
* M24 Analyse de sentiment : lexique polarisé (mots positifs/négatifs) → score + label.
  Honnête : lexicon-based (pas un classifieur entraîné).
* Résumé extractif : sélection des phrases les + informatives (fréquence de mots,
  TextRank-lite). Honnête : extractif (pas abstrait).
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple

# ---- M23 Traduction (dictionnaire FR↔EN, vocabulaire courant) ----
TRANSLATION_DICT: Dict[str, str] = {
    "bonjour": "hello", "salut": "hi", "merci": "thank you", "oui": "yes",
    "non": "no", "chat": "cat", "chien": "dog", "maison": "house", "eau": "water",
    "pain": "bread", "livre": "book", "soleil": "sun", "lune": "moon", "arbre": "tree",
    "fleur": "flower", "amour": "love", "temps": "time", "jour": "day", "nuit": "night",
    "homme": "man", "femme": "woman", "enfant": "child", "ami": "friend",
    "manger": "eat", "boire": "drink", "dormir": "sleep", "voir": "see", "parler": "speak",
    "grand": "big", "petit": "small", "bon": "good", "mauvais": "bad", "beau": "beautiful",
    "je": "I", "tu": "you", "il": "he", "elle": "she", "nous": "we", "vous": "you",
    "le": "the", "la": "the", "les": "the", "un": "a", "une": "a", "et": "and",
    "ou": "or", "mais": "but", "avec": "with", "pour": "for", "dans": "in", "sur": "on",
    "monde": "world", "ordinateur": "computer", "intelligence": "intelligence",
    "artificielle": "artificial", "modèle": "model", "apprentissage": "learning",
}
_REVERSE_DICT = {v.lower(): k for k, v in TRANSLATION_DICT.items()}


def translate_word(word: str, to_en: bool = True) -> str:
    """Traduit un mot (FR→EN si to_en, EN→FR sinon). LEMMATISE d'abord (feedback : un
    verbe conjugué comme 'dort' doit passer par le lemme 'dormir' avant traduction).
    Abstention (retourne le mot) si inconnu."""
    from .language_primitives import lemmatize_fr, lemmatize_en
    w = word.lower().strip(".,!?;:")
    # lemmatise d'abord : "mange"→"manger", "running"→"run", "dort"→table
    lemma = lemmatize_fr(w) if to_en else lemmatize_en(w)
    if to_en:
        return TRANSLATION_DICT.get(lemma, TRANSLATION_DICT.get(w, word))
    return _REVERSE_DICT.get(lemma, _REVERSE_DICT.get(w, word))


def translate(text: str, to_en: bool = True) -> str:
    """Traduit une phrase mot-à-mot (FR↔EN). Honnête : pas de grammaire/reordering avancé."""
    words = re.findall(r"\w+|[.,!?;:]", text)
    out = []
    for w in words:
        if w in ".,!?;:":
            out.append(w)
        else:
            out.append(translate_word(w, to_en))
    result = " ".join(out)
    # colle la ponctuation
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return result


# ---- M24 Sentiment (lexique polarisé) ----
POSITIVE_WORDS = {"bon", "excellent", "génial", "super", "heureux", "aimer", "merci",
                  "beautiful", "good", "great", "love", "happy", "excellent", "amazing",
                  "merveilleux", "fantastique", "parfait", "succès", "gagner"}
NEGATIVE_WORDS = {"mauvais", "terrible", "triste", "détester", "haine", "échec",
                  "bad", "awful", "sad", "hate", "terrible", "worst", "perdre",
                  "pire", "horrible", "nul", "colère", "peur"}


def sentiment(text: str) -> Dict[str, object]:
    """Analyse de sentiment : score ∈ [-1, 1] + label. Lexicon-based."""
    words = re.findall(r"\w+", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return {"score": 0.0, "label": "neutre", "n_pos": 0, "n_neg": 0}
    score = (pos - neg) / total
    label = "positif" if score > 0.2 else ("négatif" if score < -0.2 else "mitigé")
    return {"score": round(score, 3), "label": label, "n_pos": pos, "n_neg": neg}


# ---- Résumé extractif (fréquence de mots) ----
STOP_WORDS = {"le", "la", "les", "un", "une", "de", "du", "et", "ou", "mais", "the",
              "a", "an", "is", "are", "of", "to", "in", "on", "with", "for", "que"}


def summarize(text: str, n_sentences: int = 2) -> str:
    """Résumé extractif : score les phrases par somme des fréquences de mots (hors stop words),
    garde les top-n. TextRank-lite."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s) > 5]
    if len(sentences) <= n_sentences:
        return text
    # fréquences
    words = [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in STOP_WORDS]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    max_f = max(freq.values()) if freq else 1
    freq = {k: v / max_f for k, v in freq.items()}
    # score par phrase
    scored = []
    for i, s in enumerate(sentences):
        sw = [w.lower() for w in re.findall(r"\w+", s) if w.lower() in freq]
        score = sum(freq[w] for w in sw) / max(len(sw), 1)
        scored.append((score, i, s))
    # top-n par score, ordre original
    top = sorted(scored, key=lambda x: -x[0])[:n_sentences]
    top.sort(key=lambda x: x[1])         # re-ordre d'apparition
    return " ".join(s for _, _, s in top)


if __name__ == "__main__":
    print("[nlp] traduire FR→EN :", translate("le chat mange et dort dans la maison"))
    print("[nlp] traduire EN→FR :", translate("the cat is beautiful", to_en=False))
    print("[nlp] sentiment :", sentiment("ce film est génial et merveilleux, j'aime"))
    print("[nlp] sentiment négatif :", sentiment("c'est terrible et horrible, quelle haine"))
    txt = ("Le chat dort sur le canapé. Le canapé est très confortable et grand. "
           "Le chat aime le confort. La nuit il chasse les souris.")
    print("[nlp] résumé :", summarize(txt, 2))
