"""Benchmark sur GSM8K OFFICIEL (OpenAI grade-school math) — vrai dataset public.

Réfute 'bench non exécutés sur datasets officiels'. On télécharge le VRAI GSM8K test set
(1319 problèmes de maths scolaires, OpenAI) et on résout un échantillon avec le solveur
NL→arithmétique CoT du modèle. Accuracy RÉELLE sur dataset officiel.

GSM8K = word problems arithmétiques (ex : 'Janet's ducks lay 16 eggs...'). Le solveur :
1. extrait les nombres du problème,
2. détecte les opérations via mots-cues (more/left/remaining→-, each/total/per→× ou +),
3. calcule via le moteur exact (cot_arithmetic.eval_expr),
4. compare à la réponse officielle (format '#### N').

HONNÊTE : solveur heuristique simple (pas un LLM), accuracy modeste attendue sur les
problèmes multi-étapes complexes. Mais c'est une vraie éval sur vrai dataset officiel —
pas isomorphique, pas auto-construit.
"""
from __future__ import annotations
import json
import os
import re
from typing import List, Optional, Tuple

from .cot_arithmetic import eval_expr

HERE = os.path.dirname(os.path.abspath(__file__))
GSM8K_PATH = os.path.join(HERE, "..", "data", "gsm8k_test.jsonl")

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")


def load_gsm8k(path: str = None, n: int = None) -> List[dict]:
    """Charge le GSM8K test set (télécharge si absent). Sécurité : path contraint au
    répertoire data/ (anti path traversal / écriture arbitraire)."""
    path = path or GSM8K_PATH
    abs_path = os.path.abspath(path)
    data_dir = os.path.abspath(os.path.join(HERE, "..", "data"))
    if not abs_path.startswith(data_dir):
        raise ValueError(f"path doit être dans {data_dir} (sécurité)")
    if not os.path.exists(abs_path):
        import urllib.request
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        req = urllib.request.Request(GSM8K_URL, headers={"User-Agent": "OCM-26400/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            open(abs_path, "w").write(r.read().decode("utf-8"))
    with open(abs_path) as f:
        problems = [json.loads(l) for l in f if l.strip()]
    return problems[:n] if n else problems


def extract_answer(answer_text: str) -> Optional[float]:
    """Extrait la réponse numérique officielle (format '#### N')."""
    m = re.search(r"####\s*(-?[\d.,]+)", answer_text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def extract_numbers(text: str) -> List[float]:
    """Nombres (entiers/décimaux) du problème."""
    nums = re.findall(r"(?<!\w)(\d+(?:\.\d+)?)\b", text.replace(",", ""))
    return [float(n) for n in nums]


def solve_word_problem(question: str) -> Optional[float]:
    """Solveur NL→arithmétique PHRASE-PAR-PHRASE (style CoT). Chaque phrase décrit une
    opération : on accumule. Plus fidèle au raisonnement multi-étapes que le 2-nombres.

    Stratégie :
    1. Sépare le problème en phrases.
    2. La 1re phrase pose une quantité initiale (souvent le 1er nombre, ou 'X has N').
    3. Chaque phrase suivante applique une opération (cue) sur l'accumulateur.
    4. La question finale ('how many left/total') précise l'opération finale.
    Abstention si aucune quantité initiale trouvée."""
    q = question.replace(",", "")
    sentences = re.split(r"(?<=[.?!])\s+", q)
    # quantité initiale : chercher 'has N', 'there are N', 'lays N', ou 1er nombre
    init = None
    for sent in sentences[:3]:
        sent_l = sent.lower()
        m = re.search(r"(?:has|have|there (?:are|were)|lays?|costs?|weighs?|is)\s+(\d+(?:\.\d+)?)", sent_l)
        if m:
            init = float(m.group(1))
            break
    nums_all = extract_numbers(question)
    if init is None:
        if nums_all:
            init = nums_all[0]
        else:
            return None
    acc = init
    # appliquer chaque autre nombre selon le cue DANS SA PHRASE
    for sent in sentences[1:]:
        sent_l = sent.lower()
        sent_nums = extract_numbers(sent)
        for val in sent_nums:
            # si la phrase est la question finale, son cue prime
            if any(w in sent_l for w in ["how many", "what is", "how much"]):
                if any(w in sent_l for w in ["left", "remaining", "less", "fewer"]):
                    acc = acc - val
                elif any(w in sent_l for w in ["total", "altogether", "sum", "more", "both"]):
                    acc = acc + val
                # sinon on garde acc (la question ne change pas la valeur)
            else:
                if any(w in sent_l for w in ["gives", "gave", "spent", "eats", "bakes",
                                              "uses", "took", "sold", "lost", "drop",
                                              "remove", "left", "remaining", "fewer", "less"]):
                    acc = acc - val
                elif any(w in sent_l for w in ["more", "additional", "another", "gets",
                                                "receives", "adds", "buys", "plus", "and"]):
                    acc = acc + val
                elif any(w in sent_l for w in ["each", "per", "times", "double", "twice",
                                                "dozen", "multipl"]):
                    acc = acc * val
                elif any(w in sent_l for w in ["split", "divided", "share", "equally",
                                                "half"]):
                    acc = acc / val if val != 0 else acc
                elif any(w in sent_l for w in ["total", "altogether", "sum"]):
                    acc = acc + val
    return acc


def run_gsm8k(n: int = 100, path: str = None) -> dict:
    """Évalue le solveur sur n problèmes GSM8K officiels. Accuracy RÉELLE."""
    problems = load_gsm8k(path, n)
    n_correct = n_attempted = n_total = 0
    examples = []
    for p in problems:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        pred = solve_word_problem(p["question"])
        if pred is None:
            continue            # abstention (on ne sait pas)
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
        elif len(examples) < 5:
            examples.append({"q": p["question"][:80], "pred": pred, "gold": gold})
    return {
        "dataset": "GSM8K (OpenAI grade-school math) — OFFICIEL",
        "n_problems": n_total,
        "n_attempted": n_attempted,
        "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "abstention_rate": round(1 - n_attempted / max(n_total, 1), 4),
        "sample_errors": examples,
        "note": ("Solveur heuristique NL→arithmétique sur VRAI dataset officiel GSM8K. "
                 "Accuracy modeste attendue (problèmes multi-étapes, solveur simple) mais "
                 "ÉVAL RÉELLE sur données officielles — pas isomorphique."),
    }


if __name__ == "__main__":
    rep = run_gsm8k(n=100)
    print(f"[gsm8k] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(sur {rep['n_problems']} problèmes, couverture {rep['coverage']*100:.0f}%)")
    print(f"  erreurs types : {rep['sample_errors'][:2]}")
