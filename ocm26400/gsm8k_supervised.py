"""Solveur GSM8K SUPERVISÉ (k-NN sur le train set) — paradigme 'peu d'exemples'.

Au lieu d'un solveur règle-based (plafond ~5%), on APPREND du train set officiel GSM8K
(7473 problèmes + CoT annoté <<>>) :
1. Pour chaque problème, extraire une empreinte (nb de nombres, mots-clés d'opérations).
2. Signature d'opérations = séquence des opérateurs du CoT (M/D/A/S).
3. Pour un problème TEST : trouver les k problèmes TRAIN les + similaires (cosinus sur
   empreintes), voter la signature majoritaire, l'appliquer aux nombres extraits du test.

C'est du learning from examples (7K ≠ milliards), aligné avec le paradigme. Mesure si
l'apprentissage supervisé dépasse le rule-based sur GSM8K officiel.
"""
from __future__ import annotations
import json
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from .gsm8k_bench import load_gsm8k, extract_numbers, extract_answer

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(HERE, "..", "data", "gsm8k_train.jsonl")

OP_KEYWORDS = {
    "A": ["total", "altogether", "sum", "more", "both", "additional", "plus", "another",
          "combined", "together", "and"],
    "S": ["left", "remaining", "gives", "gave", "spent", "fewer", "less", "minus", "took",
          "eats", "uses", "sold", "lost", "remove", "drop", "away", "were"],
    "M": ["each", "per", "dozen", "times", "double", "twice", "multipl", "of", "every"],
    "D": ["split", "divided", "share", "equally", "half", "third", "quarter", "group"],
}


def operation_signature(answer: str) -> str:
    """Signature d'opérations depuis le CoT annoté (<<expr>>). 'MMA' = mul-mul-add."""
    exprs = re.findall(r"<<([^>]+)>>", answer)
    sig = []
    for e in exprs:
        if "*" in e:
            sig.append("M")
        elif "/" in e:
            sig.append("D")
        elif "+" in e:
            sig.append("A")
        elif "-" in e:
            sig.append("S")
    return "".join(sig)


def fingerprint(question: str) -> List[float]:
    """Empreinte du problème : #nombres (buckets) + scores de mots-clés par opération."""
    q = question.lower()
    nums = extract_numbers(question)
    n_nums = min(len(nums), 8)                 # bucket 0-8 nombres
    kw = {op: sum(q.count(w) for w in words) for op, words in OP_KEYWORDS.items()}
    # features : buckets de nb de nombres + 4 scores d'opérations
    feats = [1.0 if n_nums == i else 0.0 for i in range(9)]
    feats += [float(kw["A"]), float(kw["S"]), float(kw["M"]), float(kw["D"])]
    return feats


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def apply_signature(sig: str, nums: List[float]) -> Optional[float]:
    """Applique une signature d'opérations aux nombres (gauche→droite). Abstient si
    pas assez de nombres pour la signature."""
    if not sig or not nums:
        return None
    acc = nums[0]
    ni = 1
    for op in sig:
        if ni >= len(nums):
            # réutiliser le dernier nombre si signature + longue que les nombres
            val = nums[-1]
        else:
            val = nums[ni]
            ni += 1
        if op == "M":
            acc = acc * val
        elif op == "D":
            acc = acc / val if val != 0 else acc
        elif op == "A":
            acc = acc + val
        elif op == "S":
            acc = acc - val
    return acc


class GSM8KSupervisedSolver:
    """Solveur k-NN supervisé sur le train set GSM8K."""

    def __init__(self, train_path: str = None, k: int = 7, max_train: int = 2000):
        self.k = k
        self.train_fps: List[List[float]] = []
        self.train_sigs: List[str] = []
        self._load(train_path, max_train)

    def _load(self, train_path: str, max_train: int):
        path = train_path or TRAIN_PATH
        abs_path = os.path.abspath(path)
        data_dir = os.path.abspath(os.path.join(HERE, "..", "data"))
        if not abs_path.startswith(data_dir):
            raise ValueError("path doit être dans data/")
        if not os.path.exists(abs_path):
            return       # pas de train → solver vide (abstention)
        probs = [json.loads(l) for l in open(abs_path) if l.strip()][:max_train]
        for p in probs:
            self.train_fps.append(fingerprint(p["question"]))
            self.train_sigs.append(operation_signature(p["answer"]))

    def predict(self, question: str) -> Optional[float]:
        if not self.train_fps:
            return None
        fp = fingerprint(question)
        sims = sorted(((_cosine(fp, tf), ts) for tf, ts in zip(self.train_fps, self.train_sigs)),
                      key=lambda x: -x[0])[:self.k]
        # vote majoritaire de la signature (parmi les + similaires)
        sigs = [s for _, s in sims if s]
        if not sigs:
            return None
        sig = Counter(sigs).most_common(1)[0][0]
        nums = extract_numbers(question)
        return apply_signature(sig, nums)


def run_supervised_gsm8k(n_test: int = 200, k: int = 7, max_train: int = 2000) -> dict:
    """Évalue le solveur supervisé sur le test set GSM8K officiel."""
    solver = GSM8KSupervisedSolver(k=k, max_train=max_train)
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        pred = solver.predict(p["question"])
        if pred is None:
            continue
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
    return {
        "dataset": "GSM8K officiel (solveur SUPERVISÉ k-NN sur train)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "k": k, "n_train_used": len(solver.train_fps),
        "note": "k-NN supervisé sur 7K train (paradigme peu-d-exemples). vs rule-based 3%.",
    }


if __name__ == "__main__":
    rep = run_supervised_gsm8k(n_test=200, k=7)
    print(f"[supervised] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(couverture {rep['coverage']*100:.0f}%, k={rep['k']}, train={rep['n_train_used']})")
