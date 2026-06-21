"""Solveur GSM8K NEURAL (NL→signature d'opérations) — entraîné sur 7K traces.

Paradigme utilisateur : 'pas besoin de milliard d'exemples'. On entraîne un VRAI réseau
neural sur le train set GSM8K (7K problèmes + CoT annoté) :
- Encode la question (embeddings + GRU) → vecteur.
- Têtes : prédit le #d'étapes + chaque opération (M/D/A/S) de la signature.
- Sur le test : prédit la signature → l'applique aux nombres extraits.

C'est du NL→raisonnement SUPERVISÉ neural (seq-ish), peu d'exemples (7K). Mesure réelle
de ce qu'un petit neural apprend sur GSM8K — vs rule-based 3% et k-NN 1.5%.
"""
from __future__ import annotations
import json
import os
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .gsm8k_bench import load_gsm8k, extract_numbers, extract_answer
from .gsm8k_supervised import operation_signature, apply_signature, TRAIN_PATH, HERE

OPS = ["M", "D", "A", "S"]
OP_TO_IDX = {o: i for i, o in enumerate(OPS)}
MAX_STEPS = 4


def _build_vocab(questions: List[str], min_freq: int = 3) -> dict:
    from collections import Counter
    c = Counter()
    for q in questions:
        for w in re.findall(r"[a-z]+", q.lower()):
            c[w] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, n in c.most_common():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def _encode_question(q: str, vocab: dict, max_len: int = 60) -> List[int]:
    ids = [vocab.get(w, 1) for w in re.findall(r"[a-z]+", q.lower())][:max_len]
    return ids + [0] * (max_len - len(ids))


def _signature_to_target(sig: str) -> Tuple[int, List[int]]:
    """→ (n_steps, [op_idx per step, pad])."""
    steps = min(len(sig), MAX_STEPS)
    ops = [OP_TO_IDX[s] for s in sig[:MAX_STEPS]]
    ops += [0] * (MAX_STEPS - len(ops))
    return steps, ops


class NLCoTModel(nn.Module):
    """Question → (n_steps, op-signature). Neural NL→raisonnement."""

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.step_head = nn.Linear(hidden, MAX_STEPS + 1)        # 0..MAX_STEPS
        self.op_head = nn.Linear(hidden, MAX_STEPS * 4)          # 4 ops × MAX_STEPS

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)                  # h: (1, B, hidden)
        h = h.squeeze(0)
        steps_logits = self.step_head(h)
        op_logits = self.op_head(h).view(-1, MAX_STEPS, 4)
        return steps_logits, op_logits


def _load_train(max_train: int = 4000):
    probs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data", "gsm8k_train.jsonl"))][:max_train]
    return probs


def train_nlcot(n_train: int = 4000, n_steps_train: int = 800, lr: float = 3e-3,
                seed: int = 0) -> tuple:
    """Entraîne le NL→CoT sur le train set GSM8K. Retourne (model, vocab)."""
    torch.manual_seed(seed)
    probs = _load_train(n_train)
    vocab = _build_vocab([p["question"] for p in probs])
    model = NLCoTModel(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor([_encode_question(p["question"], vocab) for p in probs])
    step_targets = torch.tensor([_signature_to_target(operation_signature(p["answer"]))[0]
                                 for p in probs])
    op_targets = torch.tensor([_signature_to_target(operation_signature(p["answer"]))[1]
                               for p in probs])
    n = len(probs)
    for step in range(n_steps_train):
        idx = torch.randint(0, n, (min(64, n),))
        sl, ol = model(X[idx])
        loss = nn.functional.cross_entropy(sl, step_targets[idx]) + \
               nn.functional.cross_entropy(ol.view(-1, 4), op_targets[idx].view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model, vocab


@torch.no_grad()
def predict_signature(model: NLCoTModel, vocab: dict, question: str) -> str:
    x = torch.tensor([_encode_question(question, vocab)])
    sl, ol = model(x)
    n_steps = int(sl.argmax(dim=-1)[0])
    n_steps = max(1, min(n_steps, MAX_STEPS))
    ops = ol.argmax(dim=-1)[0].tolist()      # MAX_STEPS ops
    sig = "".join(OPS[o] for o in ops[:n_steps])
    return sig


def run_neural_gsm8k(n_test: int = 200, n_train: int = 4000, n_steps_train: int = 800
                     ) -> dict:
    """Évalue le neural NL→CoT sur le test set GSM8K officiel."""
    model, vocab = train_nlcot(n_train, n_steps_train)
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        sig = predict_signature(model, vocab, p["question"])
        if not sig:
            continue
        nums = extract_numbers(p["question"])
        pred = apply_signature(sig, nums)
        if pred is None:
            continue
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
    return {
        "dataset": "GSM8K officiel (NEURAL NL→CoT entraîné sur train)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "n_train": n_train, "n_train_steps": n_steps_train,
        "vs_rule_based": "3.0%", "vs_knn": "1.5%",
    }


if __name__ == "__main__":
    rep = run_neural_gsm8k(n_test=200, n_train=4000, n_steps_train=800)
    print(f"[neural] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(couv {rep['coverage']*100:.0f}%, train={rep['n_train']}, steps={rep['n_train_steps']})")
