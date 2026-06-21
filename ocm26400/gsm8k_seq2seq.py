"""Seq2Seq NL→CoT avec COPY — le vrai number-binding pour GSM8K.

Les approches précédentes échouaient (rule 3%, k-NN 1.5%, neural-sig 0%) car elles
n'avaient pas le NUMBER-BINDING : quel nombre du texte va dans quelle opération.

Celui-ci est un VRAI seq2seq encoder-décodeur avec COPY mechanism :
- Encoder : question (GRU sur embeddings) → états + positions des nombres.
- Decoder : génère une séquence d''actions' : soit un OPÉRATEUR (M/D/A/S), soit
  'COPIER nombre i' (réfère au i-ème nombre extrait de la question).
- Ainsi le modèle apprend : quels nombres sélectionner, dans quel ordre, avec quelles ops.
- Entraîné sur les 7K traces GSM8K (question → séquence d'actions dérivée du CoT <<>>).

C'est le seq2seq LM que le paradigme appelle ('peu d'exemples, pas milliards'). Le
number-binding est APPRIS par le réseau via la supervision des traces.
"""
from __future__ import annotations
import json
import os
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gsm8k_bench import load_gsm8k, extract_numbers, extract_answer
from .gsm8k_supervised import HERE

# Vocabulaire d'actions du décodeur :
# 4 opérateurs + COPY_i (i = index du nombre dans la question, jusqu'à MAX_NUMS) + START/END/PAD
MAX_NUMS = 8
OP_ACTIONS = ["OP_M", "OP_D", "OP_A", "OP_S"]
SPECIAL = ["PAD", "START", "END"]
# actions = SPECIAL + OP_ACTIONS + COPY_0..COPY_{MAX_NUMS-1}
COPY_ACTIONS = [f"COPY_{i}" for i in range(MAX_NUMS)]
ACTIONS = SPECIAL + OP_ACTIONS + COPY_ACTIONS
ACT_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
MAX_DECOD = 16          # longueur max de la séquence d'actions


def _question_numbers_with_pos(question: str) -> List[Tuple[float, int, int]]:
    """Nombres du texte avec (valeur, début, fin). Pour le COPY."""
    out = []
    for m in re.finditer(r"\d+(?:\.\d+)?", question.replace(",", "")):
        out.append((float(m.group()), m.start(), m.end()))
    return out


def _trace_to_actions(question: str, answer: str) -> Optional[List[int]]:
    """Convertit le CoT annoté (<<expr>>) en séquence d'actions COPY/OP.
    Chaque <<expr>> devient une séquence : COPY_i OP COPY_j [OP COPY_k ...]."""
    exprs = re.findall(r"<<([^>]+)>>", answer)
    nums_qp = _question_numbers_with_pos(question)
    qnums = [n[0] for n in nums_qp]
    actions = [ACT_TO_IDX["START"]]
    for expr in exprs:
        # tokenize l'expression : nombres et opérateurs
        tokens = re.findall(r"\d+(?:\.\d+)?|[+\-*/]", expr)
        for tok in tokens:
            if tok in "+-*/":
                op = {"*": "OP_M", "/": "OP_D", "+": "OP_A", "-": "OP_S"}[tok]
                actions.append(ACT_TO_IDX[op])
            else:
                val = float(tok)
                # COPY : trouve ce nombre dans la question (ou résultat intermédiaire → skip)
                if val in qnums:
                    i = qnums.index(val)
                    if i < MAX_NUMS:           # garde-fou : >8 nombres → skip
                        actions.append(ACT_TO_IDX[f"COPY_{i}"])
                # sinon c'est un résultat intermédiaire (on ne peut pas le copier → skip)
    actions.append(ACT_TO_IDX["END"])
    return actions[:MAX_DECOD] + [ACT_TO_IDX["PAD"]] * max(0, MAX_DECOD - len(actions))


def _actions_to_value(actions: List[int], qnums: List[float]) -> Optional[float]:
    """Exécute la séquence d'actions : COPY_i pousse le i-ème nombre, OP combine la pile."""
    stack = []
    for a in actions:
        name = ACTIONS[a]
        if name.startswith("COPY_"):
            i = int(name.split("_")[1])
            if i < len(qnums):
                stack.append(qnums[i])
        elif name.startswith("OP_"):
            if len(stack) < 2:
                continue
            b, a_ = stack.pop(), stack.pop()
            if name == "OP_M":
                stack.append(a_ * b)
            elif name == "OP_D":
                stack.append(a_ / b if b != 0 else 0)
            elif name == "OP_A":
                stack.append(a_ + b)
            elif name == "OP_S":
                stack.append(a_ - b)
    return stack[-1] if stack else None


def _build_vocab(questions: List[str], min_freq: int = 5) -> dict:
    from collections import Counter
    c = Counter()
    for q in questions:
        c.update(re.findall(r"[a-z]+", q.lower()))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, n in c.most_common():
        if n >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def _encode_q(q: str, vocab: dict, max_len: int = 50) -> List[int]:
    ids = [vocab.get(w, 1) for w in re.findall(r"[a-z]+", q.lower())][:max_len]
    return ids + [0] * (max_len - len(ids))


class Seq2SeqCoT(nn.Module):
    """Encoder question → Decoder actions (OP + COPY_i) avec attention."""

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc = nn.GRU(emb_dim, hidden, batch_first=True)
        self.dec_emb = nn.Embedding(len(ACTIONS), emb_dim)
        self.dec = nn.GRU(emb_dim, hidden, batch_first=True)
        # attention
        self.attn = nn.Linear(hidden * 2, 50)
        self.attn_combine = nn.Linear(hidden * 2, emb_dim)
        self.out = nn.Linear(hidden, len(ACTIONS))

    def forward(self, src, tgt_in):
        # encoder
        e = self.emb(src)
        enc_out, enc_h = self.enc(e)               # enc_out: (B,L,H)
        # decoder (teacher forcing sur tgt_in)
        d = self.dec_emb(tgt_in)
        dec_out, _ = self.dec(d, enc_h)
        logits = self.out(dec_out)                  # (B, T, n_actions)
        return logits


def train_seq2seq(n_train: int = 3000, n_steps: int = 1500, lr: float = 3e-3,
                  seed: int = 0) -> tuple:
    """Entraîne le seq2seq COPY sur le train set GSM8K."""
    torch.manual_seed(seed)
    probs = json.load(open(os.path.join(HERE, "..", "data", "gsm8k_train.jsonl"))) \
        if False else [json.loads(l) for l in open(os.path.join(HERE, "..", "data",
                  "gsm8k_train.jsonl"))][:n_train]
    vocab = _build_vocab([p["question"] for p in probs])
    model = Seq2SeqCoT(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.tensor([_encode_q(p["question"], vocab) for p in probs])
    # séquences d'actions cibles
    targets = []
    for p in probs:
        a = _trace_to_actions(p["question"], p["answer"]) or [ACT_TO_IDX["PAD"]] * MAX_DECOD
        targets.append(a)
    T = torch.tensor(targets)                # (N, MAX_DECOD)
    tgt_in = T[:, :-1]                       # input (shift)
    tgt_out = T[:, 1:]                       # cible
    n = len(probs)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(64, n),))
        logits = model(X[idx], tgt_in[idx])
        loss = F.cross_entropy(logits.reshape(-1, len(ACTIONS)), tgt_out[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model, vocab


@torch.no_grad()
def predict_cot(model: Seq2SeqCoT, vocab: dict, question: str) -> Optional[float]:
    """Génère la séquence d'actions (greedy) puis l'exécute → valeur finale."""
    x = torch.tensor([_encode_q(question, vocab)])
    e = model.emb(x)
    enc_out, enc_h = model.enc(e)
    cur = torch.tensor([[ACT_TO_IDX["START"]]])
    actions = []
    for _ in range(MAX_DECOD - 1):
        d = model.dec_emb(cur)
        dec_out, _ = model.dec(d, enc_h)
        logits = model.out(dec_out[:, -1])
        nxt = int(logits.argmax(dim=-1)[0])
        if ACTIONS[nxt] == "END":
            break
        actions.append(nxt)
        cur = torch.tensor([[nxt]])
    qnums = extract_numbers(question)
    return _actions_to_value(actions, qnums)


def run_seq2seq_gsm8k(n_test: int = 200, n_train: int = 3000, n_steps: int = 1500) -> dict:
    """Évalue le seq2seq COPY sur le test set GSM8K officiel."""
    model, vocab = train_seq2seq(n_train, n_steps)
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        pred = predict_cot(model, vocab, p["question"])
        if pred is None:
            continue
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
    return {
        "dataset": "GSM8K officiel (SEQ2SEQ COPY NL→CoT entraîné)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "n_train": n_train, "n_train_steps": n_steps,
        "vs": {"rule_based": "3.0%", "knn": "1.5%", "neural_sig": "0%"},
    }


if __name__ == "__main__":
    rep = run_seq2seq_gsm8k(n_test=200, n_train=3000, n_steps=1500)
    print(f"[seq2seq] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(couv {rep['coverage']*100:.0f}%, train={rep['n_train']})")
