"""Cascade conceptuel — ConceptVocab IDs + scratchpad cascade pour GSM8K.

La cascade marche en arithmétique (100%) car le number-binding est trivial.
En NL, chaque pas doit EXTRAIRE (num_ID, op_ID) d'une phrase, puis appliquer
l'opération à l'accumulateur — comme le crown-jewel mais sur du langage.

Chaque pas :
1. Phrase courante → ConceptVocab IDs → SpectralCoreBlock → (num_ID, op_ID)
2. acc = grok_op(acc, num) — le même mécanisme que le crown-jewel
3. L'intermédiaire (nouveau acc) est VISIBLE (scratchpad, loi L1)

La cascade se résout en chaîne → profondeur arbitraire (loi L3).
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept_vocab import ConceptVocab
from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART
from .reasoner import ReasonerBlock
from .learned_vocab import LearnedVocab


class SentenceExtractor(nn.Module):
    """Extrait (num_ID, op_ID) d'une phrase via SpectralCoreBlock (1-cos).
    Entrée : phrase IDs (ConceptVocab) → sortie : (num_pos, op_pos) dans le codebook.
    C'est l'association 1-source (L6) qui MANQUAIT — le number-binding NL."""

    def __init__(self, vocab_size: int, d_model: int = D_MODEL, seq_len: int = 30):
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = LearnedVocab(n=vocab_size, dim=PART, init="random", seed=0)
        self.embeddings.freeze()
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)
        # 2 têtes : num (dans le codebook nombres) et op (4 ops)
        self.num_head = nn.Linear(d_model, PART)  # projecte vers codebook
        self.op_head = nn.Linear(d_model, 4)      # ADD/SUB/MUL/DIV
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sent_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """sent_ids: (B, L) IDs → (num_logits (B, PART), op_logits (B, 4)).
        num_logits = cosinus avec codebook LearnedVocab (pour decode NN)."""
        B, L = sent_ids.shape
        M = self.embeddings._matrix().to(sent_ids.device)
        x_oh = F.one_hot(sent_ids, self.vocab_size).float()
        x_dense = x_oh @ M
        full = torch.zeros(B, L, D_MODEL, device=sent_ids.device)
        full[:, :, :PART] = x_dense
        out = self.core(full)
        out = self.norm(out)
        # pool (moyenne sur la séquence)
        pooled = out.mean(dim=1)  # (B, 256)
        # num : projecte vers codebook → cosinus avec chaque nombre
        num_proj = self.num_head(pooled)  # (B, 64)
        num_sims = num_proj @ M[:PART].T  # (B, PART) — pas idéal, on veut V_entier
        # op : classification 4 classes
        op_logits = self.op_head(pooled)  # (B, 4)
        return num_sims, op_logits


def train_sentence_extractor(extractor: SentenceExtractor, train_data: List[dict],
                              cv: ConceptVocab, n_steps: int = 2000,
                              lr: float = 3e-3, device: str = "cpu") -> List[float]:
    """Entraîne le SentenceExtractor sur les données GSM8K CoT.
    Chaque CoT étape = 1 exemple : (phrase_contexte, num_target, op_target).
    Loss = 1-cos(num_pred, num_target) + CE(op_pred, op_target)."""
    extractor = extractor.to(device)
    opt = torch.optim.Adam(extractor.parameters(), lr=lr)
    history = []

    # prépare les données
    examples = []
    for item in train_data:
        sent_ids = item["sent_ids"]
        num_id = item["num_id"]
        op_id = item["op_id"]
        examples.append((sent_ids, num_id, op_id))

    n = len(examples)
    M = extractor.embeddings._matrix().to(device)

    for step in range(n_steps):
        idx = torch.randint(0, n, (min(32, n),))
        total_loss = 0.0
        for i in idx:
            sent_ids, num_id, op_id = examples[i]
            x = torch.tensor([sent_ids], device=device)
            num_sims, op_logits = extractor(x)

            # num loss : 1-cos avec l'embedding du nombre cible
            target_num = M[num_id]  # (64,)
            num_loss = 1.0 - F.cosine_similarity(
                num_sims[0].unsqueeze(0), target_num[:PART].unsqueeze(0)
            ).clamp(-1, 1)

            # op loss : CE
            op_loss = F.cross_entropy(op_logits, torch.tensor([op_id], device=device))

            total_loss += num_loss + op_loss

        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()
        if step % 200 == 0:
            history.append(float(total_loss.item() / len(idx)))
    return history


def extract_training_data_from_cot(cv: ConceptVocab,
                                     n_problems: int = 3000) -> List[dict]:
    """Extrait les données d'entraînement depuis les CoT GSM8K.
    Chaque étape CoT = 1 exemple : (phrase, nombre, opération)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "gsm8k_train.jsonl")
    if not os.path.exists(path):
        return []
    probs = [json.loads(l) for l in open(path)][:n_problems]
    data = []
    for p in probs:
        question = p["question"]
        sentences = re.split(r"(?<=[.?!])\s+", question)
        exprs = re.findall(r"<<([\d.\s]+)([+\-*/])([\d.\s]+)=([\d.]+)>>", p["answer"])
        op_map = {"+": 0, "-": 1, "*": 2, "/": 3}
        for i, (a_str, op_ch, b_str, result_str) in enumerate(exprs):
            try:
                a, b = int(float(a_str)), int(float(b_str))
                # quelle phrase ? heuristique : la (i+1)ème phrase
                sent_idx = min(i + 1, len(sentences) - 1)
                sent = sentences[sent_idx] if sentences else question
                sent_ids = cv.text_to_ids(sent)[:30]  # max 30 tokens
                # le nombre extrait = b (le 2ème opérande)
                num_id = cv.get_num_id(b)
                op_id = op_map.get(op_ch, 0)
                data.append({"sent_ids": sent_ids, "num_id": num_id, "op_id": op_id})
            except (ValueError, TypeError):
                continue
    return data


def run_concept_cascade_gsm8k(n_test: int = 200, n_train: int = 3000,
                                 n_steps: int = 2000, device: str = None) -> Dict:
    """Cascade conceptuel sur GSM8K : ConceptVocab + SentenceExtractor + cascade."""
    from .gsm8k_bench import load_gsm8k, extract_answer
    from .language_primitives_grok import extract_all_numbers, CUE_TO_OP

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("[concept_cascade] Préparation ConceptVocab...")
    cv = ConceptVocab(max_concepts=50000)

    print("[concept_cascade] Extraction données CoT...")
    train_data = extract_training_data_from_cot(cv, n_train)
    print(f"  {len(train_data)} exemples d'extraction")

    print("[concept_cascade] Training SentenceExtractor...")
    extractor = SentenceExtractor(vocab_size=cv.size)
    if train_data:
        train_sentence_extractor(extractor, train_data, cv, n_steps=n_steps, device=device)

    # évaluation : cascade phrase par phrase
    print(f"[concept_cascade] Évaluation cascade sur {n_test} problèmes...")
    extractor.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    M = extractor.embeddings._matrix().to(device)

    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        # cascade : phrase par phrase
        question = p["question"]
        sentences = re.split(r"(?<=[.?!])\s+", question)
        nums = extract_all_numbers(question)
        if not nums:
            continue
        acc = nums[0]
        n_attempted += 1

        for sent in sentences[1:]:
            if sent.strip().endswith("?"):
                continue
            sent_ids = cv.text_to_ids(sent)[:30]
            if len(sent_ids) < 2:
                continue
            x = torch.tensor([sent_ids], device=device)
            with torch.no_grad():
                num_sims, op_logits = extractor(x)
                op_pred = int(op_logits[0].argmax())
            # extract number from sentence (heuristic fallback)
            sent_nums = extract_all_numbers(sent)
            if not sent_nums:
                continue
            num = sent_nums[0]
            # apply operation
            if op_pred == 0:  # ADD
                acc += num
            elif op_pred == 1:  # SUB
                acc -= num
            elif op_pred == 2:  # MUL
                acc *= num
            elif op_pred == 3:  # DIV
                acc = acc / num if num != 0 else acc

        if abs(acc - gold) < 1e-6:
            n_correct += 1

    return {
        "dataset": "GSM8K officiel (cascade conceptuel — ConceptVocab + SentenceExtractor)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "vocab_size": cv.size,
        "archi": "ConceptVocab IDs + SentenceExtractor (SpectralCoreBlock) + cascade",
        "loss": "1-cos (num) + CE (op) — crown-jewel pour le number extraction",
    }


if __name__ == "__main__":
    rep = run_concept_cascade_gsm8k(n_test=200, n_train=3000, n_steps=2000)
    print(f"\n[concept_cascade] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy']*100:.1f}%")
    print(f"  {rep['archi']}")
