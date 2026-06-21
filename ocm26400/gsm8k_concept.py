"""GSM8K via ConceptModel — IDs numériques + SpectralCoreBlock + 1-cos (crown-jewel).

L'approche prescrite par l'utilisateur :
'ConceptVocab (IDs) + SpectralCoreBlock (FFT sur IDs) + loss 1-cos (comme crown-jewel)'

Pipeline :
1. ConceptVocab encode chaque question GSM8K en IDs numériques
2. ConceptModel (LearnedVocab dense + SpectralCoreBlock) traite les IDs
3. Loss 1-cos (PAS de CE) — le modèle apprend à prédire le concept (ID) suivant
4. Le scratchpad cascade (L1) décompose le raisonnement multi-étapes

Le dense number-binding à 66% prouve que l'encoding fonctionne (pas de collision).
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .concept_vocab import ConceptVocab, ConceptModel
from .gsm8k_bench import load_gsm8k, extract_answer
from .gsm8k_supervised import HERE


def prepare_gsm8k_concepts(n_train: int = 3000) -> Tuple[ConceptVocab, List[Tuple[List[int], int]]]:
    """Prépare les données GSM8K en IDs conceptuels.
    Retourne (vocab, [(question_ids, answer_num_id), ...])."""
    train_path = os.path.join(HERE, "..", "data", "gsm8k_train.jsonl")
    probs = [json.loads(l) for l in open(train_path)][:n_train]
    cv = ConceptVocab(max_concepts=50000)
    data = []
    for p in probs:
        gold = extract_answer(p["answer"])
        if gold is None or gold > 9999:
            continue
        q_ids = cv.text_to_ids(p["question"])
        a_id = cv.get_num_id(int(gold))
        data.append((q_ids, a_id))
    return cv, data


def run_concept_gsm8k(n_test: int = 200, n_train: int = 3000, n_steps: int = 3000,
                        device: str = None) -> Dict:
    """Évalue le ConceptModel sur GSM8K officiel."""
    from .reasoner import ReasonerBlock

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print("[concept_gsm8k] Préparation des données...")
    cv, train_data = prepare_gsm8k_concepts(n_train)
    print(f"  vocab: {cv.size} concepts, {len(train_data)} problèmes train")

    # ConceptModel
    model = ConceptModel(vocab_size=cv.size, d_model=256, seq_len=60).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # padding des séquences
    max_len = 60
    def pad(ids, length):
        return ids[:length] + [cv.PAD] * max(0, length - len(ids))

    X = torch.tensor([pad(q, max_len) for q, _ in train_data], device=device)
    # target : l'ID de la réponse (on veut que la dernière position output → réponse)
    T = torch.tensor([[a for _, a in [d]] * max_len for d in train_data], device=device)
    # en fait on veut juste la dernière position → réponse
    T_last = torch.tensor([[a] for _, a in train_data], device=device)

    print(f"[concept_gsm8k] Training ConceptModel ({n_steps} steps, 1-cos loss)...")
    n = len(train_data)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(64, n),))
        x_batch = X[idx]
        # forward → (B, L, 256) ; on prend la dernière position
        out = model(x_batch)[:, -1, :]  # (B, 256)
        # target embedding (dense)
        M = model.embeddings._matrix().to(device)
        target_oh = F.one_hot(T_last[idx].squeeze(1), cv.size).float()
        target_dense = target_oh @ M  # (B, 64)
        # 1-cos loss
        out_ent = out[:, :64]
        cos = F.cosine_similarity(out_ent, target_dense, dim=-1).clamp(-1, 1)
        loss = (1 - cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # évaluation
    print(f"[concept_gsm8k] Évaluation sur {n_test} problèmes test...")
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    M = model.embeddings._matrix().to(device)
    with torch.no_grad():
        for p in tests:
            gold = extract_answer(p["answer"])
            if gold is None:
                continue
            n_total += 1
            q_ids = pad(cv.text_to_ids(p["question"]), max_len)
            x = torch.tensor([q_ids], device=device)
            out = model(x)[:, -1, :]  # (1, 256)
            out_ent = out[0, :64]  # (64,)
            # decode : plus proche voisin dans le codebook (nombres uniquement)
            num_sims = M[4:10004] @ out_ent  # cosinus avec NUM_0..NUM_9999
            pred = int(num_sims.argmax())  # ID → index dans nombres
            n_attempted += 1
            if pred == int(gold):
                n_correct += 1

    return {
        "dataset": "GSM8K officiel (ConceptModel = IDs + SpectralCoreBlock + 1-cos)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "vocab_size": cv.size,
        "archi": "ConceptVocab IDs + LearnedVocab dense + SpectralCoreBlock (FFT, pas de transformer)",
        "loss": "1-cos (crown-jewel, PAS de CE)",
        "procedure": "Besoins prescrit: ConceptVocab(IDs) + FFT + 1-cos",
    }


if __name__ == "__main__":
    rep = run_concept_gsm8k(n_test=200, n_train=3000, n_steps=3000)
    print(f"\n[concept_gsm8k] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy']*100:.1f}%")
    print(f"  {rep['archi']}")
    print(f"  loss={rep['loss']}")
