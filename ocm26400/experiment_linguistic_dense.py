#!/usr/bin/env python3
"""
EXPÉRIENCE CROWN-JEWEL SURVIVAL — passage one-hot → dense (OCM-26400, P2).

Question scientifique honnête (verdict panel d'experts) : le crown-jewel
compositionnel (decomp >> oneshot à params constants, +100pt sur la tâche
linguistique one-hot) SURVIT-IL quand on remplace le dictionnaire one-hot par
un dictionnaire DENSE (LearnedVocab) ?

On garde TOUT identique à experiment_linguistic.py :
  - même morphologie (8 stems × 2 pref × 2 suff = 60 primitives)
  - même split (SEED=0, 20 train / 12 test JAMAIS VUS)
  - même ReasonerBlock (263K params), même loss cosinus, même n_steps=1500
On ne change QUE le dictionnaire : SymbolicDict(one-hot) → LearnedVocab(dense).

Deux init testés (ablation exigée par le DA — on ne sait pas a priori lequel tient) :
  - 'ortho'   : codebook orthonormal gelé (cos inter-paires = 0). Même géométrie
                pairwise que le one-hot, simplement non aligné sur les axes.
                L'analogue dense le plus fidèle ⇒ isole la dépendance à l'alignement.
  - 'random'  : vecteurs aléatoires quasi-orthogonaux (cos inter ≈ 0.125).
                Moins séparé ⇒ teste la dépendance à la séparabilité.

Honnêteté sur la gate : decode dense retourne (idx, valid) par plus proche voisin
cosinus + marge de pureté. On compte une prédiction correcte ssi (idx==cible ET
valid). On reporte aussi l'accuracy brute (argmax seul) pour comparabilité avec
le one-hot (dont la validité est ~toujours vraie pour un block entraîné).
"""
import json, random, time
import torch

from ocm26400.amv import D_MODEL
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.reasoner import ReasonerBlock, encode_input, DEVICE
from ocm26400.experiment_linguistic import (
    STEMS, PREFIXES, SUFFIXES, N_STEM, N_PRE, N_SUF,
    ID_PREFIX, ID_SUFFIX, ID_PRE_FORM, ID_DERIVED, N_TOTAL,
    compose, surface,
)

SEED = 0


def train_block(d, ver, pairs, targets, n_steps=1500, lr=3e-3, batch=64):
    """Entraîne un ReasonerBlock : (canonical(a),canonical(b)) -> canonical(compose(a,b))."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], d) for i in idx]).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for j, i in enumerate(idx):
            ent = out[j][0:64]
            dc = d.canonical(targets[i]).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def _pairs_decomp():
    """Paires d'entraînement pour la décomposition (stem,prefix) et (préfixé,suffix)."""
    pairs, targets = [], []
    for s in range(N_STEM):
        for pi in range(N_PRE):
            a, b = s, ID_PREFIX + pi
            pairs.append((a, b)); targets.append(compose(a, b))
    for pi in range(N_PRE):
        for s in range(N_STEM):
            pre = ID_PRE_FORM + pi * N_STEM + s
            for si in range(N_SUF):
                a, b = pre, ID_SUFFIX + si
                pairs.append((a, b)); targets.append(compose(a, b))
    return pairs, targets


def _pairs_oneshot(triples):
    pairs, targets = [], []
    for s, pi, si in triples:
        pairs.append((s, ID_PREFIX + pi, ID_SUFFIX + si))   # 3-slot
        targets.append(ID_DERIVED + si * (N_STEM * N_PRE) + pi * N_STEM + s)
    return pairs, targets


def decode_step(blk, d, a, b):
    """1 forward + decode dense. Retourne (idx, valid)."""
    blk.eval()
    dev = next(blk.parameters()).device
    with torch.no_grad():
        x = encode_input(a, b, d).unsqueeze(0).to(dev)
        out = blk(x)[0]
        idx, valid = d.decode(out[0:64])
    return idx, valid


def eval_decomp(blk, d, triples):
    """Chaîne 2 étapes compose. Accuracy exacte (gated: idx==cible ET valid)."""
    blk.eval()
    correct_gated = 0
    correct_raw = 0
    n_valid = 0
    for s, pi, si in triples:
        idx1, v1 = decode_step(blk, d, s, ID_PREFIX + pi)
        if not (ID_PRE_FORM <= idx1 < ID_DERIVED):
            continue  # étape 1 ratée
        idx2, v2 = decode_step(blk, d, idx1, ID_SUFFIX + si)
        tgt = ID_DERIVED + si * (N_STEM * N_PRE) + (idx1 - ID_PRE_FORM)
        ok = (idx2 == tgt)
        correct_raw += ok
        n_valid += (ok and v1 and v2)
        correct_gated += (ok and v1 and v2)
    n = len(triples)
    return correct_gated / n, correct_raw / n, n_valid / n


def eval_oneshot(blk, d, triples):
    blk.eval()
    dev = next(blk.parameters()).device
    correct_gated = 0
    correct_raw = 0
    n_valid = 0
    for s, pi, si in triples:
        v = torch.zeros(D_MODEL)
        v[0:64] = d.canonical(s)
        v[64:128] = d.canonical(ID_PREFIX + pi)
        v[128:192] = d.canonical(ID_SUFFIX + si)
        with torch.no_grad():
            out = blk(v.unsqueeze(0).to(dev))[0]
            idx, valid = d.decode(out[0:64])
        tgt = ID_DERIVED + si * (N_STEM * N_PRE) + pi * N_STEM + s
        ok = (idx == tgt)
        correct_raw += ok
        n_valid += (ok and valid)
        correct_gated += (ok and valid)
    n = len(triples)
    return correct_gated / n, correct_raw / n, n_valid / n


def run_one(init_mode):
    """Lance le crown-jewel complet pour une init de codebook donnée. Renvoie le dict résultat."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    d = LearnedVocab(n=N_TOTAL, init=init_mode, seed=SEED).freeze()   # codebook dense GELÉ
    cos_inter = d.mean_inter_pair_cos()

    all_tr = [(s, pi, si) for s in range(N_STEM) for pi in range(N_PRE) for si in range(N_SUF)]
    random.shuffle(all_tr)
    train_tr, test_tr = all_tr[:20], all_tr[20:]

    t0 = time.time()
    pairs_d, tgt_d = _pairs_decomp()
    blk_d = train_block(d, None, pairs_d, tgt_d, n_steps=1500)
    dep_test_g, dep_test_r, dep_v = eval_decomp(blk_d, d, test_tr)

    pairs_o, tgt_o = _pairs_oneshot(train_tr)
    # one-shot : 3-slot, on entraîne un block spécifique
    torch.manual_seed(SEED)
    blk_o = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk_o.parameters(), lr=3e-3)
    for _ in range(1500):
        bidx = torch.randint(0, len(pairs_o), (64,))
        batch_in = []
        for i in bidx:
            s, bp, bs = pairs_o[i]
            v = torch.zeros(D_MODEL)
            v[0:64] = d.canonical(s); v[64:128] = d.canonical(bp); v[128:192] = d.canonical(bs)
            batch_in.append(v)
        batch_in = torch.stack(batch_in).to(DEVICE)
        out = blk_o(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for j, i in enumerate(bidx):
            ent = out[j][0:64]
            dc = d.canonical(tgt_o[i]).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / 64
        opt.zero_grad(); loss.backward(); opt.step()
    os_test_g, os_test_r, os_v = eval_oneshot(blk_o, d, test_tr)
    dt = time.time() - t0

    gap_g = dep_test_g - os_test_g
    gap_r = dep_test_r - os_test_r
    verdict = "VALIDÉ" if dep_test_g > os_test_g + 0.15 else "NON VALIDÉ"

    print(f"\n--- DENSE '{init_mode}' | cos_inter={cos_inter:.3f} | {dt:.1f}s ---")
    print(f"ONE-SHOT test  (gated / raw): {os_test_g*100:5.1f}% / {os_test_r*100:5.1f}%  (valid {os_v*100:.0f}%)")
    print(f"DÉCOMP   test  (gated / raw): {dep_test_g*100:5.1f}% / {dep_test_r*100:5.1f}%  (valid {dep_v*100:.0f}%)")
    print(f"ÉCART (gated): {gap_g*100:+.1f} points   |   VERDICT: {verdict}")

    return {
        "init": init_mode,
        "mean_inter_pair_cos": round(cos_inter, 4),
        "oneshot_test_gated": os_test_g, "oneshot_test_raw": os_test_r, "oneshot_valid_rate": os_v,
        "decomp_test_gated": dep_test_g, "decomp_test_raw": dep_test_r, "decomp_valid_rate": dep_v,
        "gap_points_gated": gap_g, "gap_points_raw": gap_r,
        "verdict": verdict, "duration_s": round(dt, 1),
    }


def main():
    print(f"OCM-26400 CROWN-JEWEL SURVIVAL (one-hot → dense) | device={DEVICE}")
    print(f"morphologie: {N_STEM} stems × {N_PRE} pref × {N_SUF} suff = {N_TOTAL} primitives")
    print(f"Référence one-hot (experiment_linguistic.py): decomp=100% / oneshot=0% / gap=+100pt")
    results = [run_one("ortho"), run_one("random")]
    all_results = {
        "task": "crown-jewel survival: one-hot → dense (english derivational morphology)",
        "n_primitives": N_TOTAL,
        "reference_onehot": {"decomp_test": 1.0, "oneshot_test": 0.0, "gap_points": 1.0},
        "dense_variants": results,
    }
    with open("ocm26400/linguistic_dense_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nRésultats: ocm26400/linguistic_dense_results.json")
    return all_results


if __name__ == "__main__":
    main()
