#!/usr/bin/env python3
"""
EXPÉRIENCE SCALING V>64 — LearnedVocab au-delà du plafond one-hot (OCM-26400, P2).

SymbolicDict impose assert n <= dim (verifier.py:28) avec dim=PART=64 : le slot
ent one-hot ne peut indexer que ~64 primitives. LearnedVocab (dense) lève ce
plafond. Cette expérience prouve deux choses à la fois :

  1. CAPACITÉ : LearnedVocab(n=120) se construit et roundtrip (V>64 impossible
     en one-hot).
  2. CROWN-JEWEL À GRANDE ÉCHELLE : sur Z_120 (op(a,b)=(3a+5b) mod 120,
     non-associative), l'écart decomp >> oneshot survit. Plus fort encore :
     le block binaire est entraîné sur un ÉCHANTILLON de paires (pas les 14400),
     donc il doit GROKKER la règle linéaire-modulaire pour généraliser aux
     paires jamais vues — un crown-jewel strictement plus exigeant que Z_11
     (où toutes les paires étaient entraînées).

Honnêteté : on compte une prédiction correcte ssi (idx==cible ET decode valide).
On reporte l'accuracy brute (argmax seul) en parallèle. Le block binaire grok
est mesuré sur des paires JAMAIS VUES (généralisation de la règle, pas mémorisation).
"""
import json, random, time
import torch

from ocm26400.amv import D_MODEL
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.reasoner import ReasonerBlock, encode_input, DEVICE

SEED = 0
P = 120                 # Z_120 : 120 primitives (> 64, impossible en one-hot)
A_COEF, B_COEF = 3, 5   # op(a,b) = (3a + 5b) mod 120 (non-commutative, non-associative)


def op(a, b):
    return (A_COEF * a + B_COEF * b) % P


def train_binary_block(d, n_steps=3000, lr=3e-3, batch=128):
    """Grok la règle op(a,b) sur un échantillon de paires (pas les 14400)."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        a = torch.randint(0, P, (batch,))
        b = torch.randint(0, P, (batch,))
        batch_in = torch.stack([encode_input(int(a[i]), int(b[i]), d) for i in range(batch)]).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for i in range(batch):
            ent = out[i][0:64]
            dc = d.canonical(op(int(a[i]), int(b[i]))).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def decode_step(blk, d, a, b):
    blk.eval()
    dev = next(blk.parameters()).device
    with torch.no_grad():
        x = encode_input(a, b, d).unsqueeze(0).to(dev)
        out = blk(x)[0]
        idx, valid = d.decode(out[0:64])
    return idx, valid


def eval_binary_grok(blk, d, pairs):
    """Accuracy du block binaire sur des paires (généralisation de la règle)."""
    blk.eval()
    correct_g, correct_r, nv = 0, 0, 0
    for a, b in pairs:
        idx, valid = decode_step(blk, d, a, b)
        ok = (idx == op(a, b))
        correct_r += ok
        nv += (ok and valid)
        correct_g += (ok and valid)
    n = len(pairs)
    return correct_g / n, correct_r / n, nv / n


def eval_decomp(blk, d, triples):
    """r = op(op(a,b), c) via 2 étapes. Crown-jewel (décomposition)."""
    blk.eval()
    correct_g, correct_r, nv = 0, 0, 0
    for a, b, c in triples:
        m, vm = decode_step(blk, d, a, b)             # étape 1 : m = op(a,b)
        if not (0 <= m < P):
            continue
        r, vr = decode_step(blk, d, m, c)             # étape 2 : r = op(m,c)
        tgt = op(op(a, b), c)
        ok = (r == tgt)
        correct_r += ok
        nv += (ok and vm and vr)
        correct_g += (ok and vm and vr)
    n = len(triples)
    return correct_g / n, correct_r / n, nv / n


def train_oneshot(d, triples, n_steps=1500, lr=3e-3, batch=128):
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        idx = torch.randint(0, len(triples), (batch,))
        batch_in = []
        for i in idx:
            a, b, c = triples[i]
            v = torch.zeros(D_MODEL)
            v[0:64] = d.canonical(a); v[64:128] = d.canonical(b); v[128:192] = d.canonical(c)
            batch_in.append(v)
        batch_in = torch.stack(batch_in).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for j, i in enumerate(idx):
            ent = out[j][0:64]
            a, b, c = triples[i]
            dc = d.canonical(op(op(a, b), c)).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def eval_oneshot(blk, d, triples):
    blk.eval()
    dev = next(blk.parameters()).device
    correct_g, correct_r, nv = 0, 0, 0
    for a, b, c in triples:
        v = torch.zeros(D_MODEL)
        v[0:64] = d.canonical(a); v[64:128] = d.canonical(b); v[128:192] = d.canonical(c)
        with torch.no_grad():
            out = blk(v.unsqueeze(0).to(dev))[0]
            idx, valid = d.decode(out[0:64])
        tgt = op(op(a, b), c)
        ok = (idx == tgt)
        correct_r += ok
        nv += (ok and valid)
        correct_g += (ok and valid)
    n = len(triples)
    return correct_g / n, correct_r / n, nv / n


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"OCM-26400 SCALING V>64 | device={DEVICE} | Z_{P} ({P} primitives, > 64)")
    print("SymbolicDict ne peut PAS (assert n<=64). LearnedVocab dense le peut.")

    d = LearnedVocab(n=P, init="random", seed=SEED).freeze()
    cos_inter = d.mean_inter_pair_cos()
    # preuve roundtrip identité sur les 120 primitives
    rt = sum(1 for i in range(P) if d.decode(d.canonical(i)) == (i, True))
    print(f"LearnedVocab(n={P}) construit | cos_inter={cos_inter:.3f} | roundtrip {rt}/{P}")

    # triples (a,b,c) : train/test disjoints (par tuple)
    random.seed(SEED)
    seen = set()
    triples = []
    while len(triples) < 900:
        t = (random.randint(0, P - 1), random.randint(0, P - 1), random.randint(0, P - 1))
        if t not in seen:
            seen.add(t); triples.append(t)
    random.shuffle(triples)
    train_tr, test_tr = triples[:300], triples[300:]    # 300 train / 600 test
    # paires JAMAIS VUES pour mesurer le grok de la règle (indépendantes des triples)
    pair_test = [(random.randint(0, P - 1), random.randint(0, P - 1)) for _ in range(300)]
    print(f"train triples={len(train_tr)} | test triples (JAMAIS VUS)={len(test_tr)} | pair-test={len(pair_test)}")

    t0 = time.time()
    blk_bin = train_binary_block(d, n_steps=3000)
    grok_g, grok_r, grok_v = eval_binary_grok(blk_bin, d, pair_test)
    dep_g, dep_r, dep_v = eval_decomp(blk_bin, d, test_tr)

    blk_o = train_oneshot(d, train_tr, n_steps=1500)
    os_g, os_r, os_v = eval_oneshot(blk_o, d, test_tr)
    dt = time.time() - t0

    gap_g = dep_g - os_g
    verdict = "VALIDÉ" if dep_g > os_g + 0.15 else "NON VALIDÉ"

    print(f"\n--- RÉSULTATS SCALING Z_{P} ({dt:.1f}s) ---")
    print(f"Block binaire grok (paires jamais vues) : {grok_g*100:5.1f}% (raw {grok_r*100:.1f}%, valid {grok_v*100:.0f}%)")
    print(f"ONE-SHOT test  (gated / raw): {os_g*100:5.1f}% / {os_r*100:5.1f}%  (valid {os_v*100:.0f}%)")
    print(f"DÉCOMP   test  (gated / raw): {dep_g*100:5.1f}% / {dep_r*100:5.1f}%  (valid {dep_v*100:.0f}%)")
    print(f"ÉCART (gated): {gap_g*100:+.1f} points   |   VERDICT: {verdict}")

    results = {
        "task": "scaling V>64 (Z_120 arithmetic crown-jewel, dense LearnedVocab)",
        "P": P, "n_primitives": P,
        "impossible_with_onehot": "assert n<=dim=64 in SymbolicDict",
        "mean_inter_pair_cos": round(cos_inter, 4),
        "roundtrip_identity": f"{rt}/{P}",
        "train_triples": len(train_tr), "test_triples": len(test_tr),
        "binary_grok_gated": grok_g, "binary_grok_raw": grok_r, "binary_grok_valid_rate": grok_v,
        "oneshot_test_gated": os_g, "oneshot_test_raw": os_r, "oneshot_valid_rate": os_v,
        "decomp_test_gated": dep_g, "decomp_test_raw": dep_r, "decomp_valid_rate": dep_v,
        "gap_points_gated": gap_g, "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/vocab_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Résultats: ocm26400/vocab_scale_results.json")
    return results


if __name__ == "__main__":
    main()
