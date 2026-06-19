#!/usr/bin/env python3
"""
EXPÉRIENCE CROWN-JEWEL OCM-26400 — Généralisation compositionnelle.

Démontre la thèse spec (Besoins.md): à paramètres constants, la DÉCOMPOSITION
(calculer l'intermédiaire m=(a o b) puis r=(m o c)) généralise bien mieux que le
one-shot (prédire (a,b,c)->r directement) sur des compositions JAMAIS VUES.

Tâche: op(a,b) = (3a + 5b) mod 11, NON-commutative + NON-associative.
       r = op(op(a,b), c).
"""
import json, random, time
import torch
from ocm26400.amv import D_MODEL, AMVVector
from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import ReasonerBlock, encode_input, lsra_solve
from ocm26400.acsp import l_align

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


def align_to(v_vec, dictionary, target_idx):
    ent = v_vec[0:64]
    d = dictionary.canonical(target_idx).to(v_vec)
    cos = (ent @ d) / (ent.norm() * d.norm() + 1e-8)
    return 1.0 - cos


def train_binary_block(dictionary, verifier, n_steps=1500, lr=3e-3, batch=64):
    """Entraîne un bloc sur l'opération BINAIRE op(a,b)->m."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = [(a, b) for a in range(P_MOD) for b in range(P_MOD)]
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], dictionary) for i in idx]).to(DEVICE)
        out = blk(batch_in)
        loss = sum(align_to(out[j], dictionary, verifier.compose(*pairs[i]))
                   for j, i in enumerate(idx)) / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def eval_binary(blk, dictionary, verifier, pairs):
    blk.eval()
    correct = 0
    with torch.no_grad():
        for a, b in pairs:
            x = encode_input(a, b, dictionary).unsqueeze(0).to(DEVICE)
            out = blk(x)[0]
            m_pred, _ = dictionary.decode(AMVVector(out).ent)
            correct += (m_pred == verifier.compose(a, b))
    return correct / len(pairs)


def train_oneshot(dictionary, verifier, triples, n_steps=1500, lr=3e-3, batch=64):
    """One-shot: input (a,b,c) -> output.ent = op(op(a,b),c). Mémorise les triples."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        idx = torch.randint(0, len(triples), (batch,))
        batch_in = []
        targets = []
        for i in idx:
            a, b, c = triples[i]
            v = torch.zeros(D_MODEL)
            v[0:64] = dictionary.canonical(a)
            v[64:128] = dictionary.canonical(b)
            v[128:192] = dictionary.canonical(c)   # op slot = 3e operande
            batch_in.append(v)
            targets.append(verifier.compose(verifier.compose(a, b), c))
        batch_in = torch.stack(batch_in).to(DEVICE)
        out = blk(batch_in)
        loss = sum(align_to(out[j], dictionary, targets[j]) for j in range(batch)) / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def eval_oneshot(blk, dictionary, verifier, triples):
    blk.eval()
    correct = 0
    with torch.no_grad():
        for a, b, c in triples:
            v = torch.zeros(D_MODEL)
            v[0:64] = dictionary.canonical(a); v[64:128] = dictionary.canonical(b)
            v[128:192] = dictionary.canonical(c)
            out = blk(v.unsqueeze(0).to(DEVICE))[0]
            r_pred, _ = dictionary.decode(AMVVector(out).ent)
            r_true = verifier.compose(verifier.compose(a, b), c)
            correct += (r_pred == r_true)
    return correct / len(triples)


def main():
    random.seed(SEED)
    d = SymbolicDict()
    ver = Verifier(d)
    print(f"OCM-26400 crown-jewel | device={DEVICE} | Z_{P_MOD}, op(a,b)=(3a+5b)%{P_MOD}")
    print(f"params ReasonerBlock: {sum(p.numel() for p in ReasonerBlock().parameters()):,}\n")

    # tous les triples
    all_triples = [(a, b, c) for a in range(P_MOD) for b in range(P_MOD) for c in range(P_MOD)]
    random.shuffle(all_triples)
    n_train = 200
    train_tr, test_tr = all_triples[:n_train], all_triples[n_train:]
    print(f"train triples={len(train_tr)} | test triples (JAMAIS VUS)={len(test_tr)}")

    t0 = time.time()
    # ── DÉCOMPOSITION (LSRA): bloc binaire grokké, puis chaîné ──
    blk_bin = train_binary_block(d, ver, n_steps=1500)
    # acc binaire (sanity)
    bin_pairs = [(a, b) for a in range(P_MOD) for b in range(P_MOD)]
    bin_acc = eval_binary(blk_bin, d, ver, bin_pairs)
    # crown jewel: chaînage sur triples non vus
    decomp_correct = 0
    for a, b, c in test_tr:
        r_pred, n_steps, ok = lsra_solve(blk_bin, d, ver, a, b, c)
        r_true = ver.compose(ver.compose(a, b), c)
        decomp_correct += (r_pred == r_true)
    decomp_acc = decomp_correct / len(test_tr)

    # ── ONE-SHOT: mémorise des triples ──
    blk_os = train_oneshot(d, ver, train_tr, n_steps=1500)
    os_train_acc = eval_oneshot(blk_os, d, ver, train_tr[:100])
    os_test_acc = eval_oneshot(blk_os, d, ver, test_tr)

    dt = time.time() - t0
    print(f"\n--- RÉSULTATS ({dt:.1f}s) ---")
    print(f"Bloc binaire grokké (sanity op(a,b)):  {bin_acc*100:5.1f}%")
    print(f"ONE-SHOT  train acc:                    {os_train_acc*100:5.1f}%")
    print(f"ONE-SHOT  test (triples jamais vus):    {os_test_acc*100:5.1f}%")
    print(f"DÉCOMP LSRA test (triples jamais vus):  {decomp_acc*100:5.1f}%  ← crown jewel")
    gap = decomp_acc - os_test_acc
    print(f"\nÉCART généralisation (decomp - oneshot): {gap*100:+.1f} points")

    verdict = "VALIDÉ" if decomp_acc > os_test_acc + 0.2 else "NON VALIDÉ"
    print(f"VERDICT spec (decomp >> oneshot): {verdict}")

    results = {
        "device": DEVICE, "P_MOD": P_MOD, "op": "3a+5b mod P",
        "train_triples": len(train_tr), "test_triples": len(test_tr),
        "binary_grok_acc": bin_acc,
        "oneshot_train_acc": os_train_acc, "oneshot_test_acc": os_test_acc,
        "decomposition_test_acc": decomp_acc,
        "gap_points": gap, "verdict": verdict,
        "duration_s": round(dt, 1),
    }
    with open("ocm26400/crown_jewel_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats sauvés: ocm26400/crown_jewel_results.json")
    return results


if __name__ == "__main__":
    main()
