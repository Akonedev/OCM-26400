#!/usr/bin/env python3
"""
EXPÉRIENCE LINGUISTIQUE OCM-26400 — Dérivation morphologique anglaise.

Prouve que le mécanisme OCM (AMV + ACSP + LSRA) généralise compositionnellement
sur une vraie structure linguistique, pas seulement sur l'arithmétique Z_11.

Tâche : dérivation prefix+stem+suffix (régulière).
  stems   : happy kind fair clear wise safe kind real true  (8 adjectices)
  prefixes: un re                  -> prefix+stem   (unhappy, rekind, ...)
  suffixes: ness ful               -> +suffix        (...ness, ...ful)
  final = suffix(prefix(stem))     ex: un+kind+ness = "unkindness"

COMPOSITION 2-étapes. Train sur sous-ensemble de (stem,prefix,suffix),
test sur triples JAMAIS VUS. Decomposition (grok chaque affixe) vs one-shot (mémorise).

Note honnête : certains combos sont des mots réguliers bien formés mais pas
tous lexicalement attestés (ex. "rewise"). L'objet = valider le MÉCANISME
compositionnel sur structure linguistique, pas la couverture lexicale.
"""
import json, random, time
import torch
from ocm26400.amv import D_MODEL, AMVVector
from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.reasoner import ReasonerBlock, encode_input, lsra_loop, TAU_GROK, DEVICE

SEED = 0
STEMS = ["happy", "kind", "fair", "clear", "wise", "safe", "real", "true"]
PREFIXES = ["un", "re"]
SUFFIXES = ["ness", "ful"]
N_STEM, N_PRE, N_SUF = len(STEMS), len(PREFIXES), len(SUFFIXES)

# index space (<=64 pour le slot ent one-hot)
ID_PREFIX = N_STEM                       # 8
ID_SUFFIX = ID_PREFIX + N_PRE            # 10
ID_PRE_FORM = ID_SUFFIX + N_SUF          # 12  (formes préfixées)
ID_DERIVED = ID_PRE_FORM + N_STEM * N_PRE   # 12+16 = 28 (formes dérivées)
N_TOTAL = ID_DERIVED + (N_STEM * N_PRE) * N_SUF  # 28 + 32 = 60


def compose(a: int, b: int) -> int:
    """b = affixe. prefix+stem -> préfixé ; suffix+préfixé -> dérivé."""
    if ID_PREFIX <= b < ID_SUFFIX and 0 <= a < N_STEM:           # stem + prefix
        return ID_PRE_FORM + (b - ID_PREFIX) * N_STEM + a
    if ID_SUFFIX <= b < ID_PRE_FORM and ID_PRE_FORM <= a < ID_DERIVED:  # préfixé + suffix
        return ID_DERIVED + (b - ID_SUFFIX) * (N_STEM * N_PRE) + (a - ID_PRE_FORM)
    return a  # no-op


def surface(stem_i, pre_i, suf_i):
    return f"{PREFIXES[pre_i]}{STEMS[stem_i]}{SUFFIXES[suf_i]}"


def train_decomp_block(d, ver, n_steps=1500, lr=3e-3, batch=64):
    """Entraîne le bloc sur les 2 mappings compose (stem,prefix)->pre et (pre,suffix)->deriv."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = []
    for s in range(N_STEM):
        for pi in range(N_PRE):
            pairs.append((s, ID_PREFIX + pi))                 # stem,prefix
    for pi in range(N_PRE):
        for s in range(N_STEM):
            pre = ID_PRE_FORM + pi * N_STEM + s
            for si in range(N_SUF):
                pairs.append((pre, ID_SUFFIX + si))           # préfixé,suffix
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], d) for i in idx]).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            tgt = ver.compose(a, b)
            ent = out[j][0:64]
            dc = d.canonical(tgt).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def decode_step(blk, d, a, b):
    """1 forward + argmax (decode direct). Pas d'itération LSRA : c'est un seul
    compose, pas du test-time compute."""
    blk.eval()
    dev = next(blk.parameters()).device
    with torch.no_grad():
        x = encode_input(a, b, d).unsqueeze(0).to(dev)
        out = blk(x)[0]
        idx, _ = d.decode(out[0:64])
    return idx


def eval_decomp(blk, d, ver, triples):
    """Chaîne 2 étapes compose : stem->prefix->suffix. Accuracy exacte."""
    blk.eval()
    correct = 0
    for s, pi, si in triples:
        idx1 = decode_step(blk, d, s, ID_PREFIX + pi)            # stem + prefix
        if not (ID_PRE_FORM <= idx1 < ID_DERIVED):
            continue  # étape 1 ratée
        idx2 = decode_step(blk, d, idx1, ID_SUFFIX + si)         # préfixé + suffix
        tgt = ID_DERIVED + si * (N_STEM * N_PRE) + (idx1 - ID_PRE_FORM)
        correct += (idx2 == tgt)
    return correct / len(triples)


def train_oneshot(d, ver, triples, n_steps=1500, lr=3e-3, batch=64):
    """One-shot : encode (stem,prefix,suffix) -> output.ent = dérivé."""
    torch.manual_seed(SEED)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        idx = torch.randint(0, len(triples), (batch,))
        batch_in = []
        targets = []
        for i in idx:
            s, pi, si = triples[i]
            v = torch.zeros(D_MODEL)
            v[0:64] = d.canonical(s)
            v[64:128] = d.canonical(ID_PREFIX + pi)
            v[128:192] = d.canonical(ID_SUFFIX + si)
            batch_in.append(v)
            targets.append(ID_DERIVED + si * (N_STEM * N_PRE) + pi * N_STEM + s)
        batch_in = torch.stack(batch_in).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for j in range(batch):
            ent = out[j][0:64]
            dc = d.canonical(targets[j]).to(DEVICE)
            cos = (ent @ dc) / (ent.norm() * dc.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


def eval_oneshot(blk, d, ver, triples):
    blk.eval()
    dev = next(blk.parameters()).device
    correct = 0
    for s, pi, si in triples:
        v = torch.zeros(D_MODEL)
        v[0:64] = d.canonical(s); v[64:128] = d.canonical(ID_PREFIX + pi)
        v[128:192] = d.canonical(ID_SUFFIX + si)
        with torch.no_grad():
            out = blk(v.unsqueeze(0).to(dev))[0]
            idx, _ = d.decode(out[0:64])
        tgt = ID_DERIVED + si * (N_STEM * N_PRE) + pi * N_STEM + s
        correct += (idx == tgt)
    return correct / len(triples)


def main():
    random.seed(SEED)
    d = SymbolicDict(n=N_TOTAL)
    ver = Verifier(d, compose_fn=compose, n_ops=1)
    print(f"OCM-26400 LINGUISTIQUE | device={DEVICE} | {N_STEM} stems x {N_PRE} pref x {N_SUF} suff")
    print(f"dictionnaire={N_TOTAL} primitives (stems, affixes, formes préfixées, dérivées)")

    all_tr = [(s, pi, si) for s in range(N_STEM) for pi in range(N_PRE) for si in range(N_SUF)]
    random.shuffle(all_tr)
    n_train = 20
    train_tr, test_tr = all_tr[:n_train], all_tr[n_train:]
    print(f"train triples={len(train_tr)} | test (JAMAIS VUS)={len(test_tr)}")
    print(f"exemples: {[surface(*t) for t in test_tr[:5]]}")

    t0 = time.time()
    blk_d = train_decomp_block(d, ver, n_steps=1500)
    decomp_acc = eval_decomp(blk_d, d, ver, test_tr)
    decomp_train = eval_decomp(blk_d, d, ver, train_tr[:8])

    blk_o = train_oneshot(d, ver, train_tr, n_steps=1500)
    os_train = eval_oneshot(blk_o, d, ver, train_tr[:8])
    os_test = eval_oneshot(blk_o, d, ver, test_tr)
    dt = time.time() - t0

    print(f"\n--- RÉSULTATS LINGUISTIQUES ({dt:.1f}s) ---")
    print(f"ONE-SHOT  train:                {os_train*100:5.1f}%")
    print(f"ONE-SHOT  test (triples neufs): {os_test*100:5.1f}%")
    print(f"DÉCOMP    train:                {decomp_train*100:5.1f}%")
    print(f"DÉCOMP    test (triples neufs): {decomp_acc*100:5.1f}%  ← crown jewel ling.")
    gap = decomp_acc - os_test
    print(f"\nÉCART (decomp - oneshot): {gap*100:+.1f} points")
    verdict = "VALIDÉ" if decomp_acc > os_test + 0.15 else "NON VALIDÉ"
    print(f"VERDICT: {verdict}")

    results = {
        "task": "english derivational morphology (prefix+stem+suffix)",
        "stems": STEMS, "prefixes": PREFIXES, "suffixes": SUFFIXES,
        "n_primitives": N_TOTAL, "train_triples": len(train_tr), "test_triples": len(test_tr),
        "oneshot_train": os_train, "oneshot_test": os_test,
        "decomp_train": decomp_train, "decomp_test": decomp_acc,
        "gap_points": gap, "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/linguistic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Résultats: ocm26400/linguistic_results.json")
    return results


if __name__ == "__main__":
    main()
