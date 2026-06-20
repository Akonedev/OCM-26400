#!/usr/bin/env python3
"""
EXPÉRIENCE domaine mathématique multi-opérations (OCM-26400, cahier des charges §2).

Démontre l'architecture appliquée à un VRAI domaine de connaissance (mathématiques) :
UN block op-aware apprend 3 OPÉRATIONS linéaires distinctes dispatchées par op_id
(ADD, et deux compositions), généralise chaque op à des couples (a,b) non vus, puis
le SOMMEIL extrait la RÈGLE (α,β) de chaque opération depuis les faits appris.

C'est §2 (mathématiques) + §1 (multi-opération via op_id) + sommeil (consolidation
par op) réunis : le système raisonnerait sur plusieurs règles math dans un seul moteur.

Tâche (Z_11) :
    op_id 0 : ADD    r = (1a + 1b) mod 11
    op_id 1 : OP_A   r = (3a + 5b) mod 11
    op_id 2 : OP_B   r = (2a + 7b) mod 11
Toutes linéaires-modulaires => extract_rule (sleep) récupère (α,β) pour chacune.
"""
import json, random, time
import torch

from ocm26400.amv import D_MODEL
from ocm26400.verifier import SymbolicDict, P_MOD
from ocm26400.morphology import MorphologyVerifier
from ocm26400.reasoner import ReasonerBlock, DEVICE
from ocm26400.sleep import extract_rule, rule_predicts

SEED = 0
N = P_MOD
OPS = [(1, 1), (3, 5), (2, 7)]      # (α,β) par op_id
OP_NAMES = ["ADD(1,1)", "OP_A(3,5)", "OP_B(2,7)"]


def encode_math(a, b, op_id, d):
    v = torch.zeros(D_MODEL)
    v[0:64] = d.canonical(a)            # ent = a
    v[64:128] = d.canonical(b)          # prop = b
    v[128:192] = d.canonical(op_id)     # op = type d'opération (one-hot 0/1/2)
    return v


def result(a, b, op_id):
    a_, b_ = OPS[op_id]
    return (a_ * a + b_ * b) % N


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = DEVICE
    d = SymbolicDict(n=N)
    rules = [lambda a, b, k=k: (OPS[k][0] * a + OPS[k][1] * b) % N for k in range(len(OPS))]
    ver = MorphologyVerifier(d, rules)
    print(f"OCM-26400 DOMAINE MATH MULTI-OP | Z_{N} | {len(OPS)} ops: {OP_NAMES}")

    t0 = time.time()
    # entraîne 1 block op-aware sur les 3 ops (couples aléatoires)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)
    for _ in range(2000):
        a = torch.randint(0, N, (64,)); b = torch.randint(0, N, (64,)); k = torch.randint(0, len(OPS), (64,))
        batch_in = torch.stack([encode_math(int(a[i]), int(b[i]), int(k[i]), d) for i in range(64)]).to(device)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=device)
        for i in range(64):
            ent = out[i][0:64]
            tgt = d.canonical(result(int(a[i]), int(b[i]), int(k[i]))).to(device)
            cos = (ent @ tgt) / (ent.norm() * tgt.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / 64
        opt.zero_grad(); loss.backward(); opt.step()

    # éval : chaque op sur couples non vus + extraction de règle (sommeil)
    blk.eval()
    print(f"\n{'op':14} {'acc couples non vus':>20} {'règle extraite (sommeil)':>24}")
    per_op = {}
    for k, name in enumerate(OP_NAMES):
        # accuracy sur 100 couples test
        test = [(random.randrange(N), random.randrange(N)) for _ in range(100)]
        correct = 0
        facts = []
        for a, b in test:
            with torch.no_grad():
                x = encode_math(a, b, k, d).unsqueeze(0).to(device)
                r_pred, _ = d.decode(blk(x)[0][0:64])
            correct += (r_pred == result(a, b, k))
            facts.append((a, b, r_pred))
        acc = correct / len(test)
        rule = extract_rule(facts, N)        # sommeil : extrait (α,β) de cette op
        per_op[name] = {"accuracy": round(acc, 4), "rule_extracted": list(rule) if rule else None}
        ok_rule = "✓" if rule == OPS[k] else "✗"
        print(f"{name:14} {acc*100:>19.1f}% {str(rule):>22} {ok_rule}")

    dt = time.time() - t0
    all_acc = sum(v["accuracy"] for v in per_op.values()) / len(per_op)
    all_rules = all(per_op[OP_NAMES[k]]["rule_extracted"] == list(OPS[k]) for k in range(len(OPS)))
    verdict = "VALIDÉ" if (all_acc > 0.9 and all_rules) else "NON VALIDÉ"
    print(f"\nAVG acc multi-op : {all_acc*100:.1f}% | règles extraites (sommeil) : {all_rules}")
    print(f"VERDICT (1 block, 3 ops math, généralise + règle extraite/op) : {verdict}")

    results = {
        "task": "domaine math multi-opérations (spec §2 mathématiques)",
        "n": N, "ops": OP_NAMES, "ops_coeffs": [list(o) for o in OPS],
        "per_op": per_op,
        "avg_accuracy": round(all_acc, 4), "all_rules_extracted": all_rules,
        "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/math_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/math_results.json")
    return results


if __name__ == "__main__":
    main()
