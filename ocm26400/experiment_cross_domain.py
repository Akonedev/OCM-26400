#!/usr/bin/env python3
"""
EXPÉRIENCE cross-domain inter-rule (OCM-26400, paradigme complet multi-domaines).

Le noyau spectral grokke 5 règles de 3 DOMAINES DIFFÉRENTS conjointement, puis GÉNÈRE
des compositions INTER-DOMAINES (math → chimie → biologie). C'est la preuve ultime du
paradigme : comprendre les règles de plusieurs domaines → composer à travers les domaines.

Règles cross-domaines (toutes sur Z_11, vérifiables) :
  Math      : add(a,b)=(a+b)%n, mul(a,b)=(a*b)%n, linop(a,b)=(3a+5b)%n
  Chimie    : react(a,b)=(a+b)%n (réaction A+B→produit)
  Biologie  : dna_complement(a)=(n-1-a)%n (complément ADN)

Inter-domaine : add(a,b) → react(result, c) → dna_complement(result2) = composition
qui traverse math→chimie→biologie en une chaîne cohérente.
"""
import json, random, time
import torch

from ocm26400.amv import D_MODEL
from ocm26400.verifier import SymbolicDict, P_MOD
from ocm26400.reasoner import ReasonerBlock, encode_input, DEVICE

N = P_MOD
CROSS_RULES = {
    "add":            (lambda a, b: (a + b) % N,           "math",      2),
    "mul":            (lambda a, b: (a * b) % N,           "math",      2),
    "linop":          (lambda a, b: (3 * a + 5 * b) % N,   "math",      2),
    "react":          (lambda a, b: (a + b) % N,           "chemistry", 2),
    "dna_complement": (lambda a: (N - 1 - a) % N,          "biology",   1),
}
CROSS_NAMES = list(CROSS_RULES.keys())


def encode_cross(a, b, rule_id, d):
    v = torch.zeros(D_MODEL)
    v[0:64] = d.canonical(a)
    v[64:128] = d.canonical(b)
    v[128:192] = d.canonical(rule_id)
    return v


def rule_fn(name, a, b=0):
    fn, dom, arity = CROSS_RULES[name]
    return fn(a, b) if arity == 2 else fn(a)


def train_cross(d, n_steps=2500, lr=3e-3, batch=128):
    torch.manual_seed(0)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        k = torch.randint(0, len(CROSS_NAMES), (batch,))
        a = torch.randint(0, N, (batch,)); b = torch.randint(0, N, (batch,))
        loss = torch.tensor(0.0, device=DEVICE)
        for i in range(batch):
            name = CROSS_NAMES[int(k[i])]; ai, bi = int(a[i]), int(b[i])
            x = encode_cross(ai, bi, int(k[i]), d).unsqueeze(0).to(DEVICE)
            out = blk(x)[0]
            ent = out[0:64]
            tgt = d.canonical(rule_fn(name, ai, bi)).to(DEVICE)
            cos = (ent @ tgt) / (ent.norm() * tgt.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk


@torch.no_grad()
def comprehend_cross(blk, d, n_test=50):
    blk.eval()
    acc = {}
    for rid, name in enumerate(CROSS_NAMES):
        ok = 0
        for _ in range(n_test):
            a = random.randrange(N); b = random.randrange(N)
            x = encode_cross(a, b, rid, d).unsqueeze(0).to(DEVICE)
            r, _ = d.decode(blk(x)[0][0:64])
            ok += (r == rule_fn(name, a, b))
        acc[name] = ok / n_test
    return acc


@torch.no_grad()
def generate_cross_chain(blk, d, chain, init):
    """Génère une chaîne inter-domaines : [(rule_name, operand)] depuis init."""
    blk.eval()
    cur = init
    for name, op in chain:
        rid = CROSS_NAMES.index(name)
        x = encode_cross(cur, op, rid, d).unsqueeze(0).to(DEVICE)
        cur, _ = d.decode(blk(x)[0][0:64])
    return cur


def cross_chain_gt(chain, init):
    cur = init
    for name, op in chain:
        cur = rule_fn(name, cur, op)
    return cur


def main():
    random.seed(0); torch.manual_seed(0)
    d = SymbolicDict(n=N)
    domains = set(dom for _, dom, _ in CROSS_RULES.values())
    print(f"CROSS-DOMAIN INTER-RULE | {len(CROSS_NAMES)} règles sur {len(domains)} domaines "
          f"({', '.join(sorted(domains))})\n")
    t0 = time.time()
    blk = train_cross(d, n_steps=2500)

    print("Compréhension par règle :")
    comp = comprehend_cross(blk, d)
    for name, acc in comp.items():
        dom = CROSS_RULES[name][1]
        print(f"  {name:18} ({dom:10}) : {acc*100:5.1f}%")

    print("\nGénération inter-domaines (chaînes mixtes math→chimie→bio) :")
    chains = [
        [("add", 3), ("react", 5), ("dna_complement", 0)],
        [("mul", 2), ("dna_complement", 0), ("add", 7)],
        [("linop", 4), ("react", 2), ("dna_complement", 0)],
    ] * 10
    ok = 0
    for c in chains:
        init = random.randrange(N)
        pred = generate_cross_chain(blk, d, c, init)
        truth = cross_chain_gt(c, init)
        ok += (pred == truth)
    acc = ok / len(chains)
    print(f"  inter-domaines : {ok}/{len(chains)} = {acc*100:.1f}%")

    dt = time.time() - t0
    verdict = "VALIDÉ" if acc > 0.7 and all(v > 0.85 for v in comp.values()) else "PARTIEL"
    print(f"\nVERDICT (cross-domain inter-rule, {len(domains)} domaines) : {verdict}")
    json.dump({"task": "cross-domain inter-rule", "domains": list(domains),
               "n_rules": len(CROSS_NAMES), "comprehension": comp,
               "inter_domain_acc": round(acc, 4), "verdict": verdict,
               "duration_s": round(dt, 1)},
              open("ocm26400/cross_domain_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
