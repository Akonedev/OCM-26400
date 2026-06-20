"""Orchestrateur d'entraînement REPRODUCTIBLE (OCM-26400) — version exécutable des procédures.

Suit EXACTEMENT l'ordre du DAG imposé par le juge (voir PROCEDURES.md §0, §2-§4) :

    STAGE 0  Build       : SymbolicDict + Verifier (ground-truth symbolique)
    STAGE 1  Pre-train   : train_with_acsp → grok binaire (ACSP DIFFÉRENTIABLE, Gumbel ST)
    STAGE 2  Gate        : train_reasoner_with_confidence → meta[0] confiant (LSRA arrêtable)
    STAGE 3  Multi-rule  : train_omni_rules → cross-domain (add/mul/linop conjointes)
    STAGE 4  Eval        : bench.py (LEVEL agrégé) + eval_harness (pipeline SOTA)

REPRODUCTIBILITÉ (PROCEDURES.md §0) :
    seed = 0 ; device = cuda si dispo ; optimizer = Adam(lr=3e-3) [ocm26400 canonique]
    (ATTENTION : spxlm_v6 utilise AdamW(lr=1e-3, wd=0.1) — NE PAS confondre — voir PROCEDURES.md §5)

NON-SATURATION GPU (directive utilisateur) :
    les stages tournent SÉQUENTIELLEMENT (pas de parallélisme GPU). Par défaut `--smoke`
    (peu de steps, valide le pipeline). `--full` pour l'entraînement réel (plus long).

USAGE :
    python3 -m ocm26400.train                 # smoke (pipeline complet, ~30s)
    python3 -m ocm26400.train --full          # entraînement réel
    python3 -m ocm26400.train --stages 1,2    # seulement certains stages
"""
from __future__ import annotations
import argparse
import json
import os
import time
import torch

from .amv import D_MODEL
from .verifier import SymbolicDict, Verifier
from .diff_decode import train_with_acsp
from .reasoner import train_reasoner_with_confidence, lsra_loop, encode_input
from . import bench


HERE = os.path.dirname(os.path.abspath(__file__))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def stage0_build(p_mod: int = 11) -> tuple:
    """STAGE 0 — ground-truth symbolique (PROCEDURES.md §1)."""
    d = SymbolicDict(n=p_mod, dim=64)
    ver = Verifier(d, n_ops=1)
    print(f"[STAGE 0] SymbolicDict(n={p_mod}) + Verifier op(a,b)=(3a+5b) mod {p_mod}")
    return d, ver


def stage1_pretrain(d, ver, n_steps: int, device: str):
    """STAGE 1 — PRE-TRAINING CANONIQUE (PROCEDURES.md §2) : train_binary_block.

    ⚠️ PROCÉDURE OBLIGATOIRE : train_binary_block — loss (1−cos), Adam lr=3e-3, batch 64,
    seed 0, n_steps 1500. C'EST la procédure qui produit le grok binaire 100% (crown-jewel,
    experiment_composition). NE PAS substituer une autre loss sinon le grok n'a pas lieu
    (directive utilisateur : « suivre les procédures sinon ça ne marchera pas »).

    Vérification du grokking (PROCEDURES.md §2) : binary_acc >= 0.99 sur données non vues."""
    from .experiment_composition import train_binary_block
    print(f"[STAGE 1] train_binary_block (PROCÉDURE §2, n_steps={n_steps})...")
    t0 = time.time()
    blk = train_binary_block(d, ver, n_steps=n_steps)   # loss 1-cos, Adam 3e-3, seed 0
    # quick grok check : le block prédit-il compose(a,b) sur quelques paires ?
    acc = _quick_grok_acc(blk, d, ver, device)
    print(f"[STAGE 1] done {time.time()-t0:.1f}s | grok_acc≈{acc:.2f} (cible ≥0.99)")
    return blk, acc


def stage2_gate(d, ver, n_steps: int, device: str):
    """STAGE 2 — entraîne meta[0] confiant → gate LSRA arrêtable (PROCEDURES.md §2).

    C'est ce qui rend la boucle v(t+1)=Block(v(t)) stoppable à sigmoid(meta[0])>=τ_grok."""
    print(f"[STAGE 2] train_reasoner_with_confidence (n_steps={n_steps})...")
    t0 = time.time()
    blk = train_reasoner_with_confidence(d, ver, n_steps=n_steps, lr=3e-3,
                                         batch=64, device=device)
    # check : lsra_loop converge-t-elle avant max_iter ?
    depth = _quick_depth(blk, d, ver, device)
    print(f"[STAGE 2] done {time.time()-t0:.1f}s | depth moyen≈{depth:.1f} (cible < max_iter)")
    return blk, depth


def stage3_multirule(n_steps: int, device: str):
    """STAGE 3 — multi-règles conjointes (add/mul/linop) → cross-domain (PROCEDURES.md §4)."""
    from .omni_rules import train_omni_rules
    print(f"[STAGE 3] train_omni_rules (add/mul/linop, n_steps={n_steps})...")
    t0 = time.time()
    res = train_omni_rules(n_steps=n_steps, lr=3e-3)
    print(f"[STAGE 3] done {time.time()-t0:.1f}s")
    return res


def stage4_eval() -> dict:
    """STAGE 4 — LEVEL agrégé (bench.py) + pipeline SOTA (eval_harness démo)."""
    print("[STAGE 4] bench.run_bench() + eval_harness.run_demo()...")
    level = bench.run_bench()
    from .eval_harness import run_demo
    demo = run_demo()
    print(f"[STAGE 4] LEVEL={level.get('LEVEL')} | harness verdict={demo['comparison']['verdict']}")
    return {"level": level, "harness": demo}


# ---------------- helpers de vérification rapide ----------------

@torch.no_grad()
def _quick_grok_acc(blk, d, ver, device, n_check: int = 20) -> float:
    """Accuracy rapide du grok binaire sur n_check paires (proxy, pas le test complet)."""
    ok = 0
    for a in range(min(n_check, d.n)):
        for b in range(min(2, d.n)):
            x = encode_input(a, b, d).to(device)
            v = blk(x.unsqueeze(0))[0]
            pred = int(v[: d.n].argmax().item())
            if pred == ver.compose(a, b):
                ok += 1
    total = min(n_check, d.n) * min(2, d.n)
    return ok / max(total, 1)


@torch.no_grad()
def _quick_depth(blk, d, ver, device, n_check: int = 5) -> float:
    """Profondeur moyenne de convergence de lsra_loop (proxy)."""
    from .reasoner import TAU_GROK
    depths = []
    for a in range(min(n_check, d.n)):
        b = 0
        x0 = encode_input(a, b, d).to(device)
        v = x0
        for t in range(8):
            v = blk(v.unsqueeze(0))[0]
            conf = torch.sigmoid(v[2 * 64]).item()
            if conf >= TAU_GROK:
                depths.append(t + 1)
                break
        else:
            depths.append(8)
    return sum(depths) / max(len(depths), 1)


# ---------------- orchestration ----------------

def run_pipeline(stages: list, smoke: bool = True, device: str = None) -> dict:
    """Exécute les stages demandés SÉQUENTIELLEMENT (non-saturation GPU)."""
    device = device or _device()
    smoke_steps = {"s1": 200, "s2": 150, "s3": 300}
    full_steps = {"s1": 1500, "s2": 800, "s3": 2000}
    steps = smoke_steps if smoke else full_steps
    report = {"device": device, "smoke": smoke, "stages_run": []}

    if 0 in stages:
        d, ver = stage0_build()
        report["stages_run"].append(0)
    else:
        d, ver = stage0_build()

    if 1 in stages:
        _, acc = stage1_pretrain(d, ver, steps["s1"], device)
        report["grok_acc"] = acc
        report["stages_run"].append(1)
    if 2 in stages:
        _, depth = stage2_gate(d, ver, steps["s2"], device)
        report["mean_depth"] = depth
        report["stages_run"].append(2)
    if 3 in stages:
        stage3_multirule(steps["s3"], device)
        report["stages_run"].append(3)
    if 4 in stages:
        report["eval"] = stage4_eval()
        report["stages_run"].append(4)
    return report


def main():
    ap = argparse.ArgumentParser(description="Orchestrateur entraînement OCM-26400")
    ap.add_argument("--full", action="store_true", help="entraînement réel (vs smoke)")
    ap.add_argument("--stages", default="0,1,2,3,4",
                    help="stages séparés par virgule (ex: 1,2)")
    args = ap.parse_args()
    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    t0 = time.time()
    report = run_pipeline(stages, smoke=not args.full)
    report["total_time_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE, "train_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[PIPELINE] terminé en {report['total_time_s']}s | stages={report['stages_run']}")
    print(f"[PIPELINE] rapport → {out}")


if __name__ == "__main__":
    main()
