#!/usr/bin/env python3
"""PRE-TRAINING AUTOMATIQUE — grok les primitives de TOUS les domaines.

Suit la procédure canonique (PROCEDURES.md §2):
1. Sanity check P1 (train_binary_block 100 steps, loss → 0)
2. Curriculum v4 Phase 1 SOLO (chaque opérateur grok individuellement à gate ≥ 0.99)
3. Phase 2 SOMMEIL (obligatoire, consolide mémoire → compréhension)
4. Phase 3 CASCADE (scratchpad, composition profonde)

Usage: python auto_pretrain.py [--device cpu|cuda] [--steps 2000]
"""
import torch, argparse, time, json
from ocm26400.amv import AMVVector
from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.reasoner import encode_input, TAU_GROK, train_reasoner_with_confidence
from ocm26400.experiment_composition import train_binary_block, eval_binary, lsra_solve
from ocm26400.primitive_grok_curriculum import run_curriculum_v4, GATES

def main():
    ap = argparse.ArgumentParser(description="Pre-training automatique OCM-26400")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--cascade-depth", type=int, default=3)
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)
    t0 = time.time()
    print("="*60)
    print("AUTO PRE-TRAINING — curriculum v4 (SOLO → sommeil → cascade)")
    print(f"device={device} steps={args.steps} depth={args.cascade_depth}")
    print("="*60)
    # sanity check
    from ocm26400.unified_training import run_unified_pipeline
    rep = run_unified_pipeline(device=device)
    # curriculum v4 complet
    cur = run_curriculum_v4(solo_steps=args.steps, cascade_depth=args.cascade_depth, device=device)
    report = {
        "pipeline_verdict": rep.get("overall_verdict"),
        "curriculum_verdict": cur.verdict,
        "cascade_acc": cur.cascade_acc,
        "gates": GATES,
        "time_s": round(time.time() - t0, 1),
        "device": device,
    }
    print(f"\n{'='*60}\nPRE-TRAINING COMPLETE ({report['time_s']}s)")
    print(f"  pipeline: {report['pipeline_verdict']}")
    print(f"  curriculum: {report['curriculum_verdict']}")
    print(f"  cascade depth {args.cascade_depth}: {report['cascade_acc']*100:.0f}%")
    with open("ocm26400/pretrain_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  rapport: ocm26400/pretrain_report.json")

if __name__ == "__main__":
    main()
