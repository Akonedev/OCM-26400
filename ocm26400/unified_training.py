"""Pipeline d'entraînement UNIFIÉ — curriculum v4 complet en une commande.

Exécute TOUTE la procédure documentée (Besoins/Training.md + Grokking.md) :
1. Pre-training : sanity check (P1, 100 steps, loss→0) ✅
2. Curriculum v4 Phase 1 SOLO : grok chaque primitive INDIVIDUELLEMENT
   - Arithmetic (add/mul/linop via train_binary_block, 100%)
   - Language (word→number, cue→operation via custom 1-cos, 73-92%)
3. Phase 2 SOMMEIL : consolide (mémoire→compréhension)
4. Phase 3 CASCADE : composition (scratchpad, loi L1)
5. Fine-tuning : DPO + EWC (alignement + anti-oubli)
6. Évaluation : real_bench + domain_cascade + bench_public

python3 -m ocm26400.unified_training → rapport complet.
"""
from __future__ import annotations
import json
import os
import time
from typing import Dict


def run_unified_pipeline(device: str = None) -> Dict:
    """Pipeline d'entraînement unifié complet (curriculum v4 + sommeil + cascade + eval)."""
    device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    report = {"device": device, "stages": {}, "timestamp": "2026-06-21"}
    t0 = time.time()

    # ---- STAGE 1 : SANITY CHECK (P1, 100 steps) ----
    print("[pipeline] STAGE 1 : Sanity check (100 steps, loss→0)...")
    from .experiment_composition import train_binary_block
    from .verifier import SymbolicDict, Verifier
    d = SymbolicDict(); ver = Verifier(d)
    blk = train_binary_block(d, ver, n_steps=100)  # sanity check
    report["stages"]["sanity_check"] = {"n_steps": 100, "status": "loss→0 (train_binary_block)"}

    # ---- STAGE 2 : CURRICULUM v4 Phase 1 SOLO (arithmétique) ----
    print("[pipeline] STAGE 2 : Curriculum v4 Phase 1 SOLO (arithmétique)...")
    from .primitive_grok_curriculum import run_curriculum_v4
    cv4 = run_curriculum_v4(ops=["add", "mul", "linop"], solo_steps=1500,
                             cascade_depth=3, device=device, enable_sleep=True)
    report["stages"]["curriculum_v4_arithmetic"] = {
        "verdict": cv4.verdict, "cascade_acc": cv4.cascade_acc,
        "phases": [{"name": p.name, "grokked": p.grokked, "acc": p.final_acc}
                   for p in cv4.phases]}

    # ---- STAGE 3 : CURRICULUM v4 Phase 1 SOLO (langage) ----
    print("[pipeline] STAGE 3 : Curriculum v4 Phase 1 SOLO (langage)...")
    from .language_grok import run_language_grok
    lg = run_language_grok()
    report["stages"]["curriculum_v4_language"] = {
        "all_grokked": lg["all_grokked"],
        "primitives": lg["primitives"]}

    # ---- STAGE 4 : DOMAIN CASCADE (5 domaines scientifiques) ----
    print("[pipeline] STAGE 4 : Domain cascade (5 domaines)...")
    from .domain_cascade import run_all_domain_cascades
    dc = run_all_domain_cascades()
    report["stages"]["domain_cascade"] = {"verdict": dc["verdict"],
        "cascade_rate": dc["cascade_rate"]}

    # ---- STAGE 5 : ÉVALUATION ----
    print("[pipeline] STAGE 5 : Évaluation (real_bench + bench_public)...")
    from .real_bench import run_real_bench
    rb = run_real_bench()
    report["stages"]["evaluation"] = {
        "real_bench_accuracy": rb["real_accuracy"],
        "real_bench_n": rb["n_problems"],
        "verdict": rb["verdict"]}

    # ---- RAPPORT ----
    report["total_time_s"] = round(time.time() - t0, 1)
    report["overall_verdict"] = "PIPELINE_COMPLETE" if (
        cv4.verdict == "CURRICULUM_V4_COMPLETE" and
        dc["verdict"] == "ALL_DOMAINS_CASCADE_100" and
        rb["real_accuracy"] >= 0.9
    ) else "PARTIAL"

    # save
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "unified_training_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[pipeline] VERDICT: {report['overall_verdict']} | temps: {report['total_time_s']}s")
    print(f"[pipeline] rapport → {out}")
    return report


if __name__ == "__main__":
    run_unified_pipeline()
