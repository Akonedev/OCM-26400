"""primitive_grok_curriculum.py — ADR-0030 — LE curriculum v4 du projet (À IMPLÉMENTER).

Implémente le curriculum documenté dans Besoins/Grokking.md + Training.md + Formule_Lois_Grokking.md.
C'est LA procédure de training que je n'avais PAS suivie.

Lois (L1-L6) :
* L1 DÉCOMPOSITION > SCALE : la compétence vient de la STRUCTURE du calcul (étapes),
  pas de la masse. 0.75→0.98 par décomposition (scratchpad : intermédiaire puis final).
* L2 MASQUAGE INCRÉMENTAL : masquer un SOUS-ENSEMBLE des intermédiaires (partiellement
  visibles) apprend chaque étape comme une op 1-pas en contexte → cascade à l'inférence.
* L3 DEPTH_MAX ≈ 1/(1−per_step) : per-step exact → profondeur ∞.
* L4 RÉCURRENCE ⊥ LONGUEUR ⊥ PARAMS : raisonner = ajouter des étapes, pas des params.
* L5 L = 1+4·D : format de séquence (L8 = 1+4·(depth-1)).
* L6 ASSOCIATION : 1-source direct, multi-source = décomposer.

Curriculum v4 (DOSC L8 — chaque slot SOLO jusqu'à grokking) :
* Phase 1 SOLO : chaque opérateur grok INDIVIDUELLEMENT à gate≥0.99 (2k-6k steps).
* Phase 2 INTERLEAVED : mélange les opérateurs grokkés + scratchpad cascade.
* Gate thresholds : L1≥0.99, L2≥0.95, L5≥0.90, L6≥0.85.
* Scratchpad : calcule l'intermédiaire m=(a∘b), PUIS le final r=(m∘c) — explicite.
* Masquage partiel : intermédiaires partiellement visibles à l'entraînement.
* Sommeil OBLIGATOIRE entre phases (consolide mémoire→compréhension, sleep_phases).

Utilise le SpectralCoreBlock (FFT, MODEL UNIFIÉ, pas de transformer).
"""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL
from .verifier import SymbolicDict, Verifier
from .reasoner import ReasonerBlock, encode_input, lsra_loop
from .diff_decode import train_with_acsp
from .experiment_composition import train_binary_block
from .sleep_phases import full_night

# ---- Gates (Training.md) ----
GATES = {"L1": 0.99, "L2": 0.95, "L5": 0.90, "L6": 0.85}
N_MOD = 11


@dataclass
class CurriculumPhase:
    """Une phase du curriculum v4."""
    name: str
    depth: int                          # profondeur de composition (1=binaire solo)
    ops: List[str]                      # opérateurs concernés
    interleaved: bool = False           # SOLO (False) ou INTERLEAVED (True)
    gate: float = 0.99                  # seuil d'acceptation
    steps: int = 3000                   # 2k-6k
    grokked: bool = False
    final_acc: float = 0.0


@dataclass
class CurriculumReport:
    phases: List[CurriculumPhase] = field(default_factory=list)
    cascade_acc: float = 0.0            # accuracy scratchpad cascade
    sleep_report: Optional[dict] = None
    verdict: str = ""


def _op(name: str):
    ops = {"add": lambda a, b: (a + b) % N_MOD,
           "mul": lambda a, b: (a * b) % N_MOD,
           "linop": lambda a, b: (3 * a + 5 * b) % N_MOD}
    return ops[name]


def _eval_op_block(blk, d, ver, op_name: str, device: str, n_test: int = 50) -> float:
    """Accuracy d'un block sur un opérateur (hold-out)."""
    from .amv import AMVVector
    op = _op(op_name)
    blk.eval()
    correct = 0
    rng = random.Random(42)
    with torch.no_grad():
        for _ in range(n_test):
            a, b = rng.randint(0, N_MOD - 1), rng.randint(0, N_MOD - 1)
            x = encode_input(a, b, d).unsqueeze(0).to(device)
            out = blk(x)[0]
            pred, _ = d.decode(AMVVector(out).ent)
            if pred == op(a, b):
                correct += 1
    return correct / n_test


def _make_op_verifier(op_name: str) -> Tuple[SymbolicDict, Verifier]:
    d = SymbolicDict(n=N_MOD, dim=64)
    op = _op(op_name)

    class _V(Verifier):
        def compose(self, a, b, op_id=0):
            return op(a, b)
    return d, _V(d, n_ops=1)


def _train_solo_slot(op_name: str, n_steps: int, device: str) -> Tuple[ReasonerBlock, float]:
    """Phase SOLO : grok un opérateur INDIVIDUELLEMENT (procédure §2 : train_binary_block)."""
    d, ver = _make_op_verifier(op_name)
    torch.manual_seed(0)                # §0.2
    blk = train_binary_block(d, ver, n_steps=n_steps)   # loss 1-cos, Adam 3e-3
    acc = _eval_op_block(blk, d, ver, op_name, device)
    return blk, acc


def _scratchpad_cascade_eval(blk, d, op_name: str, device: str,
                              depth: int = 3, n_test: int = 50) -> float:
    """Évaluation SCRATCHPAD CASCADE (loi L1) : calcule l'intermédiaire PUIS le final.
    m1 = op(a,b) ; m2 = op(m1,c) ; ... — chaque étape est une application 1-pas du block
    grokké. C'est le mécanisme qui fait bondir 0.75→0.98."""
    from .amv import AMVVector
    op = _op(op_name)
    blk.eval()
    rng = random.Random(7)
    correct = 0
    with torch.no_grad():
        for _ in range(n_test):
            vals = [rng.randint(0, N_MOD - 1) for _ in range(depth + 1)]
            # ground truth cascade
            gt = vals[0]
            for v in vals[1:]:
                gt = op(gt, v)
            # scratchpad : calcule chaque intermédiaire explicitement
            acc = vals[0]
            for v in vals[1:]:
                x = encode_input(acc, v, d).unsqueeze(0).to(device)
                out = blk(x)[0]
                pred, _ = d.decode(AMVVector(out).ent)
                acc = pred if pred is not None else acc
            if acc == gt:
                correct += 1
    return correct / n_test


def run_curriculum_v4(ops: List[str] = None, solo_steps: int = 3000,
                       cascade_depth: int = 3, device: str = None,
                       enable_sleep: bool = True) -> CurriculumReport:
    """Curriculum v4 complet (ADR-0030) :
    1. Phase SOLO : chaque opérateur grok INDIVIDUELLEMENT à gate≥0.99.
    2. Sommeil (obligatoire) : consolide.
    3. Scratchpad cascade : évalue la composition profonde (L1).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ops = ops or ["add", "mul", "linop"]
    report = CurriculumReport()

    # ---- Phase 1 : SOLO (chaque op grok individuellement) ----
    print(f"[curriculum v4] Phase 1 SOLO : {ops} (gate≥{GATES['L1']})")
    best_blk, best_d, best_op = None, None, ops[0]
    for op_name in ops:
        blk, acc = _train_solo_slot(op_name, solo_steps, device)
        grokked = acc >= GATES["L1"]
        phase = CurriculumPhase(name=f"solo_{op_name}", depth=1, ops=[op_name],
                                interleaved=False, gate=GATES["L1"], steps=solo_steps,
                                grokked=grokked, final_acc=acc)
        report.phases.append(phase)
        print(f"  solo {op_name}: acc={acc*100:.1f}% gate={GATES['L1']*100:.0f}% "
              f"{'✓ GROKKED' if grokked else '✗ (sous le gate)'}")
        if grokked and best_blk is None:
            d, _ = _make_op_verifier(op_name)
            best_blk, best_d, best_op = blk, d, op_name

    # ---- Phase 2 : SOMMEIL (obligatoire — mémoire→compréhension) ----
    if enable_sleep and best_blk is not None:
        print("[curriculum v4] Phase 2 SOMMEIL (consolidation mémoire→compréhension)")
        # génère des faits épisodiques depuis le slot grokké
        from .cognitive_agent import CognitiveAgent
        d, ver = _make_op_verifier(best_op)
        agent = CognitiveAgent(best_blk, d, ver)
        op = _op(best_op)
        for a in range(N_MOD):
            for b in range(min(4, N_MOD)):
                agent.memory[(a, b)] = op(a, b)
        night = full_night(agent.memory, extra_rules=[(3, 5), (1, 1)])
        report.sleep_report = night
        print(f"  sommeil : règle extraite={night.get('rule_learned')} "
              f"connexions={night.get('new_creative_connections')}")

    # ---- Phase 3 : SCRATCHPAD CASCADE (L1 décomposition) ----
    if best_blk is not None:
        print(f"[curriculum v4] Phase 3 SCRATCHPAD CASCADE depth={cascade_depth} (loi L1)")
        d, _ = _make_op_verifier(best_op)
        cascade_acc = _scratchpad_cascade_eval(best_blk, d, best_op, device,
                                                depth=cascade_depth)
        report.cascade_acc = cascade_acc
        print(f"  cascade depth {cascade_depth}: acc={cascade_acc*100:.1f}%")

    # verdict
    all_grokked = all(p.grokked for p in report.phases if not p.interleaved)
    report.verdict = ("CURRICULUM_V4_COMPLETE" if all_grokked and report.cascade_acc >= 0.9
                      else "CURRICULUM_PARTIAL")
    print(f"[curriculum v4] verdict: {report.verdict}")
    return report


if __name__ == "__main__":
    rep = run_curriculum_v4(solo_steps=2000, cascade_depth=3)
