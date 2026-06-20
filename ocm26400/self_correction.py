"""Auto-correction / auto-amélioration (OCM-26400, cahier des charges 's'améliore
automatiquement' + demande utilisateur 's'autocorriger et s'autoaméliorer').

Le modèle se corrige lui-même : son module de RAISONNEMENT sert de self-check contre
sa MÉMOIRE. Pour chaque fait stocké (a,b)->r, on RE-RAISONNE ; si le résultat diffère
du fait mémorisé -> CONFLIT détecté -> correction. C'est la self-consistency (technique
réelle de vérification : un raisonnement fiable rattrape les erreurs de mémorisation).

* reason_pair(agent, a, b, noise) : re-raisonne (a,b) via le block.
* self_consistency_confidence(...) : k re-raisonnements (bruités) -> taux d'accord
  (mesure d'incertitude interne ; accord bas = fait peu fiable).
* self_correct(agent, verifier)    : re-raisonne chaque fait mémoire, CORRIGE les
  conflits. Justesse avant -> après.
* self_improve(agent, verifier)    : itère self_correct, courbe de justesse.

HONNÊTE : la détection est INTERNE (re-raisonnement, pas vérité externe). Le verifier
(compose, vérité) n'est utilisé que pour MESURER la justesse (métrique), pas pour
détecter. Si le block est fiable (grokké), il rattrape les erreurs de mémoire ; s'il
est peu fiable, self_consistency_confidence signale l'incertitude (à ne pas corriger
aveuglément).
"""
from collections import Counter
from typing import Optional, List, Dict
import torch

from .reasoner import encode_input


@torch.no_grad()
def reason_pair(agent, a: int, b: int, noise_std: float = 0.0) -> Optional[int]:
    """Re-raisonne (a,b) via le block de l'agent (option : bruit pour variation)."""
    blk = agent.blk
    d = agent.d
    dev = next(blk.parameters()).device
    x = encode_input(a, b, d).unsqueeze(0).to(dev)
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    out = blk(x)[0]
    r_pred, _ = d.decode(out[0:64])
    return r_pred


@torch.no_grad()
def self_consistency_confidence(agent, a: int, b: int, k: int = 5,
                                noise_std: float = 0.5) -> float:
    """Taux d'accord de k re-raisonnements bruités (1.0 = certain, bas = incertain).

    Mesure d'incertitude INTERNE : un fait peu fiable produit des prédictions qui
    varient sous bruit -> accord bas."""
    preds = [reason_pair(agent, a, b, noise_std) for _ in range(k)]
    c = Counter(preds)
    return max(c.values()) / k


def _accuracy(agent, verifier) -> float:
    if not agent.memory:
        return 1.0
    ok = sum(1 for (a, b), r in agent.memory.items() if r == verifier.compose(a, b))
    return ok / len(agent.memory)


@torch.no_grad()
def self_correct(agent, verifier=None, noise_std: float = 0.0) -> Dict:
    """Passe d'auto-correction : re-raisonne chaque fait mémoire, CORRIGE les conflits.

    Le verifier (vérité) sert uniquement à MESURER la justesse avant/après (métrique).
    La détection est interne (re-raisonnement)."""
    acc_before = _accuracy(agent, verifier) if verifier else None
    checked = corrected = 0
    for (a, b), r_stored in list(agent.memory.items()):
        r_pred = reason_pair(agent, a, b, noise_std)
        checked += 1
        if r_pred is not None and r_pred != r_stored:
            agent.memory[(a, b)] = r_pred       # CORRECTION
            corrected += 1
    acc_after = _accuracy(agent, verifier) if verifier else None
    return {"checked": checked, "corrected": corrected,
            "acc_before": acc_before, "acc_after": acc_after}


def self_improve(agent, verifier, rounds: int = 10, noise_std: float = 0.0) -> List[Dict]:
    """Itère l'auto-correction jusqu'à convergence (plus aucun conflit). Courbe de justesse."""
    curve = []
    for _ in range(rounds):
        stats = self_correct(agent, verifier, noise_std)
        curve.append(stats)
        if stats["corrected"] == 0:
            break
    return curve
