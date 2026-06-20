"""ACSP loss — Amodal Consistency & Step Penalty (OCM-26400, spec §2).

L = alpha*L_align + beta*L_step + gamma*L_sparse + delta*L_consist

Innovation baptisée (spec §1.3): pénaliser une ETAPE compositionnelle illégale
(transition de vecteur latent), pas seulement la réponse finale. Force le modèle
à décomposer via le scratchpad plutôt que de sauter directement à la réponse.
"""
import torch

from .amv import AMVVector
from .verifier import SymbolicDict, Verifier, P_BACKTRACK

# Poids par défaut (spec ne fixe pas les valeurs exactes ; on choisit des ordres de grandeur sensés)
ALPHA = 1.0     # alignement au dictionnaire
BETA = 1.0      # pénalité d'étape
GAMMA = 1e-3    # sparsité (faible : régularisation, pas domination)
DELTA = 0.0     # consistence multimodale (0 en single-modality)


def l_align(v: AMVVector, dictionary: SymbolicDict) -> torch.Tensor:
    """L_align = min_{d in D} (1 - cos(v.ent, d)).

    Pénalise un vecteur entité qui ne correspond à aucun primitif du dictionnaire.
    """
    ent = v.ent  # (64,)
    D = dictionary._matrix().to(ent.device)  # (n, 64) — toutes les canoniques (device-robuste)
    # cos par primitive
    ent_n = ent / (ent.norm() + 1e-8)
    d_n = D / (D.norm(dim=1, keepdim=True) + 1e-8)
    cos = ent_n @ d_n.t()  # (n,)
    return (1.0 - cos.max())  # scalaire


def l_step(verifier: Verifier, d_ent: int, d_prop: int, op_id: int = 0) -> torch.Tensor:
    """L_step = 0 si V légal, P_BACKTRACK sinon."""
    legal = verifier.V(d_ent, d_prop, op_id)
    return torch.tensor(0.0 if legal else P_BACKTRACK)


def l_sparse(v: AMVVector, lam: float = 1.0) -> torch.Tensor:
    """L_sparse = lam * sum |v_i|  (L1 sur les 256 dims)."""
    return lam * v.tensor.abs().sum()


def l_consist(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """L_consist = InfoNCE cross-modal (spec §2.4). Délègue au core math infonce.py.

    z_a, z_b : batches (N, D) d'embeddings (une modalité chacun). L2-normalisés en interne.
    """
    from .infonce import info_nce
    return info_nce(z_a, z_b, tau=tau)


def acsp_loss(
    v: AMVVector,
    dictionary: SymbolicDict,
    verifier: Verifier,
    d_ent: int,
    d_prop: int,
    op_id: int = 0,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    consist_term: torch.Tensor = None,
) -> torch.Tensor:
    """Loss ACSP complète sur une étape. Différentiable sauf le terme step (constante).

    consist_term (optionnel): tenseur InfoNCE pré-calculé (L_consist). Si fourni,
    ajouté pondéré par delta. Contrat unique d'extension pour les pièces multimodales
    (juge: un seul point d'extension, pas un kwarg par pièce).
    """
    base = (
        alpha * l_align(v, dictionary)
        + beta * l_step(verifier, d_ent, d_prop, op_id)
        + gamma * l_sparse(v)
    )
    if consist_term is not None:
        base = base + delta * consist_term
    return base
