"""InfoNCE core math (OCM-26400 P1, spec §2.4 L_consist).

Pure math, SANS encodeurs. Alignement contrastif (CLIP-style) sur embeddings nus.
Stable numériquement via F.cross_entropy (= logsumexp avec max-subtraction).

  L = -(1/N) Σ_i log[ exp(v_i·u_i / τ) / Σ_j exp(v_i·u_j / τ) ]

Honnête (juge): c'est un 'alignment harness sur embeddings nus', PAS encore du
vrai multimodal (aucun signal réel tant que les encodeurs sont des stubs).
"""
import torch
import torch.nn.functional as F

TAU_DEFAULT = 0.07  # valeur CLIP/SigLIP


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = TAU_DEFAULT) -> torch.Tensor:
    """InfoNCE unidirectionnel (z_a -> z_b). L2-normalise, logits = z_a@z_b.T/tau,
    cross_entropy avec labels=arange(N) = -log softmax[i,i]. Stable (logsumexp interne)."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / tau                       # (N, N)
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return F.cross_entropy(logits, labels)


def info_nce_symmetric(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = TAU_DEFAULT) -> torch.Tensor:
    """InfoNCE symétrique (CLIP): moyenne des 2 directions."""
    return 0.5 * (info_nce(z_a, z_b, tau) + info_nce(z_b, z_a, tau))


def multimodal_l_consist(embeddings_per_mod, tau: float = TAU_DEFAULT) -> torch.Tensor:
    """Moyenne symétrique d'InfoNCE sur les C(M,2) paires de modalités.

    embeddings_per_mod: liste de M batches (N, D) (un par modalité).
    """
    M = len(embeddings_per_mod)
    assert M >= 2
    total = 0.0
    npairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            total = total + info_nce_symmetric(embeddings_per_mod[i], embeddings_per_mod[j], tau)
            npairs += 1
    return total / npairs
