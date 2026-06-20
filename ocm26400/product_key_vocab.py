"""Product-Key Memory — capacité 2√V + decode O(√V) (P-D, archi PRÉSERVÉE).

Techno : Product-Key Memory (Lample et al. 2024, Large Memory Layers). 2 sous-codebooks de
√V clés -> V=n_sub² vecteurs adressables en O(2·n_sub) au lieu de O(V). Pour V=1M :
2×1000 clés au lieu d'un codebook de 1M.

VRAI APPORT (mesuré) : CAPACITÉ (2√V clés -> mémoire + trainabilité) + decode O(√V).

HONNÊTE (correction du panel) : P-D N'améliore PAS la séparabilité — au contraire, les
produits (somme de 2 clés) clusterisent PLUS qu'un codebook plat (cos au plus proche
voisin mesuré : PK 0.58-0.62 vs LearnedVocab plat 0.35). Le plafond de packing R^64 est
GÉOMÉTRIQUE et n'est PAS levé par PQ. Ne pas claim le contraire.

INTERFACE IDENTIQUE à LearnedVocab (canonical/decode/_matrix/uniformity_loss/...). Le
noyau spectral (SpectralCoreBlock) et acsp.l_align le consomment SANS AUCUNE modification
-> l'architecture de l'utilisateur est PRÉSERVÉE (P-D ne touche pas au noyau).

* canonical(i) = normalize(K1[i//n_sub] + K2[i%n_sub])    (combinaison additive)
* decode(q)    : 2-stage (argmax sur K1, argmax sur K2) -> O(2·n_sub), PAS de O(V)
* _matrix()    : les V produits (V,dim) — pour le contrat l_align (lourd si V énorme)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .amv import PART
from .learned_vocab import TAU_PURE, DELTA_MARGIN


class ProductKeyVocab(nn.Module):
    """Vocabulaire product-key : 2 codebooks √V -> V produits, O(√V) decode."""

    def __init__(self, n: int, dim: int = PART, seed: int = 0):
        super().__init__()
        assert n >= 1 and dim >= 1
        self.n = n
        self.dim = dim
        self.n_sub = max(2, int(math.ceil(math.sqrt(n))))     # √V clés par codebook
        g = torch.Generator().manual_seed(seed)
        self.K1 = nn.Parameter(torch.randn(self.n_sub, dim, generator=g))
        self.K2 = nn.Parameter(torch.randn(self.n_sub, dim, generator=g))

    def _km(self, which: int) -> torch.Tensor:
        K = self.K1 if which == 1 else self.K2
        return K / (K.norm(dim=1, keepdim=True) + 1e-8)

    def canonical(self, idx: int) -> torch.Tensor:
        if not (0 <= idx < self.n):
            return torch.zeros(self.dim)
        a, b = idx // self.n_sub, idx % self.n_sub
        v = self._km(1)[a] + self._km(2)[b]
        return v / (v.norm() + 1e-8)

    def _matrix(self) -> torch.Tensor:
        """Tous les V produits (V, dim) unit-norm — pour l_align (lourd si V énorme)."""
        M1, M2 = self._km(1), self._km(2)
        P = (M1[:, None, :] + M2[None, :, :]).reshape(-1, self.dim)[: self.n]
        return P / (P.norm(dim=1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def decode(self, vec: torch.Tensor):
        """Decode 2-stage O(2·n_sub) : argmax sur K1 puis K2. Retourne (idx, valid)."""
        head = vec[: self.dim].to(torch.float32)
        q = head / (head.norm() + 1e-8)
        M1 = self._km(1).to(q.device); M2 = self._km(2).to(q.device)
        s1 = q @ M1.T                                            # (n_sub,)
        s2 = q @ M2.T                                            # (n_sub,)
        k1 = int(torch.argmax(s1).item())
        k2 = int(torch.argmax(s2).item())
        idx = k1 * self.n_sub + k2
        if idx >= self.n:
            idx = idx % self.n
        cand = self.canonical(idx).to(q.device)
        cos1 = float((q @ cand).item())
        # 2e meilleur produit : (k1, 2e k2) ou (2e k1, k2) -> max comme cos2 (marge)
        k1b = int(torch.topk(s1, 2).indices[1].item())
        k2b = int(torch.topk(s2, 2).indices[1].item())
        cand2 = max((q @ self.canonical(k1 * self.n_sub + k2b).to(q.device)).item(),
                    (q @ self.canonical(k1b * self.n_sub + k2).to(q.device)).item())
        valid = (cos1 >= TAU_PURE) and ((cos1 - cand2) >= DELTA_MARGIN)
        return idx, bool(valid)

    def uniformity_loss(self) -> torch.Tensor:
        """Anti-collapse : mean cos² sur les clés de CHAQUE codebook (2 codebooks séparés)."""
        loss = 0.0
        for K in (self.K1, self.K2):
            M = K / (K.norm(dim=1, keepdim=True) + 1e-8)
            G = M @ M.T
            iu = torch.triu_indices(self.n_sub, self.n_sub, offset=1)
            loss = loss + (G[iu[0], iu[1]] ** 2).mean()
        return loss / 2

    def mean_inter_pair_cos(self) -> float:
        """Diagnostic : cos moyen inter-paires sur un ÉCHANTILLON de produits (coût maîtrisé)."""
        with torch.no_grad():
            idx = torch.randperm(self.n)[: min(200, self.n)]
            vecs = torch.stack([self.canonical(int(i)) for i in idx])
            G = vecs @ vecs.T
            m = len(idx)
            iu = torch.triu_indices(m, m, offset=1)
            return float(G[iu[0], iu[1]].mean().item())

    def freeze(self):
        self.K1.requires_grad_(False); self.K2.requires_grad_(False)
        return self
