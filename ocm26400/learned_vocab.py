"""LearnedVocab — dictionnaire DENSE préservant l'identité (OCM-26400, P2).

Alternative dense au SymbolicDict one-hot. Chaque primitive = un vecteur dense
unit-norm dans le slot ent (64 dims), tiré d'une table apprise E ∈ R^{V×64}.

HONNÊTETÉ (verdict panel d'experts 19/06, pièce P2 « scale langage ») :

* CLAIM RETIRÉ — similarité distributionnelle (walk~walked). Aucun terme de la
  loss n'injecte de signal distributionnel ; l_align est une loss d'IDENTITÉ et
  le hinge traite toutes les paires également. LearnedVocab = encodage dense
  préservant l'identité qui supporte la GÉNÉRALISATION DE DÉCOMPOSITION
  (crown-jewel), PAS une similarité sémantique. (cf. falsification DA.)

* V > 64 possible — les embeddings denses n'imposent pas assert n<=dim
  (contrairement à SymbolicDict verifier.py:28). Vérifié empiriquement
  (experiment_vocab_scale.py sur Z_120).

* ANTI-COLLAPSE SÉRIEUX (Leçon 5) — uniformity_loss en COSINUS (pénalité sur
  cos² inter-paires), garde-fou de rang secondaire. Le rang seul ne capte pas
  le collapse clusterisé full-rank, d'où la loss d'uniformité comme primaire.

* DECODE par plus proche voisin cosinus + marge de pureté
  (cos1 ≥ TAU_PURE ET cos1 − cos2 ≥ DELTA_MARGIN), PAS de one-hot allclose.
  Un seul espace (cosinus partout, vecteurs unit-norm) — lève l'incohérence
  numérique pointée par le DA (decode cosinus vs anti-collapse euclidien).

Classe SÉPARÉE de SymbolicDict : les fichiers existants qui pincent le one-hot
(test_verifier.py, experiment_*.py) restent VERTS inchangés. Même surface
duck-typée consommée par reasoner.py / acsp.py / experiments :
    canonical(idx) -> Tensor(dim,)   # vecteur unit-norm de la primitive idx
    decode(vec)     -> (idx, valid)   # plus proche voisin cosinus + marge
    _matrix()       -> Tensor(V, dim) # lignes unit-norm (contrat l_align)
"""
import torch
import torch.nn as nn

from .amv import PART

TAU_PURE = 0.85       # decode : cosinus au plus proche voisin ≥ 0.85
DELTA_MARGIN = 0.05   # decode : écart cos1 − cos2 ≥ 0.05 (lève l'ambiguïté)


class LearnedVocab(nn.Module):
    """Dictionnaire dense E ∈ R^{V×64} de vecteurs unit-norm. V peut dépasser 64.

    Init 'ortho' (défaut, n ≤ dim) : sous-espace orthonormalisé via QR →
    cos inter-paires = 0 exact (même géométrie pairwise que le one-hot, simplement
    non aligné sur les axes — l'analogue dense le plus fidèle). Init 'random'
    (n > dim ou forcé) : vecteurs aléatoires normalisés, quasi-orthogonaux en
    dim=64. Geler E (.freeze()) en fait un codebook fixe = analogue direct du
    one-hot, ce qui isole la géométrie du problème de cible mobile.
    """

    def __init__(self, n: int, dim: int = PART, init: str = "ortho", seed: int = 0):
        super().__init__()
        assert n >= 1 and dim >= 1
        self.n = n
        self.dim = dim
        g = torch.Generator().manual_seed(seed)

        if init == "ortho" and n <= dim:
            # Q de la décomposition QR de (dim, n) → n colonnes orthonormales.
            M = torch.randn(dim, n, generator=g)
            Q, _ = torch.linalg.qr(M)
            E = Q.T.contiguous()              # (n, dim), lignes unit-norm orthogonales
        else:
            E = torch.randn(n, dim, generator=g)

        E = E / (E.norm(dim=1, keepdim=True) + 1e-8)   # unit-norm par ligne
        self.E = nn.Parameter(E.clone())

    # --- surface duck-typée (consommée par reasoner.py / acsp.py / experiments) ---

    def _matrix(self) -> torch.Tensor:
        """Lignes unit-norm (V, dim), différentiable. Contrat pour acsp.l_align."""
        return self.E / (self.E.norm(dim=1, keepdim=True) + 1e-8)

    def canonical(self, idx: int) -> torch.Tensor:
        """Vecteur unit-norm (dim,) de la primitive idx (zéro si hors range)."""
        if not (0 <= idx < self.n):
            return torch.zeros(self.dim)
        return self._matrix()[idx]

    def decode(self, vec: torch.Tensor):
        """(idx, valid). Plus proche voisin cosinus + marge de pureté.

        valid = True ssi cos1 ≥ TAU_PURE ET cos1 − cos2 ≥ DELTA_MARGIN.
        Le device de la table est aligné sur celui de vec (le block tourne sur
        GPU, la table peut rester sur CPU comme SymbolicDict).
        """
        head = vec[: self.dim].to(torch.float32)
        head_n = head / (head.norm() + 1e-8)
        M = self._matrix().to(head.device)
        cos = head_n @ M.T                          # (V,) similarités cosinus
        cos1, idx = torch.max(cos, dim=0)
        cos1 = float(cos1.item())
        idx = int(idx.item())
        if self.n >= 2:
            # 2e meilleur score pour le test de marge (masque le meilleur)
            masked = cos.clone()
            masked[idx] = -2.0
            cos2 = float(torch.max(masked).item())
        else:
            cos2 = -1.0
        valid = (cos1 >= TAU_PURE) and ((cos1 - cos2) >= DELTA_MARGIN)
        return idx, bool(valid)

    # --- anti-collapse (Leçon 5) ---

    def uniformity_loss(self) -> torch.Tensor:
        """Moyenne des cos² inter-paires. 0 si E isotrope, croît si alignement.

        À minimiser quand E est entraînable, à monitorer (≤ 0.5²=0.25 en cos²,
        i.e. cos moyen ≤ 0.5) quand E est gelé."""
        M = self._matrix()                          # (V, dim) unit-norm
        G = M @ M.T                                 # (V, V) cos inter-paires
        iu = torch.triu_indices(self.n, self.n, offset=1)
        pairs = G[iu[0], iu[1]]                     # cos(i,j), i < j
        return (pairs ** 2).mean()

    def mean_inter_pair_cos(self) -> float:
        """Diagnostic : cos moyen inter-paires (E sain ⇒ ≤ 0.5)."""
        with torch.no_grad():
            M = self._matrix()
            G = M @ M.T
            iu = torch.triu_indices(self.n, self.n, offset=1)
            return float(G[iu[0], iu[1]].mean().item())

    def freeze(self):
        """Gèle E (codebook fixe = analogue direct du one-hot). Renvoie self."""
        self.E.requires_grad_(False)
        return self

    def to(self, *args, **kwargs):
        """Surchargé pour garder self.n/self.dim (nn.Module.to gère E)."""
        super().to(*args, **kwargs)
        return self
