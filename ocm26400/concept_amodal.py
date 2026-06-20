"""Alignement amodal multi-vues (OCM-26400, spec Besoins §A1.3 + cahier des charges).

Implémente l'alignement amodal du Mentalese : plusieurs « vues » (modalités) d'un
même concept C convergent vers un vecteur unique v_C :

    f_T(C) ~ f_A(C) ~ f_V(C) ~ v_C        (spec §A1.3)

C'est la brique « capturer en une passe + alignement amodal » du cahier des charges
(capturer simultanément grammaire/vocabulaire/phonèmes/etc. d'un concept).

CONSTRUCTION HONNÊTE (suite du verdict panel 19/06) :
* P1 a livré InfoNCE pur (infonce.py) — l'alignement contrastif.
* P2 a livré LearnedVocab (learned_vocab.py) — la table E cible.
* Le verdict a DEFERRED le terme d'an crage ||f_m(C) − E_C|| « à ajouter quand P2
  livrera la table E ». P2 est livré => on l'ajoute MAINTENANT. C'est CE terme qui
  rend l'alignement « AMV » (ancré dans la partition/dictionnaire) et pas du CLIP
  recopié (où l'espace appris pouvait vivre dans un sous-espace sans rapport).

Les « vues » sont des encodeurs de modalité SIMULÉS (Embedding concept_id -> R^64),
indépendants à l'init (désalignés), qui s'alignent par la loss. HONNÊTE : c'est un
HARNESS — les modalités sont des placeholders pour de futurs signaux réels
(texte/audio/image via v6 ou encodeurs externes). La math d'alignement + l'ancrage
sont réels et validés ; on ne prétend PAS avoir du vrai multimodal (objection DA).

Loss = L_consist (InfoNCE symétrique sur les paires de modalités, P1)
     + anchor_w * L_anchor (MSE de chaque vue vers la canonique LearnedVocab E_C)
"""
import torch
import torch.nn as nn

from .infonce import multimodal_l_consist, TAU_DEFAULT
from .learned_vocab import LearnedVocab
from .amv import PART


class ModalityEncoder(nn.Module):
    """Encodeur d'une modalité : concept_id -> vue (PART,). Indépendant à l'init.

    Placeholder honnête pour un futur encodeur réel (texte char-level, audio FFT,
    image). Ici : table d'embedding apprise. Les K encodeurs démarrent désalignés
    (init indépendante) puis s'alignent par L_consist + L_anchor.
    """
    def __init__(self, n_concepts: int, dim: int = PART, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Parameter(torch.randn(n_concepts, dim, generator=g) * 0.1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb[ids]                          # (N, dim)


def amodal_align_loss(views, vocab: LearnedVocab, ids: torch.Tensor,
                      tau: float = TAU_DEFAULT, anchor_w: float = 1.0):
    """Loss d'alignement amodal.

    Args:
        views: liste de K tenseurs (N, dim) — une vue par modalité (même ordre de
               concepts => views[k][i] est la vue k du concept ids[i]).
        vocab: LearnedVocab gelé — fournit la canonique E_C (ancrage AMV).
        ids:   (N,) indices de concepts.
    Returns:
        (total, {consist, anchor}) — total = L_consist + anchor_w * L_anchor.
    """
    # (1) alignement contrastif amodal : mêmes concepts rapprochés cross-modalités
    l_consist = multimodal_l_consist(views, tau=tau)
    # (2) ancrage AMV : chaque vue tire vers la canonique du concept dans E
    target = vocab._matrix()[ids]                     # (N, dim) canoniques unit-norm
    l_anchor = sum(((v - target) ** 2).mean() for v in views) / len(views)
    total = l_consist + anchor_w * l_anchor
    return total, {"consist": float(l_consist.detach()), "anchor": float(l_anchor.detach())}


def train_amodal(vocab: LearnedVocab, encoders, n_concepts: int, n_steps: int = 800,
                 lr: float = 3e-3, batch: int = 64, device: str = "cpu",
                 tau: float = TAU_DEFAULT, anchor_w: float = 1.0):
    """Entraîne les K encodeurs de modalité à s'aligner + s'ancrer."""
    params = [p for enc in encoders for p in enc.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(n_steps):
        ids = torch.randint(0, n_concepts, (batch,), device=device)
        views = [enc(ids) for enc in encoders]
        loss, _ = amodal_align_loss(views, vocab, ids, tau=tau, anchor_w=anchor_w)
        opt.zero_grad(); loss.backward(); opt.step()
    return encoders


@torch.no_grad()
def cross_view_retrieval(encoders, n_concepts: int, device: str = "cpu"):
    """Retrieval@1 cross-vue : la vue 0 du concept i retrouve-t-elle la vue 1 du
    MÊME concept i comme plus proche voisin (sur les N concepts) ? Moyenné sur les
    paires de modalités. 1.0 = alignement amodal parfait."""
    all_ids = torch.arange(n_concepts, device=device)
    views = [enc(all_ids) for enc in encoders]        # K x (N, dim)
    # normalisation L2
    vn = [v / (v.norm(dim=-1, keepdim=True) + 1e-8) for v in views]
    k = len(encoders)
    hits, n_pairs = 0, 0
    for a in range(k):
        for b in range(a + 1, k):
            sim = vn[a] @ vn[b].T                     # (N, N) similarité cross-vue
            pred = sim.argmax(dim=-1)                 # (N,) concept prédit depuis vue a
            hits += (pred == all_ids).sum().item()
            n_pairs += n_concepts
    return hits / n_pairs if n_pairs else 0.0


@torch.no_grad()
def anchor_decode_accuracy(encoders, vocab: LearnedVocab, n_concepts: int,
                           device: str = "cpu"):
    """Précision du décodage ancré : decode(vue_k(C)) récupère-t-il C ?
    Mesure que l'ancrage AMV ramène chaque vue dans le dictionnaire."""
    all_ids = torch.arange(n_concepts, device=device)
    correct = 0
    total = 0
    for enc in encoders:
        views = enc(all_ids)                          # (N, dim)
        for i in range(n_concepts):
            idx, valid = vocab.decode(views[i])
            correct += (idx == int(all_ids[i]))
            total += 1
    return correct / total if total else 0.0
