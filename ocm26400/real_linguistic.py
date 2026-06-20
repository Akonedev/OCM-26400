"""Alignement amodal sur VUES LINGUISTIQUES RÉELLES (OCM-26400, cahier des charges).

Contrairement à concept_amodal.py (vues SIMULÉES — encodeurs placeholder concept_id),
ICI les vues sont des MODALITÉS LINGUISTIQUES RÉELLES tirées de 1000 vrais mots
anglais (spxlm_v6/real_vocab_dataset.json) :

    texte        : les caractères du mot (vue orthographique)
    morphologie  : plural/past/gerund/third (vraie conjugaison)
    phonologie   : phoneme_pattern + vowels/consonants/syllables
    sémantique   : category + synonym (+ lexname)

Chaque modalité est un ENCODEUR RÉEL (feature-bag déterministe des vraies features +
projection apprise), aligné amodalement via InfoNCE (P1) — f_texte(C)~f_morpho(C)~
f_phono(C)~f_sém(C) pour chaque VRAI mot C. C'est la brique « capturer en une passe +
alignement amodal sur vraies modalités » du cahier des charges.

Honnête : ce sont de vraies modalités LINGUISTIQUES (texte/morpho/phono/sémantique),
pas audio/image/vidéo (qui nécessiteraient un encodeur signal réel — voir frontière).
Mais ce ne sont plus des stubs : chaque vue dérive des VRAIES features du mot.
"""
import os, json
import torch
import torch.nn as nn

from .infonce import multimodal_l_consist, TAU_DEFAULT
from .amv import PART

# chemin du dataset réel (1000 mots anglais annotés)
_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "spxlm_v6", "real_vocab_dataset.json"
)
MODALITIES = ["texte", "morphologie", "phonologie", "semantique"]


def load_real_words(path: str = None, limit: int = None):
    """Charge les vrais mots + leurs features multi-vues."""
    p = path or _DATASET
    data = json.load(open(p))
    if limit:
        data = data[:limit]
    return data


def _feature_bag(strings, dim: int = PART) -> torch.Tensor:
    """Sac de features déterministe : hache les caractères des vraies features -> dim-d."""
    v = torch.zeros(dim)
    for s in strings:
        for ch in str(s).lower():
            v[(ord(ch) * 167) % dim] += 1.0
    return v


def view_bag(word_entry, modality: str, dim: int = PART) -> torch.Tensor:
    """Vecteur feature-bag RÉEL d'une modalité pour un vrai mot."""
    if modality == "texte":
        return _feature_bag([word_entry.get("word", "")], dim)
    if modality == "morphologie":
        return _feature_bag([word_entry.get(k, "") for k in
                             ("plural", "past", "gerund", "third")], dim)
    if modality == "phonologie":
        return _feature_bag([word_entry.get("phoneme_pattern", ""),
                             str(word_entry.get("vowels", "")),
                             str(word_entry.get("consonants", "")),
                             str(word_entry.get("syllables", ""))], dim)
    if modality == "semantique":
        return _feature_bag([word_entry.get("category", ""),
                             word_entry.get("synonym", ""),
                             word_entry.get("lexname", "")], dim)
    raise ValueError(modality)


class RealViewEncoder(nn.Module):
    """Encodeur d'une modalité réelle : feature-bag (fixe) -> MLP appris (dim)."""
    def __init__(self, dim: int = PART, hidden: int = 128, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim),
        )
        for p in self.mlp.parameters():
            nn.init.normal_(p, std=0.1, generator=g) if p.dim() > 1 else nn.init.zeros_(p)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        return self.mlp(bag)


def _enc_device(encoders):
    first = next(iter(encoders.values()))
    return next(first.parameters()).device


def build_views(words, encoders, dim: int = PART):
    """Pour chaque mot, les K vues encodées. Retourne dict modalité -> (N, dim)."""
    dev = _enc_device(encoders)
    bags = {m: torch.stack([view_bag(w, m, dim) for w in words]).to(dev) for m in MODALITIES}
    return {m: encoders[m](bags[m]) for m in MODALITIES}


def amodal_real_loss(views, tau: float = TAU_DEFAULT):
    """Loss d'alignement amodal sur les vues réelles (InfoNCE symétrique multimodal)."""
    return multimodal_l_consist(list(views.values()), tau=tau)


def train_real_amodal(words, encoders, n_steps=600, lr=3e-3, batch=64,
                      dim: int = PART, tau: float = TAU_DEFAULT):
    """Entraîne les K encodeurs de modalités réelles à s'aligner amodalement."""
    dev = _enc_device(encoders)
    bags = {m: torch.stack([view_bag(w, m, dim) for w in words]).to(dev) for m in MODALITIES}
    params = [p for enc in encoders.values() for p in enc.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    N = len(words)
    for _ in range(n_steps):
        idx = torch.randint(0, N, (batch,))
        views = {m: encoders[m](bags[m][idx]) for m in MODALITIES}
        loss = amodal_real_loss(views, tau=tau)
        opt.zero_grad(); loss.backward(); opt.step()
    return encoders


@torch.no_grad()
def cross_view_retrieval_real(words, encoders, dim: int = PART):
    """Retrieval@1 cross-vue sur les VRAIS mots : la vue 'texte' du mot i retrouve-t-elle
    la vue 'morphologie' du MÊME mot i ? Moyenné sur les paires de modalités."""
    N = len(words)
    views = build_views(words, encoders, dim)
    dev = _enc_device(encoders)
    vn = {m: v / (v.norm(dim=-1, keepdim=True) + 1e-8) for m, v in views.items()}
    ar = torch.arange(N, device=dev)
    hits, n_pairs = 0, 0
    for a in range(len(MODALITIES)):
        for b in range(a + 1, len(MODALITIES)):
            ma, mb = MODALITIES[a], MODALITIES[b]
            sim = vn[ma] @ vn[mb].T
            pred = sim.argmax(dim=-1)
            hits += (pred == ar).sum().item()
            n_pairs += N
    return hits / n_pairs if n_pairs else 0.0
