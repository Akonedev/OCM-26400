# PROCEDURES — OCM-26400

**Manuel de procédures de création, pre-training, training, tuning/fine-tuning**
**Package : `ocm26400/` — Dépôt : `MathsBase/`**

> Version : 1.0 — 20 Juin 2026
> Objet : permettre à tout opérateur de **reproduire à l'identique** l'architecture, le pré-entraînement (grokking), l'entraînement (ACSP + curriculum + anti-shortcut) et le tuning (multi-règles conjointes, flow-matching, alignement amodal) sans commettre les erreurs déjà rencontrées et corrigées.
>
> Source de vérité : le code source `ocm26400/*.py` (lignes citées), les verdicts validés `ocm26400/STATUS.md`, `ocm26400/EXPERT_PANEL_VERDICT.md`, `TESTING.md`, `CODE_EXACT_SPECTRALM.md`, `AUDIT_VERIFIE_v6.md`. Les valeurs numériques sont **exactes** (vérifiées dans le code), pas approximatives.

---

## SOMMAIRE

0. [Convention de reproductibilité (préambule obligatoire)](#0-convention-de-reproductibilité-préambule-obligatoire)
1. [PROCÉDURE DE CRÉATION (architecture)](#1-procédure-de-création-architecture)
2. [PROCÉDURE DE PRE-TRAINING (grokking des primitives)](#2-procédure-de-pre-training-grokking-des-primitives)
3. [PROCÉDURE DE TRAINING (ACSP, curriculum, anti-shortcut)](#3-procédure-de-training-acsp-curriculum-anti-shortcut)
4. [PROCÉDURE DE TUNING / FINE-TUNING](#4-procédure-de-tuning--fine-tuning)
5. [ERREURS CONNUES ET COMMENT LES ÉVITER](#5-erreurs-connues-et-comment-les-éviter)
6. [CONFIGURATION COMPLÈTE (référence)](#6-configuration-complète-référence)
7. [CHECKLIST DE REPRODUCTION (à cocher)](#7-checklist-de-reproduction-à-cocher)

---

## 0. Convention de reproductibilité (préambule obligatoire)

Toutes les procédures ci-dessous supposent **simultanément** les conventions suivantes. Sans elles, la reproductibilité n'est PAS garantie.

### 0.1 Environnement

```bash
cd /media/akone/Dev/Dev_D/01_Dev_1/IA_Dev_D/12_Hermes/Recherches/MathsBase
# Python via uv (cf. pyproject.toml / uv.lock). Version recommandée : voir .python-version.
# Torch installé avec backend CUDA (NVIDIA) ou ROCm (AMD). Vérifier :
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### 0.2 Seeds (OBLIGATOIRES — toute dérive vient de là)

```python
import random, torch
random.seed(0)
torch.manual_seed(0)
# CUDA / ROCm déterminisme (au prix d'un léger surcoût) :
torch.use_deterministic_algorithms(True)   # optionnel, mais fixe les FFT plans
# Note : le curriculum ré-itle `torch.manual_seed(0)` à chaque palier — ne PAS changer.
```

Tous les trainers du paquet font `torch.manual_seed(0)` (voir `reasoner.py:85`, `diff_decode.py:66`, `omni_rules.py`, `experiment_composition.py:31`). Le curriculum utilise aussi `random.seed(42)` pour le split train/test (`curriculum.py`). **Ne pas modifier ces seeds sans renormaliser les seuils.**

### 0.3 Device

```python
# ocm26400/reasoner.py:19
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Sur AMD ROCm, `torch.cuda.is_available()` renvoie `True` (backend HIP). Aucun code spécial n'est requis côté Python, **sauf** la fixation de `seq_len` (voir §5.6 sur le cache de plan FFT).

### 0.4 Optimizer canonique (OCM-26400)

**Toujours** `torch.optim.Adam(lr=3e-3)` pour les trainers `ocm26400` (math, composition, conjugation, sleep, self-improve, curriculum, omni_rules, refinement, ACSP). C'est `Adam` (et non `AdamW`) — ce dernier est réservé au modèle spectral `spxlm_v6` (voir §6.2). Ne pas confondre les deux recettes.

### 0.5 Règle d'or (lue dans EXPERT_PANEL_VERDICT.md)

> **"À chaque étape, la pièce précédente fournit le contrat, la suivante le consomme. Pas de boucle, pas de collage : c'est un DAG."**

L'ordre d'intégration canonique (juge) est **P1 InfoNCE pur + fix seam ACSP → P2 LearnedVocab + compose op_id (pierre angulaire) → P3 LSRA gate calibrée (abstention) → P4 pont v6→AMV (NO-GO tant que P1-P3 ne sont pas verts)**. Ne jamais intégrer une pièce avant que la précédente ne fournisse son contrat.

---

## 1. PROCÉDURE DE CRÉATION (architecture)

Cette section couvre la construction du noyau unifié (SpectralCoreBlock), du vecteur amodal (AMV-256 et ses 4 partitions), du dictionnaire symbolique, du vérifieur déterministe, du bloc de raisonnement et des décodeurs. Tout est défini dans `ocm26400/*.py`.

### 1.1 Étape 1 — Définir les constantes globales

Vérifier d'abord que ces valeurs sont bien celles du code (elles sont la « signature » du projet) :

```python
# ocm26400/amv.py:11-12
D_MODEL = 256            # dimension totale de l'AMV
PART    = 64             # taille d'UNE partition (4 partitions x 64 = 256)

# ocm26400/verifier.py:16-19
P_MOD       = 11         # base modulaire (Z_11) — tâche compositionnelle non-assoc
A_COEF      = 3          # op(a,b) = (3*a + 5*b) mod 11
B_COEF      = 5
P_BACKTRACK = 1000.0     # pénalité massive d'étape illégale (spec §2.2)

# ocm26400/reasoner.py:18-19
TAU_GROK = 0.9           # seuil de confiance pour stopper la récurrence LSRA
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
```

**Pourquoi ces valeurs :**
- `P_MOD = 11` est premier, petit, et rend `op(a,b) = (3a+5b) mod 11` **non-commutative** et **non-associative**. La non-associativité est cruciale : elle garantit que l'intermédiaire `m = op(a,b)` est **structurellement nécessaire** pour calculer `r = op(m,c)` — c'est ce qui rend le crown-jewel (décomposition >> one-shot) non trivial.
- `P_BACKTRACK = 1000.0` est choisi `>> 1` pour que toute étape illégale **domine** la loss (cf. `test_acsp.py:test_acsp_dominated_by_step_penalty_when_illegal`).
- `TAU_GROK = 0.9` : la gate LSRA s'arrête quand `sigmoid(meta[0]) >= 0.9`. Comme on pousse `meta[0]` vers `CONF_TARGET = 4.0` (`sigmoid(4) ≈ 0.98 > 0.9`), un bloc bien entraîné s'arrête en 1 pas sur les données connues.

### 1.2 Étape 2 — Construire le vecteur AMV-256 et ses 4 partitions

Le vecteur amodal mentalese est un tenseur `R^256` à **partitionnement dur** contigu (QPLS). Fichier : `ocm26400/amv.py`.

```python
import torch

D_MODEL = 256
PART    = 64

class AMVVector:
    """v = [ v_ent(64) || v_prop(64) || v_op(64) || v_meta(64) ] ∈ R^256."""
    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        assert tensor.shape[-1] == D_MODEL, f"AMV doit faire {D_MODEL} dims"
        self.tensor = tensor

    @classmethod
    def zeros(cls, device=None):  # torch.zeros(256)
        return cls(torch.zeros(D_MODEL, device=device))

    @classmethod
    def randn(cls, device=None):
        return cls(torch.randn(D_MODEL, device=device))

    # Vues (slices) — écritures in-place propagées, gradients coulent
    @property
    def ent(self):   return self.tensor[0:64]      # entity / racine
    @property
    def prop(self):  return self.tensor[64:128]    # property / modifieur
    @property
    def op(self):    return self.tensor[128:192]   # operator / op_id
    @property
    def meta(self):  return self.tensor[192:256]   # metadata (confidence, etc.)
```

#### Layout de la partition `meta(64)` — DÉCISION JUGE 19/06 (OBLIGATOIRE)

Pour résoudre un conflit de contention détecté par le panel expert, la partition `meta` est subdivisée en **3 rôles**. Ne pas réutiliser ces indices pour autre chose :

| Index | Rôle | Pièce consommatrice |
|---|---|---|
| `meta[0]` (192) | Confidence LSRA `c = sigmoid(meta[0])` | `reasoner.py:123` (`lsra_loop`), `train_reasoner_with_confidence` pousse à `CONF_TARGET = 4.0` |
| `meta[1]` (193) | Confidence source / bridge | Pièce 3 (concept_amodal) — ne pas écraser par `CONF_TARGET` |
| `meta[2]` (194) | Score de consistance cross-modale brute | Pièce 1 (InfoNCE / `multimodal_l_consist`) |

> **Erreur courante (§5.3) :** si une seule de ces pièces écrit dans `meta[0]`, la gate LSRA se déclenche au hasard. Ne JAMAIS partager `meta[0]` entre deux pertes.

### 1.3 Étape 3 — Construire le dictionnaire symbolique et le vérifieur

Fichiers : `ocm26400/verifier.py`.

```python
class SymbolicDict:
    """Chaque primitive = vecteur canonique one-hot dans les P_MOD premières dims
    du slot ent (64). Décodage = argmax + test de pureté (one-hot exact)."""
    def __init__(self, n: int = P_MOD, dim: int = PART):
        assert n <= dim               # sinon overflow du slot ent
        self.n, self.dim = n, dim

    def canonical(self, idx: int) -> torch.Tensor:
        v = torch.zeros(self.dim)
        if 0 <= idx < self.n: v[idx] = 1.0
        return v

    def _matrix(self) -> torch.Tensor:   # (n, 64) pour calcul cosinus vectorisé
        return torch.stack([self.canonical(i) for i in range(self.n)])

    def decode(self, vec):
        head = vec[: self.n]
        idx  = int(torch.argmax(head).item())
        expected = torch.zeros(self.n, device=head.device, dtype=head.dtype)
        expected[idx] = 1.0
        valid = bool(torch.allclose(head, expected, atol=1e-3))
        return idx, valid


class Verifier:
    """Vérifieur DÉTERMINISTE. Connaît la table d'opération. compose_fn pluggable."""
    def __init__(self, dictionary, compose_fn=None, n_ops: int = 1):
        self.dict = dictionary
        self.n_ops = n_ops
        self._compose_fn = compose_fn     # None -> arithmétique Z_P_MOD

    def compose(self, a, b, op_id: int = 0) -> int:
        # op_id TOUJOURS présent (rétrocompatible, permet dispatch multi-règles)
        if self._compose_fn is not None: return self._compose_fn(a, b)
        return (A_COEF * a + B_COEF * b) % P_MOD

    def V(self, d_ent, d_prop, op_id=0) -> bool:
        # légal ssi ent & prop dans le dictionnaire ET op connu
        return (0 <= d_ent < self.dict.n
                and 0 <= d_prop < self.dict.n
                and 0 <= op_id < self.n_ops)

    def is_valid_intermediate(self, a, b, m, op_id=0) -> bool:
        return m == self.compose(a, b, op_id=op_id)
```

**Points de contrat (à ne JAMAIS casser) :**
- `compose(a, b, op_id=0)` — signature **unifiée** avec `op_id` (correction juge §5.4). L'ancienne signature `compose(a,b)` est rétrocompatible grâce au défaut `op_id=0`.
- `SymbolicDict.decode` utilise `atol=1e-3` pour le test de pureté one-hot. Tout `LearnedVocab` (densité) qui surcharge `decode` doit **préserver ce contrat** (cf. §5.8 sur le risque Frankenstein).

### 1.4 Étape 4 — Construire le ReasonerBlock (noyau latent)

Fichier : `ocm26400/reasoner.py:22-43`. C'est un **MLP résiduel** sur `R^256`. Il apprend à transformer un AMV `(ent=a, prop=b)` en un AMV dont `ent = op(a,b)`.

```python
class ReasonerBlock(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, hidden)   # 256 -> 512
        self.fc2  = nn.Linear(hidden, d_model)   # 512 -> 256
        # INIT OBLIGATOIRE (std=0.02, biases zéros) — cf. §5.7 LazyLinear
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):                        # (B, 256) ou (256,)
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return x + h                              # résiduel
```

**Paramètres :** ~263K params par bloc (cité dans `experiment_composition.py:102`). C'est la référence "params constants" du projet.

**Warnings archi (§5.9, §5.10) :**
- Le résiduel `x + h` ne fait **PAS** de ce bloc une contraction de Banach (rien ne garantit `ρ(I+J_f) < 1`). L'appeler "opérateur de relaxation" est une **aspiration**, pas une propriété démontrée.
- Itérer `v_{t+1} = blk(v_t)` ne raffine PAS la prédiction courante : la prop reste `b`, donc à l'étape suivante le bloc calcule `op(op(a,b), b)`. C'est une **recomposition**, pas un raffinement. Le P3 originel (supervision de trajectoire géométrique) était tautologique et a été **enterré** (`experiment_refinement.py:1-33`).

### 1.5 Étape 5 — Construire le SpectralCoreBlock (architecture du projet, noyau unifié)

Fichier : `ocm26400/spectral_core.py`. C'est **L'ARCHITECTURE du projet** (issue de spXLM v6) — on ne change pas d'architecture, on porte le noyau spectral dans `ocm26400` pour unifier raisonnement / classification / génération sous le **même** block.

```python
import math, torch, torch.nn as nn

class SpectralCoreBlock(nn.Module):
    """Noyau spectral FFT bidirectionnel de l'utilisateur. Noyau unifié.
    Accepte (B, d) [un AMV, L=1] ou (B, L, d) [une chaîne compositionnelle]."""

    def __init__(self, d_model: int = D_MODEL, seq_len: int = 64, bidirectional: bool = True):
        super().__init__()
        self.d_model, self.bidirectional, self.seq_len = d_model, bidirectional, seq_len

        self.in_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Filtre fréquentiel complexe APPRIS, par dimension
        scale = 1.0 / math.sqrt(d_model)
        self.filter_real = nn.Parameter(torch.randn(seq_len // 2 + 1, d_model) * scale + 1.0)
        self.filter_imag = nn.Parameter(torch.randn(seq_len // 2 + 1, d_model) * scale)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):                       # (B,d) -> (B,d) ; (B,L,d) -> (B,L,d)
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze = True   # (B, 1, d)
        B, L, D = x.shape

        h = self.norm1(x)
        h = self.in_proj(h)
        X_freq = torch.fft.rfft(h, dim=1)        # (B, F, D), F = L//2 + 1

        # Multiplication complexe par filtre appris (slicing sur F réel)
        fr = self.filter_real[:X_freq.shape[1], :].unsqueeze(0)
        fi = self.filter_imag[:X_freq.shape[1], :].unsqueeze(0)
        X_real = X_freq.real * fr - X_freq.imag * fi
        X_imag = X_freq.real * fi + X_freq.imag * fr
        X_filtered = torch.complex(X_real, X_imag)

        y = torch.fft.irfft(X_filtered, n=L, dim=1)
        y = self.out_proj(y)

        x = x + y                                 # résiduel spectral
        x = x + self.ffn(self.norm2(x))           # FFN
        return x.squeeze(1) if squeeze else x
```

**Propriétés clés (à respecter) :**
- Complexité `O(L log L)` — **pas d'attention** `O(L²)`, pas de KV cache.
- **Stabilité de Parseval** : `‖x‖² = ‖FFT(x)‖²`. Les filtres appris préservent la norme si leur module reste borné (le `+1.0` sur la partie réelle garantit un filtre proche de l'identité à l'init).
- Init : partie réelle biaisée à `+1.0`, partie imaginaire centrée, amplitude `1/sqrt(d_model)` — pour qu'à l'init le filtre soit ≈ identité (signal in == signal out) et que la dérive spectrale soit progressive.
- `seq_len` est fixé à la construction (défaut `64`). En inférence, **padder à `seq_len` fixe** pour éviter la recompilation du plan FFT sous ROCm (§5.6).

### 1.6 Étape 6 — Câbler le tout : encode_input, LSRA loop

Fichier : `ocm26400/reasoner.py:46-128`.

```python
def encode_input(a, b, dictionary):
    v = torch.zeros(D_MODEL)
    v[0:64]   = dictionary.canonical(a)         # ent
    v[64:128] = dictionary.canonical(b)         # prop
    return v                                     # op (128:192) et meta (192:256) = 0

def lsra_loop(blk, dictionary, x0, max_iter=8, tau=TAU_GROK):
    """v(t+1) = Block(v(t)). Stop à T* = min{t | sigmoid(meta[0]) >= tau}.
    Sinon -> [ANOMALIE_CAUSALE] (confident=False)."""
    device = next(blk.parameters()).device
    v = x0.to(device).clone()
    blk.eval()
    with torch.no_grad():
        for t in range(max_iter):
            v = blk(v.unsqueeze(0))[0]
            conf = torch.sigmoid(v[192]).item()
            if conf >= tau:
                idx, _ = dictionary.decode(v[0:64])
                return idx, t + 1, True          # stop anticipé, confiant
        idx, _ = dictionary.decode(v[0:64])
        return idx, max_iter, False              # ANOMALIE
```

### 1.7 Étape 7 — Construire le décodeur flow-matching (génération amodale)

Fichier : `ocm26400/generators.py`. À utiliser pour la génération (cf. §4.2). Remplace les anciens `audio_dec`/`image_dec` (MSE feature regression) par un vrai décodeur neural conditionné par l'AMV (Lipman 2023 / Esser SD3).

```python
class AMVConditionedDecoder(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int = D_MODEL, hidden: int = 128):
        super().__init__()
        self.x_dim = x_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),             nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def _velocity(self, x_t, t, cond):
        t = t.view(-1, 1)
        return self.net(torch.cat([x_t, t, cond], dim=-1))

    def flow_match_loss(self, cond, x_target):
        noise = torch.randn_like(x_target)
        t     = torch.rand(x_target.shape[0])
        x_t   = (1 - t).view(-1, 1) * noise + t.view(-1, 1) * x_target
        target_vel = x_target - noise
        return ((self._velocity(x_t, t, cond) - target_vel) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond, steps: int = 8):
        x  = torch.randn(cond.shape[0], self.x_dim, device=cond.device)
        dt = 1.0 / steps
        for i in range(steps):
            t  = torch.full((cond.shape[0],), i * dt, device=cond.device)
            x  = x + self._velocity(x, t, cond) * dt   # Euler
        return x
```

---

## 2. PROCÉDURE DE PRE-TRAINING (grokking des primitives)

**Objectif :** qu'un seul `ReasonerBlock` apprenne la primitive `op(a,b)` sur **toutes** les `P_MOD² = 121` paires et généralise à 100% sur les paires **non vues** en boucle ouverte. C'est le "block binaire grokké" réutilisé par 7 expériences (composition, recursion, sleep, self_improve, omni_generate, curriculum, refinement).

### 2.1 Initialisation des poids (RÈGLE — std=0.02)

```python
# Déjà fait dans ReasonerBlock.__init__ (cf §1.4) — RAPPEL :
nn.init.normal_(self.fc1.weight, std=0.02)
nn.init.normal_(self.fc2.weight, std=0.02)
nn.init.zeros_(self.fc1.bias)
nn.init.zeros_(self.fc2.bias)
```

**Pourquoi std=0.02 :**
- Trop grand (défaut PyTorch `~1/sqrt(256) ≈ 0.0625`) : le résiduel `x + h` est dominé par `h` au démarrage → divergence.
- Trop petit : convergence trop lente pour grok en 1500 pas.
- 0.02 est le même choix que GPT-2/BERT pour les projections linéaires, validé empiriquement ici.

Pour le `SpectralCoreBlock` : init "identité spectrale" (filtre réel `+1.0`, imaginaire `0`, amplitude `1/sqrt(d_model)`) — voir §1.5. **Ne PAS** appliquer `std=0.02` aux filtres spectraux : cela casserait la stabilité de Parseval.

### 2.2 Procédure — `train_binary_block` (LA procédure canonique)

Fichier : `ocm26400/experiment_composition.py:30-43` (procédure de référence réutilisée partout).

```python
def train_binary_block(dictionary, verifier, n_steps=1500, lr=3e-3, batch=64):
    torch.manual_seed(0)                                 # §0.2
    blk = ReasonerBlock().to(DEVICE)                     # init std=0.02 interne
    opt = torch.optim.Adam(blk.parameters(), lr=lr)      # Adam, PAS AdamW

    pairs = [(a, b) for a in range(P_MOD) for b in range(P_MOD)]   # 121 paires

    for step in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))     # tirage WITH replacement
        batch_in = torch.stack(
            [encode_input(pairs[i][0], pairs[i][1], dictionary) for i in idx]
        ).to(DEVICE)
        out = blk(batch_in)

        # LOSS : alignement cosinus, coefficient 1.0, moyenné sur le batch
        loss = torch.tensor(0.0, device=DEVICE)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            target = dictionary.canonical(verifier.compose(a, b)).to(DEVICE)
            ent    = out[j][0:64]
            cos    = (ent @ target) / (ent.norm() * target.norm() + 1e-8)
            loss   = loss + (1.0 - cos)
        loss = loss / batch

        opt.zero_grad(); loss.backward(); opt.step()
    return blk
```

**Hyperparamètres EXACTS (à reproduire à l'identique) :**

| Paramètre | Valeur | Source |
|---|---|---|
| `lr` | `3e-3` | `experiment_composition.py:30` |
| `n_steps` | `1500` (défaut) ; `2000` pour multi-op (cf `experiment_math.py`) | idem |
| `batch` | `64` (sauf refinement `128`, omni_rules `128`) | idem |
| `optimizer` | `Adam` (pas AdamW) | idem |
| `seed` | `0` (`torch.manual_seed(0)`) | idem |
| `device` | `DEVICE` (cuda si dispo) | idem |
| `loss` | `(1 - cos)` sur `ent[0:64]` vs `canonical(op(a,b))`, moyenné sur batch | idem |
| `epsilon norm` | `1e-8` dans le cosinus | idem |
| `tirage` | `torch.randint(0, 121, (batch,))` WITH replacement | idem |

**Variante avec gate calibrée (apprend en plus la confidence)** — `ocm26400/reasoner.py:81-107` (`train_reasoner_with_confidence`) :

```python
def train_reasoner_with_confidence(dictionary, verifier, n_steps=800, lr=3e-3, batch=64, device=DEVICE):
    torch.manual_seed(0)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = [(a, b) for a in range(dictionary.n) for b in range(dictionary.n)]
    CONF_TARGET = 4.0                              # sigmoid(4) ≈ 0.98 > TAU_GROK

    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], dictionary) for i in idx]).to(device)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=device)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            m   = verifier.compose(a, b)
            ent = out[j][0:64]
            d   = dictionary.canonical(m).to(device)
            cos = (ent @ d) / (ent.norm() * d.norm() + 1e-8)
            loss = loss + (1.0 - cos)                              # alignement ent
            loss = loss + (out[j][192] - CONF_TARGET) ** 2          # confidence meta[0]
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk
```

**Différences clés :**
- `n_steps = 800` (plus court : la gate est plus facile à apprendre que la primitive pure).
- Loss = `(1 - cos)` **+** `(meta[0] - 4.0)²` — on pousse `sigmoid(meta[0])` vers `0.98` pour que la gate s'arrête en 1 pas.

### 2.3 Comment VÉRIFIER que le grokking a eu lieu

Le "grokking" est **opérationnel** (pas un concept séparé avec une phase de mémorisation explicite) : on entraîne 1500-2000 pas, puis on teste la généralisation sur des paires invisibles. Pour le confirmer, exécuter :

```python
# Option A : évaluation binaire brute (diff_decode.py:85-97)
from ocm26400.diff_decode import eval_binary
acc = eval_binary(blk, d, ver, n_test=121)
# Sur 121 paires : attendre 1.0 (100%) après 1500 pas

# Option B : évaluation via le crown-jewel (experiment_composition.py)
# - Binary block grokké : ~100%
# - ONE-SHOT sur triples jamais vus : ~0.5% (pure mémorisation)
# - DÉCOMP LSRA (2 pas) : ~100%  -> c'est ÇA qui prouve le grokking
# Gap décomposition - one-shot : +99.5 points (validé dans STATUS.md)
```

**Critère de grokking accepté :**
- `acc_binaire ≥ 0.99` sur 121 paires.
- ET `décomposition ≥ 0.95` sur 100-200 triples **jamais vus** (preuve que la primitive est généralisable, pas mémorisée).

**Si `acc_binaire < 0.95` après 1500 pas :** vérifier dans l'ordre — (1) seed, (2) init `std=0.02`, (3) optimizer Adam `lr=3e-3` (pas AdamW), (4) device, (5) epsilon `1e-8` dans le cosinus.

### 2.4 Variante multi-op (un seul bloc, plusieurs opérations)

Fichier : `ocm26400/experiment_math.py`. Le `op_id` one-hot dans le slot `op` (128:192) sélectionne l'opération.

```python
# 3 opérations sur Z_11 :
OPS = [(1, 1), (3, 5), (2, 7)]    # ADD, OP_A, OP_B
def result(a, b, op_id): return (OPS[op_id][0] * a + OPS[op_id][1] * b) % 11

def encode_math(a, b, op_id, d):
    v = torch.zeros(256)
    v[0:64]   = d.canonical(a)
    v[64:128] = d.canonical(b)
    v[128 + op_id] = 1.0                       # one-hot dans le slot op
    return v

# Boucle identique, n_steps = 2000 (multi-op demande +500 pas), batch = 64
# Loss = (1 - cos) comme en §2.2, target = canonical(result(a, b, k))
```

**Hyperparamètres multi-op :** `n_steps = 2000` (+500 vs mono-op), `batch = 64`, `lr = 3e-3`, `seed = 0`. Validation : précision par op > 0.9 sur 100 couples non vus ET règle `(α,β)` extraite correctement via `sleep.extract_rule` (`experiment_math.py`).

---

## 3. PROCÉDURE DE TRAINING (ACSP, curriculum, anti-shortcut)

Cette section couvre le training "réel" : la loss ACSP (différentiable via Gumbel straight-through), le curriculum progressif, la surveillance de la convergence et l'anti-shortcut.

### 3.1 La loss ACSP — Causal Rigor Loss

Fichier : `ocm26400/acsp.py`. Formule :

```
L = α·L_align + β·L_step + γ·L_sparse + δ·L_consist
```

Poids par défaut :
```python
ALPHA = 1.0     # alignement au dictionnaire       (acsp.py:15)
BETA  = 1.0     # pénalité d'étape illégale        (acsp.py:16)
GAMMA = 1e-3    # sparsité (régularisation faible) (acsp.py:17)
DELTA = 0.0     # consistence multimodale (0 en single-modality) (acsp.py:18)
```

**Définition des termes (formules exactes) :**

- **`L_align = min_{d ∈ D} (1 - cos(v.ent, d))`** — pénalise un vecteur entité qui ne correspond à aucune primitive du dictionnaire. Vaut `0` si `v.ent` est exactement une primitive canonique.
- **`L_step = 0 si V(ent, prop, op) légale, sinon P_BACKTRACK (= 1000)`** — pénalité massive. **ATTENTION : version de base NON différentiable** (`torch.tensor(0.0 if legal else P_BACKTRACK)` ne porte pas de gradient). C'est la "version décorative" dénoncée par le verdict expert.
- **`L_sparse = lam · Σ|v_i|`** sur les 256 dims (L1). Empêche la mémorisation de bruit. `lam=1.0` par défaut dans la fonction ; le poids effectif est `γ=1e-3` dans `acsp_loss`.
- **`L_consist = info_nce(z_a, z_b, τ=0.07)`** — InfoNCE cross-modal (cf. §4.3). Single-modality → 0.

**Sanity check (test RED obligatoire, `test_acsp.py:56-64`) :** sur un batch "parfait" (`v.ent = primitive canonique + étape légale`), la loss doit descendre **sous 0.05**. Si ce n'est pas le cas, l'implémentation est cassée.

### 3.2 Rendre `L_step` DIFFÉRENTIABLE — Gumbel straight-through (P-B, obligatoire)

La version de base ne câble pas de gradient vers le noyau. Le verdict expert impose d'utiliser la version différentiable (`ocm26400/diff_decode.py`).

```python
def decode_gumbel(logits_n, tau=1.0, hard=True):
    """logits (n,) -> quasi-one-hot (n,) avec gradient.
    hard=True : forward = one_hot(argmax), backward = softmax (straight-through)."""
    g = -torch.log(-torch.log(torch.rand_like(logits_n) + 1e-20) + 1e-20)
    y = F.softmax((logits_n + g) / tau, dim=-1)
    if hard:
        idx = y.argmax(dim=-1)
        y_hard = F.one_hot(idx, num_classes=y.shape[-1]).float()
        return y_hard - y.detach() + y          # straight-through (Jang 2016 / Bengio 2013)
    return y

def l_step_diff(v, verifier, a, b, op_id=0, tau=1.0):
    """Pénalité différentiable : écart entre le symbole décodé soft et le légal compose(a,b)."""
    n       = verifier.dict.n
    soft    = decode_gumbel(v.ent[:n], tau=tau, hard=True)        # (n,) avec gradient
    correct = verifier.compose(a, b, op_id=op_id)
    return P_BACKTRACK * (1.0 - soft[correct])                    # gradient réel vers le noyau

def acsp_loss_diff(v, dictionary, verifier, a, b, op_id=0, tau=1.0):
    """ACSP end-to-end différentiable."""
    return (ALPHA * l_align(v, dictionary)
          + BETA  * l_step_diff(v, verifier, a, b, op_id, tau)
          + GAMMA * l_sparse(v))
```

**Référence techno :** Jang et al. 2016 (Categorical Reparameterization with Gumbel-Softmax), Bengio et al. 2013 (Estimating or Propagating Gradients for Stochastic Neurons). Straight-through estimator : forward = `argmax` (dur), backward = `softmax` (différentiable).

### 3.3 Le trainer ACSP réel — `train_with_acsp`

Fichier : `ocm26400/diff_decode.py:62-82`. **C'est ce qui rend ACSP vivant dans l'entraînement** (avant cela, ACSP était décoratif : `reasoner.py` et `omni_rules.py` utilisaient du `(1-cos)` ad-hoc).

```python
def train_with_acsp(dictionary, verifier, n_steps=1500, lr=3e-3, batch=64, device=DEVICE):
    torch.manual_seed(0)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    pairs = [(a, b) for a in range(dictionary.n) for b in range(dictionary.n)]
    for _ in range(n_steps):
        idx = torch.randint(0, len(pairs), (batch,))
        batch_in = torch.stack([encode_input(pairs[i][0], pairs[i][1], dictionary) for i in idx]).to(device)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=device)
        for j, i in enumerate(idx):
            a, b = pairs[i]
            v = AMVVector(out[j])
            loss = loss + acsp_loss_diff(v, dictionary, verifier, a, b)    # ACSP différentiable
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk
```

**Hyperparamètres identiques à `train_binary_block`** (`n_steps=1500, lr=3e-3, batch=64, Adam, seed=0`). Ce trainer ajoute le signal de **légalité symbolique** par-dessus l'alignement — améliore empiriquement le grokking.

### 3.4 Le curriculum progressif (4 phases)

Fichier : `ocm26400/curriculum.py`. Principe : on ne passe à la phase suivante que si la phase courante passe les critères (accuracy + anti-shortcut).

```python
from dataclasses import dataclass

@dataclass
class PhaseResult:
    phase: str
    accuracy: float
    train_test_gap: float
    passed: bool
    steps: int

@dataclass
class Curriculum:
    n: int = P_MOD                               # = 11
    accuracy_threshold: float = 0.85
    max_shortcut_gap: float = 0.15               # anti-shortcut : gap train/test > δ -> bloqué
```

**Phases (ordre strict, `curriculum.py:phases`) :**

```
1. primitives     — op(a,b) sur paires isolées
2. paires         — m = op(a,b) comme intermédiaire
3. chaînes        — r = op(m, c) (composition 2-pas)
4. inter-règles   — mélange de règles (omni_rules, cf §4.1)
```

**Critère de passage (`curriculum.py:evaluate_phase`) :**

```python
# Split : random.seed(42) ; shuffle des all_pairs ; n_test pour train, n_test pour test
passed = (test_acc >= accuracy_threshold) AND (gap <= max_shortcut_gap)
# gap = train_acc - test_acc
```

**Attention :** les valeurs de seuil varient entre la classe (`accuracy_threshold=0.85, max_shortcut_gap=0.15`) et l'expérience qui l'utilise (`experiment_curriculum.py` fixe `accuracy_threshold=0.80, max_shortcut_gap=0.20`). Pour reproduire l'expérience curriculum validée, **utiliser les seuils de l'expérience** (0.80 / 0.20). Pour une évaluation plus stricte, utiliser les seuils de la classe (0.85 / 0.15).

**Paliers de steps recommandés (`experiment_curriculum.py`) :** `n_steps ∈ [500, 1000, 1500]` — on ré-entraîne `train_binary_block` à chaque palier et on ré-évalue. L'accuracy doit grimper avec le nombre de pas.

### 3.5 Anti-shortcut (gap train/test < δ) — RÈGLES CRITIQUES

L'anti-shortcut n'est **pas** une option. C'est, avec le scratchpad, ce qui rend le grokking possible (TESTING.md E12 vs E13 : 0% → 100%).

#### 3.5.1 Principe (CODE_EXACT_SPECTRALM.md §6.1)

> **Pour tout champ fᵢ tel que fᵢ = f(fⱼ, fₖ...) : masquer TOUTES les variables Vⱼ, Vₖ en input qui sont algébriquement récupérables depuis fᵢ.**

Le gradient choisit toujours le chemin le plus court. Si `ans` est visible pendant qu'on entraîne `m1`, le modèle apprend `m1 = ans - c` (soustraction triviale) au lieu de `m1 = a × b` (vraie multiplication depuis la prose).

#### 3.5.2 Anti-shortcut ASYMÉTRIQUE (loi L2, TESTING.md insight 6)

**Masquer un sous-ensemble stratégique, PAS TOUT.** E12 (18/18 cas masqués) → **0%**. E13 (masque asymétrique + format structuré) → **100%**.

Pour OCM-26400, la traduction est directe : pendant qu'on entraîne le block à produire `m = op(a,b)`, **ne pas fournir `op(a,b)` dans une autre partition**. Le `op` slot (128:192) doit contenir l'**op_id** (sélecteur), jamais le résultat.

#### 3.5.3 Surveillance du gap train/test

```python
# À chaque phase :
gap = train_acc - test_acc
if gap > max_shortcut_gap:   # 0.15 (classe) ou 0.20 (expérimental)
    # NE PAS passer à la phase suivante
    # Diagnostiques :
    #   - train_acc ≈ 1.0 ET test_acc < seuil -> mémorisation pure, augmenter n_steps
    #   - train_acc et test_acc tous deux bas -> underfitting, augmenter capacité / lr
```

**Preuve chiffrée (sans vs avec anti-shortcut, CODE_EXACT_SPECTRALM.md §6.5) :**
- `m1_short = 0.999` (avec shortcut, ans visible)
- `m1_honest = 0.240` (sans, ans masqué)
- `cascade = 0.240` (la VRAIE métrique de généralisation)
- Avec anti-shortcut complet (v3) : `m1_honest = 1.000`, `cascade = 0.984`.

> **La cascade est la VRAIE métrique de généralisation. Les métriques individuelles peuvent mentir.**

### 3.6 Surveillance de la convergence — signaux à monitorer

Pendant l'entraînement, surveiller (dans cet ordre de priorité) :

1. **Loss cosine moyenne par batch** — doit descendre monotone vers `< 0.01` après 1000 pas. Si plateau > 0.05 : vérifier l'init, le lr, le seed.
2. **Accuracy binaire (`eval_binary`)** — grimpe typiquement entre les pas 500 et 1500.
3. **Gap train/test** — doit rester `< 0.15` (classe) ou `< 0.20` (expérimental).
4. **Cascade accuracy** (compositionnelle) — la métrique honnête. Pour le crown-jewel arithmétique : attendre **+99.5 points** de gap entre one-shot et décomposition (validé dans STATUS.md).
5. **Validité du décodage** (`SymbolicDict.decode` retourne `valid`) — si `valid` chute, le vecteur ent devient flou (problème de sharpening, cf §5.5).

**Courbe typique validée (crown-jewel, 33s GPU) :**
- Binary block grokké : 100.0%
- ONE-SHOT test (1131 triples jamais vus) : 0.5%
- DÉCOMP LSRA test : **100.0%**
- Gap : **+99.5 points**

---

## 4. PROCÉDURE DE TUNING / FINE-TUNING

Cette section couvre l'extension du noyau grokké vers : (1) le multi-règles conjointes via `omni_rules`, (2) la génération amodale par flow-matching, (3) la perte jointe classify + generate, (4) l'alignement amodal par InfoNCE.

### 4.1 Multi-règles conjointes — `omni_rules`

Fichier : `ocm26400/omni_rules.py`. On entraîne **un seul** bloc op-aware conjointement sur 3 règles.

```python
N = P_MOD                                       # = 11
RULES = {
    "add":   lambda a, b: (a + b) % N,
    "mul":   lambda a, b: (a * b) % N,           # bilinéaire (vraiment différent d'add)
    "linop": lambda a, b: (3 * a + 5 * b) % N,
}
RULE_NAMES = list(RULES.keys())                  # ["add", "mul", "linop"]

def encode_rule(a, b, op_id, d):
    v = torch.zeros(D_MODEL)
    v[0:64]   = d.canonical(a)                   # ent
    v[64:128] = d.canonical(b)                   # prop
    v[128:192] = d.canonical(op_id)              # op = one-hot rule id
    return v
```

**Procédure d'entraînement (`omni_rules.py:train_omni_rules`) :**

```python
def train_omni_rules(d, n_steps=2000, lr=3e-3, batch=128):
    torch.manual_seed(0)
    blk = ReasonerBlock().to(DEVICE)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)
    for _ in range(n_steps):
        k = torch.randint(0, len(RULE_NAMES), (batch,))     # règle tirée
        a = torch.randint(0, N, (batch,))
        b = torch.randint(0, N, (batch,))
        batch_in = torch.stack([encode_rule(int(a[i]), int(b[i]), int(k[i]), d) for i in range(batch)]).to(DEVICE)
        out = blk(batch_in)
        loss = torch.tensor(0.0, device=DEVICE)
        for i in range(batch):
            name = RULE_NAMES[int(k[i])]
            tgt  = d.canonical(RULES[name](int(a[i]), int(b[i]))).to(DEVICE)
            ent  = out[i][0:64]
            cos  = (ent @ tgt) / (ent.norm() * tgt.norm() + 1e-8)
            loss = loss + (1.0 - cos)
        loss = loss / batch
        opt.zero_grad(); loss.backward(); opt.step()
    return blk
```

**Hyperparamètres EXACTS :** `n_steps=2000`, `batch=128` (double vs mono-règle), `lr=3e-3`, `Adam`, `seed=0`.

**Validation (`omni_rules.py:comprehend`) :** précision par règle sur `n_test=60` couples non vus. Acceptable si `> 0.9` par règle. Permet ensuite la génération inter-règles via `generate_chain` (chaîne mixte avec `op_id` par étape) et le calcul de ground-truth via `inter_rule_gt`.

### 4.2 Génération par flow-matching (décodeur conditionné AMV)

Fichier : `ocm26400/generators.py` (code en §1.7). Usage typique :

```python
decoder = AMVConditionedDecoder(x_dim=784, cond_dim=256, hidden=128)  # ex. MNIST 28x28
opt = torch.optim.Adam(decoder.parameters(), lr=3e-3)
for step in range(n_steps):
    cond = amv_batch                                 # (B, 256) — sortie du ReasonerBlock
    x_target = real_signals                          # (B, x_dim) — images/audio/etc.
    loss = decoder.flow_match_loss(cond, x_target)
    opt.zero_grad(); loss.backward(); opt.step()

# Inférence : Euler à 8 pas depuis le bruit
generated = decoder.sample(cond, steps=8)            # (B, x_dim)
```

**Pourquoi flow-matching et pas MSE** (Lipman 2023 / Esser SD3) :
- MSE feature-regression (ancienne approche `omni.py` `audio_dec`/`image_dec`) : décodeur linéaire qui régresse les features → pas de vraie génération, juste une projection.
- Flow-matching : apprend un champ de vélocité `v(x_t, t, cond)` intégré depuis le bruit gaussien vers la cible, **conditionné par l'AMV**. Vraie génération neurale.

### 4.3 Perte jointe classify + generate

Pour un fine-tuning complet (classification amodale + génération), on combine :

```python
# 1. Branche classification / raisonnement (sur le ReasonerBlock)
loss_reason = (1.0 - cos(out.ent, canonical(target)))           # cf §2.2

# 2. Branche génération (sur le décodeur flow-matching)
loss_gen    = decoder.flow_match_loss(cond=out, x_target=signal)

# 3. Branche alignement amodal (InfoNCE, cf §4.4)
loss_align  = info_nce_symmetric(z_a, z_b, tau=0.07)

# Perte jointe
L_total = loss_reason + λ_g * loss_gen + λ_a * loss_align
```

**Poids recommandés (point de départ, à ajuster) :** `λ_g = 1.0`, `λ_a = 0.1`. L'alignement amodal est une régularisation, pas un objectif dominant.

### 4.4 Alignement amodal — InfoNCE + ancrage

Fichier : `ocm26400/infonce.py`. C'est la **première pièce à intégrer** selon le verdict du juge (P1, cf §0.5).

```python
TAU_DEFAULT = 0.07   # valeur CLIP/SigLIP (infonce.py:14)

def info_nce(z_a, z_b, tau=TAU_DEFAULT):
    """L = -(1/N) Σ_i log[ exp(v_i·u_i / τ) / Σ_j exp(v_i·u_j / τ) ]
    z_a, z_b : batches (N, D). L2-normalisés en interne."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.t() / tau                    # (N, N)
    labels = torch.arange(z_a.shape[0])
    return F.cross_entropy(logits, labels)           # numériquement stable (logsumexp interne)

def info_nce_symmetric(z_a, z_b, tau=TAU_DEFAULT):
    return 0.5 * (info_nce(z_a, z_b, tau) + info_nce(z_b, z_a, tau))

def multimodal_l_consist(embeddings_per_mod, tau=TAU_DEFAULT):
    """InfoNCE symétrique moyenné sur C(M,2) paires de modalités. assert M >= 2."""
    ...
```

**Ancrage AMV (spec §1, §A1.3) :** `f_T(C) ~ f_A(C) ~ f_V(C) ~ f_S(C) ~ v_C` — toutes les modalités convergent vers le **même** AMV. On ajoute un terme d'ancrage :

```python
L_ancrage = ‖f_m(C) - E_C‖    # E_C = ligne du dictionnaire (LearnedVocab) pour C
```

**Procédure d'entraînement amodal sur données réelles** (`ocm26400/experiment_real_linguistic.py` + `real_linguistic.py`) :

```python
# 1000 mots réels, 4 modalités (text / morphology / phonology / semantic)
encoders = {m: RealViewEncoder(seed=i) for i, m in enumerate(MODALITIES)}   # seeds 0,1,2,3
opt = torch.optim.Adam(all_encoder_params, lr=3e-3)                         # joint sur toutes les modalités

for step in range(n_steps):       # 1500 (override du défaut 600)
    idx = torch.randint(0, N, (batch,))                                     # batch=64
    views = {m: encoders[m](bags[m][idx]) for m in MODALITIES}
    loss  = amodal_real_loss(views, tau=TAU_DEFAULT)                        # = multimodal_l_consist
    opt.zero_grad(); loss.backward(); opt.step()
```

**Hyperparamètres :** `n_steps=1500`, `batch=64`, `dim=PART(64)`, `tau=TAU_DEFAULT=0.07`, `lr=3e-3`, `Adam`. Validation : `retrieval@1 > 0.7` sur la meilleure paire informative (critère VALIDÉ).

**Limitation honnête (cf. §5.x) :** text ↔ morphology s'alignent fortement (dérivés des caractères du mot) ; phonology / semantic sont limités par les **collisions de sacs de features** (pattern 'ccvc' partagé → non distinguable quel que soit l'encodeur). Ce n'est pas un bug de l'encodeur, c'est un plafond lié aux données.

### 4.5 Conjugaison / morphologie — dispatch par op_id

Fichier : `ocm26400/experiment_conjugation.py`. Un seul bloc gère 3 temps via `op_id` (slot op). Substituts arithmétiques : past=verb+6, gerund=verb+12, third=verb+18 ; avec `N_VERBS=16`, `N_TRAIN_VERBS=12` (4 held-out), `N_TOTAL=64`.

**Procédure (1500 pas, batch=64, lr=3e-3, seed=0) :** identique à §2.2, sauf que la cible = `ver.compose(verb, 0, op_id=tense)`.

**Résultat honnête (`experiment_conjugation.py:122-136`) :**
- **op_id dispatch WORKS** : 1 bloc gère 3 temps, `dispatch_ok` si `tr_avg > 0.95`.
- **Flat verb → form map MEMORIZES** : 0% sur les verbes non vus. La morphologie nécessite une **décomposition compositionnelle** (stem + affixe), pas une table plate. Le crown-jewel linguistique (+100pt) confirme ce point.

> **Leçon :** op_id dispatch (multi-op dans 1 bloc) et généralisation compositionnelle sont **orthogonaux**. Le premier marche, le second exige la décomposition.

---

## 5. ERREURS CONNUES ET COMMENT LES ÉVITER

Toutes les erreurs ci-dessous ont été **rencontrées et corrigées** dans le projet. Les reproduire est la cause n°1 de non-reproductibilité.

### 5.1 Conjugaison naive (flat map)

**Symptôme :** 0% de généralisation sur verbes non vus.
**Cause :** le bloc apprend une table `verb → forme_conjuguée` plate ; aucune structure compositionnelle.
**Fix :** décomposer en `(stem, affixe)` et entraîner les sous-fonctions séparément, puis composer. Cf. `experiment_linguistic.py` (+100pt) et `experiment_linguistic_dense.py` (survie en dense).

### 5.2 Règle `y → i` (morphologie char-level en FFT)

**Symptôme :** 0% en morphologie char-level dans FFT (TESTING.md E14).
**Cause :** le padding domine le signal spectral ; la morphologie char-level est incompatible avec FFT.
**Fix :** utiliser des IDs numériques (pas des chars) pour les morphèmes, et le format SBS (Step-By-Step, cf §5.x ci-dessous). La FFT a besoin de **structure syntaxique** (séparateurs `|`, padding aligné), pas de chars bruts.

### 5.3 Gate stricte V=120 / dim=64 (problème de packing)

**Symptôme :** à `V=120, dim=64`, la gate stricte `cos1 ≥ 0.85` rejette beaucoup de sorties correctes mais imparfait. `validity` ne grimpe qu'à 34% → 54% → 65% à 3000/6000/10000 pas (raw stable ~99%).
**Cause :** c'est un problème de **netteté (sharpening)**, pas de correction. La gate bute géométriquement sur le packing 120/64.
**Fix honnête (levers) :** (1) relaxer la gate pour grand V, (2) augmenter dim, (3) sharpenner le bloc. **Ne PAS** conclure "le modèle ne sait pas" — la sortie brute est correcte, c'est le seuil qui est trop strict.

### 5.4 `compose()` signature fracturée (manque `op_id`)

**Symptôme :** crash ou perte du dispatch multi-op.
**Cause :** `verifier.py:65` prenait `compose(a, b)` **sans** `op_id` ; seul `V()` en avait un. `is_valid_intermediate` appelait `compose(a,b)` en perdant le dispatch.
**Fix (verdict juge) :** `compose(a, b, op_id=0)` + `is_valid_intermediate(a, b, m, op_id=0)` avec défaut `op_id=0` → rétrocompatible. `MorphologyVerifier` surcharge avec `op_id=CONJUGATE`. **Toujours** passer `op_id` dans tout nouveau code.

### 5.5 Seam bug ACSP (appel `l_consist()` à zéro arg)

**Symptôme :** crash `TypeError` au runtime.
**Cause :** `acsp.py` (ancienne version) appelait `delta * l_consist()` **sans argument**, or `l_consist(z_a, z_b, tau)` exige 2 args positionnels.
**Fix :** supprimer la ligne ; passer `consist_term: torch.Tensor | None = None` en kwarg à `acsp_loss` et faire `base + delta * consist_term` seulement si non `None`. C'est le **contrat unique d'extension** (un seul point, pas un kwarg par pièce).

### 5.6 LazyLinear init / cache plan FFT sous ROCm

**Symptôme (init) :** divergence ou convergence très lente.
**Cause :** défaut PyTorch Linear `~1/sqrt(256) ≈ 0.0625` trop grand pour le résiduel `x + h`.
**Fix :** init explicite `nn.init.normal_(weight, std=0.02)` + `nn.init.zeros_(bias)` pour **toutes** les couches du `ReasonerBlock` (déjà fait dans le code, cf §1.4). Ne jamais supprimer ces lignes.

**Symptôme (ROCm) :** ralentissement extrême en inférence variable-length.
**Cause :** recompilation du plan FFT à chaque longueur de séquence différente.
**Fix :** padder à `seq_len` fixe :
```python
seq_len_fixed = 64                       # ou config.seq_len
x = torch.full((1, seq_len_fixed), PAD_ID, device=dev)
x[:, :prompt_len] = prompt_tokens
# Un plan FFT compilé -> appels suivants < 0.3s
```
Optionnellement : `export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` ou `torch.cuda.set_per_process_memory_fraction(...)`.

### 5.7 `delta_m` dans deux espaces (cosinus vs euclidien)

**Symptôme :** seuils incohérents entre décodage et anti-collapse.
**Cause :** le décodage utilisait `cos1 - cos2 ≥ 0.05` (COSINUS) ; l'anti-collapse hinge utilisait `‖E_p - E_q‖ ≥ 0.05` (EUCLIDIEN). Pour des vecteurs unit-norm : `‖a - b‖ = 0.05 ⇔ cos = 0.99875` — les deux seuils ne parlent pas de la même chose.
**Fix :** UN seul espace (COSINUS partout, unit-norm). Décodage : `cos1 ≥ 0.85 ET cos1 - cos2 ≥ 0.05`. Anti-collapse : `cos_moyen_inter-paires ≤ 0.5` via uniformity loss.

### 5.8 Test de rang ne détecte pas le collapse

**Symptôme :** `matrix_rank(E) ≥ min(V, 64) - 2` passe sur une matrice E **clusterisée full-rank dégénérée**.
**Fix :** ajouter une **uniformity loss** (pénalité sur le cosinus moyen inter-paires) + un test de couverture angulaire. Garder le rang seulement comme garde-fou secondaire.

### 5.9 Polymorphisme silencieux (risque Frankenstein)

**Symptôme :** `LearnedVocab` (dense) sous-classe `SymbolicDict` et surcharge `canonical` / `decode` ; 6+ fichiers existants assument la sémantique one-hot (`allclose atol=1e-3`, `v[idx] = 1.0`). Passer un `LearnedVocab` où un `SymbolicDict` est attendu change le contrat **sans erreur de type** → échecs silencieux.
**Fix :** test de régression pinçant `SymbolicDict` one-hot comme défaut. Toujours typer explicitement et tester les deux chemins.

### 5.10 P3 (test-time compute) tautologique — REFRAMÉ

**Symptôme :** la trajectoire supervisée originelle `ent_t = (1-λ)^t · canonical(a) + (1 - (1-λ)^t) · canonical(op(a,b))` avec `λ < 0.5` rend le 1-pas **faux par construction** (`argmax = a` puisque `(1-λ) > 0.5`).
**Leçon :** on ne démontre PAS que l'itération **achète** de l'accuracy — on démontre qu'on peut entraîner un bloc à être délibérément lent.
**Reframe honnête :** la valeur réelle de la boucle LSRA est la **gate calibrée** qui REFUSE les OOD (100% ANOMALIE, AUROC 1.0), PAS "TTC improves accuracy". Ne jamais reformuler le P3 comme "l'itération améliore la précision".

### 5.11 Anti-shortcut symétrique (masquer TOUT)

**Symptôme :** 0% d'apprentissage (TESTING.md E12).
**Cause :** masquer les 18/18 variables récupérables détruit tout signal.
**Fix :** masquer **uniquement** les variables **algébriquement récupérables** depuis la cible courante (anti-shortcut asymétrique, loi L2).

### 5.12 Prémisse "92% v6" fausse

**Symptôme :** surrenchère sur les capacités de `spxlm_v6`.
**Cause :** le 92% était un cherry-pick de 9 champs linguistiques excluant `cat_id=0.22, syn_id=0.0`. La vraie source `v6_full_vocab_v3_results.json` donne `best_avg = 0.695`, et ce 69.5% vient de `diffuse_fill n_steps=4` (itératif), **pas** single-forward (~50-60% en single-forward).
**Fix :** toujours citer `v6_full_vocab_v3_results.json` comme source de vérité. Ne JAMAIS affirmer 92%.

### 5.13 Anti-shortcut mal placé (masquer le stem)

**Symptôme :** anti-shortcut instable, masque trop de signal.
**Cause :** masquer le **stem** au lieu des variables **algébriquement récupérables**.
**Fix :** masquer `c + d + m1 + m2 + ans` (variables récupérables depuis `c`), pas le stem.

### 5.14 Phases arithmétiques trop courtes

**Symptôme :** cascade < 0.95 en DOSC.
**Cause :** phases arithmétiques (entre positions de scratchpad) à 4000 pas au lieu de 6000.
**Fix :** les phases **arithmétiques** nécessitent **6000 pas MINIMUM** ; les phases **d'extraction depuis la prose** nécessitent seulement 2000 pas. C'est l'erreur n°6 des "6 erreurs courantes" (CODE_EXACT_SPECTRALM.md §19).

### 5.15 v6 generation crash (`causal_weight` 1D)

**Symptôme :** `IndexError` en génération causale.
**Cause :** `SpectralBlock.causal_weight` était 1D `(F,)` mais indexé `[:n, :]` (2D).
**Fix :** `causal_weight` → 2D `(F, 1)` avec broadcast propre.
**Limitation résiduelle :** la sortie causale reste du bruit (`cat plural=` → `--663c------`). Cause architecurelle : "un poids par fréquence dans le domaine de Fourier ne rend pas l'opération strictement causale. La causalité exacte exige un masque temporel + gestion de phase. Le 'SpectralBlock causal doux' est une fiction. → v6 = raisonneur, pas générateur."

### 5.16 Scaling d_model non-monotone

**Symptôme :** augmenter `d_model` CASSE le grokking (TESTING.md E01 vs E06).
**Leçon :** `d_model = 256` est optimum. `d=768` (268M Transformer) → 0% d'understanding. **Ne PAS** croire que "plus gros = meilleur". C'est non-monotone.

---

## 6. CONFIGURATION COMPLÈTE (référence)

### 6.1 Constantes du paquet `ocm26400` (tableau maître)

| Constante | Valeur | Fichier:ligne | Rôle |
|---|---|---|---|
| `D_MODEL` | `256` | `amv.py:11` | Dimension totale AMV |
| `PART` | `64` | `amv.py:12` | Taille d'une partition (4 × 64) |
| `P_MOD` | `11` | `verifier.py:16` | Base modulaire (Z₁₁) |
| `A_COEF` | `3` | `verifier.py:17` | Coeff entité : op(a,b) = (3a+5b) mod 11 |
| `B_COEF` | `5` | `verifier.py:18` | Coeff propriété |
| `P_BACKTRACK` | `1000.0` | `verifier.py:19` | Pénalité d'étape illégale (spec §2.2) |
| `TAU_GROK` | `0.9` | `reasoner.py:18` | Seuil de confidence pour stopper LSRA |
| `DEVICE` | `"cuda" if cuda else "cpu"` | `reasoner.py:19` | Device par défaut |
| `CONF_TARGET` | `4.0` | `reasoner.py:89` | Push meta[0] ici (sigmoid(4) ≈ 0.98 > 0.9) |
| `ALPHA` | `1.0` | `acsp.py:15` | Poids L_align |
| `BETA` | `1.0` | `acsp.py:16` | Poids L_step |
| `GAMMA` | `1e-3` | `acsp.py:17` | Poids L_sparse |
| `DELTA` | `0.0` | `acsp.py:18` | Poids L_consist (0 single-modality) |
| `TAU_DEFAULT` | `0.07` | `infonce.py:14` | Température InfoNCE (CLIP/SigLIP) |
| `N` | `P_MOD` (11) | `omni_rules.py:28` | Base modulo pour les règles |

### 6.2 Optimizers & hyperparamètres par trainer

| Trainer | Optimizer | lr | n_steps | batch | seed | Fichier |
|---|---|---|---|---|---|---|
| `train_binary_block` | Adam | 3e-3 | 1500 | 64 | 0 | `experiment_composition.py:30` |
| `train_reasoner_with_confidence` | Adam | 3e-3 | 800 | 64 | 0 | `reasoner.py:81` |
| `train_with_acsp` | Adam | 3e-3 | 1500 | 64 | 0 | `diff_decode.py:62` |
| `train_omni_rules` | Adam | 3e-3 | 2000 | 128 | 0 | `omni_rules.py` |
| `experiment_math` (multi-op) | Adam | 3e-3 | 2000 | 64 | 0 | `experiment_math.py` |
| `experiment_conjugation` | Adam | 3e-3 | 1500 | 64 | 0 | `experiment_conjugation.py` |
| `train_calibrated_block` (refinement) | Adam | 3e-3 | 1000 | 128 | 0 | `experiment_refinement.py` |
| `experiment_real_linguistic` | Adam (joint) | 3e-3 | 1500 | 64 | 0,1,2,3 | `real_linguistic.py:104` |

**Référence SPXLM v6 (à NE PAS confondre) :**
```python
# spxlm_v6/model.py (spectral) — recette DIFFÉRENTE :
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
warmup_steps = max(500, T // 20)                                  # 5% du training, min 500
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)           # OBLIGATOIRE
# Init : normal_(mean=0.0, std=0.02), weight tying lm_head/embedding
# Defaults : vocab_size=200, d_model=256, n_blocks=3, seq_len=64, refine_steps=3, mode="reasoning"
```

### 6.3 Schémas d'initialisation

| Composant | Init | Fichier |
|---|---|---|
| `ReasonerBlock.fc1/fc2.weight` | `normal_(std=0.02)` | `reasoner.py:33-34` |
| `ReasonerBlock.fc1/fc2.bias` | `zeros_` | `reasoner.py:35-36` |
| `SpectralCoreBlock.filter_real` | `randn * (1/sqrt(d_model)) + 1.0` | `spectral_core.py:39` |
| `SpectralCoreBlock.filter_imag` | `randn * (1/sqrt(d_model))` | `spectral_core.py:40` |
| `SpectralCoreBlock.in_proj/out_proj/FFN` | défauts PyTorch Linear | — |
| `AMVConditionedDecoder` | défauts PyTorch | — |
| `SpXLMv6` (toutes) | `normal_(mean=0.0, std=0.02)` + weight tying | `spxlm_v6/model.py` |

### 6.4 Seuils et critères de validation

| Critère | Seuil | Source |
|---|---|---|
| Grokking binaire | `acc ≥ 0.99` sur 121 paires | Validé STATUS.md |
| Crown-jewel arith | décomp - one-shot ≥ **+99.5 pts** (objectif +20) | Validé STATUS.md |
| Crown-jewel linguistique | décomp - one-shot ≥ **+100 pts** | Validé STATUS.md |
| Curriculum classe | `accuracy_threshold=0.85`, `max_shortcut_gap=0.15` | `curriculum.py:44-45` |
| Curriculum expérimental | `accuracy_threshold=0.80`, `max_shortcut_gap=0.20` | `experiment_curriculum.py` |
| Récursion profonde | `depth5_acc > 0.7` | `experiment_recursion.py` |
| Comprehend omni | `> 0.9` par règle | `experiment_math` |
| Refinement gate | `accept_rate > 0.8 AND reject_rate > 0.8` | `experiment_refinement.py` |
| Amodal réel | `retrieval@1 > 0.7` (meilleure paire) | `experiment_real_linguistic.py` |
| ACSP sanity | loss < 0.05 sur batch parfait | `test_acsp.py:64` |

### 6.5 Format SBS (Step-By-Step) pour le scratchpad

**Règle d'or (CODE_EXACT_SPECTRALM.md §3.2) :** chaque intermédiaire apparaît **EXACTEMENT UNE FOIS**.

```python
# CORRECT (un seul m1, masqué en cascade eval) :
f"{stem}#{a}{op1}{b}{op2}{c}{op3}{d}={m1:03d}|{m2:03d}|{ans:03d}"

# FAUX — m1 dupliqué, le modèle peut le copier :
f"{stem}:{a}{op1}{b}={m1:03d};{m1:03d}{op2}{c}={ans:03d}"
```

**Format SBS pour k=3** (réduit la distance op→résultat de 12 à 4 tokens, loi L9) :
```
"stem#m1; op1 c m2; op2 d m3; op3 e ans"
```

**Formule de longueur :** `L = 1 + 4·k` (k = nombre d'étapes). Capacité batch (3GB, d=256, n_blk=3) :
- k=1 : L=5, batch ~3800
- k=2 : L=9, batch ~2100
- k=3 : L=13, batch ~1460
- k=5 : L=21, batch ~900
- k=10 : L=41, batch ~460

### 6.6 Sanity checks à exécuter AVANT tout long entraînement (CODE_EXACT_SPECTRALM.md §13)

- **SC-1 :** Overfit 1 batch (100 pas sur ~32 tokens, loss < 0.01 obligatoire).
- **SC-2 :** Validation du format (positions des champs alignées avec les chars numériques ; chaque intermédiaire **EXACTEMENT UNE FOIS**).
- **SC-3 :** Vérification incrémentale du masque (imprimer 5 exemples masqués ; vérifier `single_frac ≈ 0.3 ± 0.05`).
- **SC-4 :** VRAM (`assert torch.cuda.memory_allocated() < 3e9` après le 1er forward).
- **SC-5 :** Accuracy par champ au pas 500 (chaque champ > 0.05, au-dessus du hasard).
- **SC-6 :** Cascade eval au pas 1000 (`A_cascade > 0.1`, strictement au-dessus du hasard). Si bloqué : réduire d, augmenter single_frac, vérifier le format.

### 6.7 Schedule DOSC (curriculum par dépendance topologique)

```
Phase 1 : masquer A uniquement (jamais B ni C)        -> grok A seul
Phase 2 : masquer B uniquement (A VISIBLE)             -> grok B = f(A) sans interférence
Phase 3 : masquer C uniquement (A, B VISIBLES)         -> grok C = g(A,B) sans interférence
Phase N : joint (interleaved)                          -> consolidation
```

**Phase Interleaved (anti-forgetting, §7) :** DOSC pur oublie catastrophiquement à chaque transition (P1→P2 : `c_honest 0.993 → 0.005 (-99.5%)` en 2000 pas — le weight space est partagé).

**Solution :** échantillonner **UN** champ par batch (probabilité 1/k). En 3 batches consécutifs, chaque tâche a une chance de renforcement → empêche l'oubli. **Chaque tâche conserve son anti-shortcut en interleaved** (loi L8 : ne PAS relâcher le masquage).

### 6.8 Commandes de reproduction

```bash
cd /media/akone/Dev/Dev_D/01_Dev_1/IA_Dev_D/12_Hermes/Recherches/MathsBase

# Tests RED (85-133 tests verts) :
python3 -m pytest ocm26400/ -q

# Crown-jewel arithmétique (~33s GPU) :
python3 -m ocm26400.experiment_composition
# Résultats attendus : binary 100%, one-shot 0.5%, décomp 100%, gap +99.5pts

# Crown-jewel linguistique (~27s GPU) :
python3 -m ocm26400.experiment_linguistic
# Résultats : one-shot 0%, décomp 100%, gap +100pts

# Survie one-hot → dense P2 (~64s) :
python3 -m ocm26400.experiment_linguistic_dense

# Scaling V>64 (Z_120) P2 (~90s) :
python3 -m ocm26400.experiment_vocab_scale

# Gate calibrée + abstention P3 (~45s) :
python3 -m ocm26400.experiment_refinement
# Résultats : accept_rate > 0.8, reject_rate > 0.8, AUROC ~1.0
```

Résultats JSON : `ocm26400/{crown_jewel,linguistic,linguistic_dense,vocab_scale,refinement}_results.json`.

---

## 7. CHECKLIST DE REPRODUCTION (à cocher)

Avant de lancer un entraînement, cocher **toutes** les cases :

### Pré-requis
- [ ] `cd MathsBase` ; `python3 -c "import torch; print(torch.cuda.is_available())"` renvoie `True`.
- [ ] `random.seed(0)` et `torch.manual_seed(0)` posés **avant** la création du modèle.
- [ ] `torch.use_deterministic_algorithms(True)` (optionnel mais recommandé).
- [ ] `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` (cf `reasoner.py:19`).

### Architecture
- [ ] `D_MODEL = 256`, `PART = 64` (4 partitions × 64).
- [ ] `P_MOD = 11`, `A_COEF = 3`, `B_COEF = 5`, `P_BACKTRACK = 1000.0`.
- [ ] `TAU_GROK = 0.9`, `CONF_TARGET = 4.0` (sigmoid(4) ≈ 0.98 > 0.9).
- [ ] Layout `meta` respecté : `meta[0]` = confidence LSRA, `meta[1]` = confidence bridge, `meta[2]` = consist cross-modale (§1.2).
- [ ] `ReasonerBlock` init : `normal_(std=0.02)` + `zeros_` biases (§5.7).
- [ ] `SpectralCoreBlock` init : filtre réel `+1.0`, imaginaire `0`, amplitude `1/sqrt(d_model)`.

### Pre-training (grokking)
- [ ] Optimizer = `Adam(lr=3e-3)` — **PAS AdamW** (réservé à spxlm_v6).
- [ ] `n_steps = 1500` (mono-op), `2000` (multi-op).
- [ ] `batch = 64` (mono), `128` (multi-règles / refinement).
- [ ] Loss = `(1 - cos)` sur `ent[0:64]` vs `canonical(op(a,b))`, moyenné sur batch.
- [ ] Epsilon `1e-8` dans le cosinus.
- [ ] Tirage `torch.randint(0, 121, (batch,))` WITH replacement.
- [ ] Sanity check `test_acsp.py:test_full_acsp_near_zero_on_perfect_trajectory` < 0.05.

### Training (ACSP)
- [ ] `acsp_loss_diff` utilisée (version différentiable Gumbel straight-through, `diff_decode.py`).
- [ ] `l_step_diff` (gradient réel) — **PAS** `l_step` (constante non-diff).
- [ ] Poids ACSP : `ALPHA=1.0, BETA=1.0, GAMMA=1e-3, DELTA=0.0` (single-modality).
- [ ] `compose(a, b, op_id=0)` — signature unifiée (§5.4).
- [ ] Curriculum : 4 phases (primitives → paires → chaînes → inter-règles).
- [ ] Critère de passage : `test_acc ≥ threshold AND gap ≤ max_shortcut_gap`.
- [ ] Anti-shortcut **asymétrique** (loi L2) — ne JAMAIS masquer toutes les variables.

### Tuning / Fine-tuning
- [ ] Multi-règles : `n_steps=2000, batch=128, lr=3e-3`, `op_id` one-hot dans slot op.
- [ ] Flow-matching : `AMVConditionedDecoder`, loss Euler, sample 8 pas.
- [ ] Perte jointe : `reason + λ_g·gen + λ_a·align`, `λ_g=1.0, λ_a=0.1` (départ).
- [ ] InfoNCE : `tau=0.07`, `info_nce_symmetric` pour CLIP-like.
- [ ] Amodal réel : `n_steps=1500` (override du défaut 600), `batch=64`, `dim=64`.

### Surveillance
- [ ] Loss cosine moyenne `< 0.01` après 1000 pas.
- [ ] `eval_binary` sur 121 paires ≈ 1.0 après 1500 pas.
- [ ] Gap train/test `< 0.15` (classe) ou `< 0.20` (expérimental).
- [ ] Cascade accuracy (compositionnelle) > one-shot de **+99.5 pts** (crown-jewel).
- [ ] `SymbolicDict.decode` retourne `valid=True` (pas de sharpening problem).

### Anti-erreurs (§5)
- [ ] Pas de conjugaison flat map (décomposition stem + affixe obligatoire).
- [ ] Pas de morphologie char-level en FFT (IDs numériques + SBS).
- [ ] Gate relaxée si V > 64 (sharpening, pas correction).
- [ ] `compose(a, b, op_id=0)` partout (rétrocompatible).
- [ ] `acsp_loss` appelée avec `consist_term` en kwarg (pas d'appel `l_consist()` à zéro arg).
- [ ] Init `std=0.02` explicite (pas de LazyLinear default).
- [ ] Padding à `seq_len` fixe sous ROCm (cache plan FFT).
- [ ] Un seul espace (COSINUS unit-norm) pour `delta_m`.
- [ ] Uniformity loss pour anti-collapse (pas seulement rang).
- [ ] Test de régression pinçant `SymbolicDict` one-hot (anti-Frankenstein).
- [ ] P3 reframé en gate calibrée + abstention (PAS "TTC improves accuracy").
- [ ] Anti-shortcut asymétrique (PAS masquer tout).
- [ ] Citer `v6_full_vocab_v3_results.json` (best_avg=0.695, PAS 92%).
- [ ] Phases arithmétiques ≥ 6000 pas (PAS 4000).
- [ ] `d_model = 256` (PAS 768 — non-monotone).

---

## ANNEXE — Glossaire

- **AMV** (Amodal Mentalese Vector) — vecteur `R^256` à 4 partitions (ent/prop/op/meta).
- **ACSP** (Amodal Consistency & Step Penalty) — loss `α·L_align + β·L_step + γ·L_sparse + δ·L_consist`.
- **LSRA** (Latent Symbolic Recurrence with Abstention) — boucle récurrente `v(t+1) = Block(v(t))` avec gate symbolique.
- **Grokking** — passage d'une accuracy faible à une généralisation complète après un nombre suffisant de pas (ici, opérationnel : 1500-2000 pas suffisent).
- **Crown-jewel** — la démonstration que la décomposition compositionnelle (LSRA, 2-pas) généralise à +99.5 pts vs one-shot, sur données jamais vues.
- **Anti-shortcut** — masquer les variables algébriquement récupérables pour forcer le calcul réel (loi L2).
- **DOSC** (Dependencies-One-at-a-Time Sequential Curriculum) — curriculum par dépendance topologique, un champ par phase.
- **SBS** (Step-By-Step) — format de scratchpad où chaque intermédiaire apparaît exactement une fois, distance op→résultat ≈ 4 tokens.
- **Flow-matching** — apprentissage d'un champ de vélocité intégré depuis le bruit vers la cible (Lipman 2023).
- **InfoNCE** — loss contrastive CLIP/SigLIP avec température τ=0.07.
- **P3** — pièce "test-time compute", **reframée** en gate calibrée + abstention (AUROC 1.0 sur OOD), PAS en "itération améliore l'accuracy".

---

*Document généré le 20 Juin 2026 à partir du code source `ocm26400/*.py` et des verdicts validés. Toutes les valeurs numériques sont exactes (vérifiées en code). Pour toute divergence, le code source fait foi.*
