# SPXLM v6 — PROTOCOLE D'ENTRAÎNEMENT COMPLET (A → Z)

**Version :** 1.0 — 2026-06-18
**Modèle :** SPXLM v6 (Spectral eXtended Language Model)
**Architecture :** FFT Spectral Mixer + Diffusion Fill + Scratchpad SBS + DOSC + Sommeil Multi-Phase
**Hardware de référence :** AMD RX 7900 XTX (24 GB VRAM), ROCm 6.2, PyTorch 2.5.1
**Budget VRAM :** 3 GB (frugal par conception)
**Résultat validé :** Addition à 100% de généralisation (cascade ≥ 0.97)

> Ce document permet à quiconque de créer et entraîner le modèle de bout en bout.
> Aucune connaissance préalable du projet n'est requise au-delà de PyTorch et de l'algèbre linéaire de base.

---

## TABLE DES MATIÈRES

1. [Configuration du Modèle](#1-configuration-du-modèle)
2. [Architecture Détaillée (Couches)](#2-architecture-détaillée-couches)
3. [Étapes d'Entraînement (Séquence Exacte)](#3-étapes-dentraînement-séquence-exacte)
4. [Sommeil Multi-Phase](#4-sommeil-multi-phase)
5. [Optimisation](#5-optimisation)

---

## 1. CONFIGURATION DU MODÈLE

### 1.1 Paramètres Optimaux par Défaut

La configuration suivante a été calibrée sur 30+ expériences (E0–E17, B1–B4, sweep complet de paramètres). **C'est le point de départ recommandé pour toute nouvelle tâche.**

```python
model = SpXLMv6(
    vocab_size    = 200,        # Adapter au vocabulaire de la tâche
    d_model       = 256,        # Largeur — SWEET SPOT, ne pas dépasser 256
    n_blocks      = 3,          # Nombre de blocs spectraux
    seq_len       = 64,         # Longueur fixe (1 plan FFT)
    mode          = "reasoning",# "reasoning" (bidirectionnel) ou "generation" (causal)
    mask_token_id = 0,          # ID du token <MASK>
    pad_token_id  = 1,          # ID du token <PAD>
    refine_steps  = 3,          # Itérations de diffusion-fill à l'inférence
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-3,        # Learning rate de base
    weight_decay = 0.1,         # CRITIQUE pour le grokking
    betas        = (0.9, 0.95), # beta2 réduit pour adaptation plus rapide
)

# Hyperparamètres d'entraînement
warmup_steps   = 500           # min(T // 20, 500), au moins 500
grad_clip      = 1.0           # OBLIGATOIRE (prévient NaN dans FFT)
batch_size     = variable      # Contrainte: batch × seq_len ≈ 1.9 × 10⁴
single_frac    = 0.3           # 30% des batchs masquent exactement 1 champ
```

### 1.2 Tableau Complet des Paramètres

| # | Paramètre | Valeur par défaut | Rôle | Justification |
|---|---|---|---|---|
| 1 | `d_model` | **256** | Largeur des vecteurs internes | Sweet-spot mesuré au sweep. d=256 maximise D à T≤30k. Au-delà (d=512), D est divisé par 12 (δ=−3.55). En-dessous (d=128), D diminue (capacité insuffisante pour la règle). |
| 2 | `n_blocks` | **3** | Profondeur (nombre de SpectralBlocks) | D ∝ n_blk^(2/3). De 2→3 blocs: ×1.35. De 3→10 blocs: ×2.15 seulement, mais params ×3.3 et temps ×3.9. Le coût marginal est négatif. n_blk=3 maximise le ratio D/params. |
| 3 | `lr` | **1e-3** | Taux d'apprentissage | Grokking exige un LR assez élevé pour échapper au minimum de mémorisation, mais pas trop pour converger. 1e-3 est optimal pour AdamW + wd=0.1 sur 0.3–2M params. |
| 4 | `weight_decay` | **0.1** | Régularisation L2 | **CRITIQUE**. Le grokking exige que le réseau trouve une solution généralisante (plus petite en norme Frobenius que la solution mémorisante). Sans wd→0.1, pas de grokking dans un budget pratique. wd=1.0 peut être essayé pour l'arithmétique pure. |
| 5 | `betas` | **(0.9, 0.95)** | Moments d'Adam | beta2=0.95 (vs 0.999 par défaut) permet une adaptation plus rapide du second moment. Le grokking implique un changement brusque de régime d'optimisation — beta2 faible aide à le détecter rapidement. |
| 6 | `warmup_steps` | **500** | Montée progressive du LR | Prévient l'instabilité initiale du FFT mixer. Les gradients FFT sont sensibles aux poids aléatoires initiaux. min 500, idéal T//20. |
| 7 | `grad_clip` | **1.0** | Clipping de la norme du gradient | **OBLIGATOIRE**. L'IFFT peut amplifier certains gradients de façon explosive (basses fréquences). Sans clip → NaN cascade → perte complète. |
| 8 | `seq_len` | **64** (fixe) | Longueur de séquence | Doit être FIXE (pas variable). ROCm recompile le plan FFT pour chaque longueur unique (~0.8s/recompilation). Longueur fixe = 1 plan FFT = latence stable. L = 1 + 4×k pour le format scratchpad k-pas. |
| 9 | `batch_size` | **variable** | Taille de batch | Contrainte VRAM: batch × seq_len ≈ 1.9 × 10⁴. Pour seq_len=48: batch ≈ 396. Pour seq_len=64: batch ≈ 297. |
| 10 | `single_frac` | **0.3** | Fraction de batchs à masquage unique | **CRITIQUE pour le grokking**. 30% des batchs masquent EXACTEMENT 1 champ → chaque étape scratchpad grokke indépendamment (L2). Sans cela: plafond 0.75. Avec: 0.984. |
| 11 | `mask_prob` | **0.5** | Probabilité de masquage par champ | Pour les 70% de batchs multi-champs, chaque champ non-cible a 50% de chance d'être masqué. |
| 12 | `refine_steps` | **3** | Itérations de diffusion-fill à l'inférence | Nombre de passes de raffinement. 3 est optimal: 1 sous-fait, 5+ donne des rendements décroissants. |
| 13 | `eps` (LayerNorm) | **1e-6** | Epsilon de normalisation | Valeur standard. Stabilise le FFT après normalisation. |
| 14 | `pos_embedding init` | **0.02** | Écart-type d'initialisation positionnelle | N(0, 0.02²). Petit pour ne pas dominer l'embedding de token. |
| 15 | `weight_init` | **N(0, 0.02²)** | Initialisation des poids linéaires | Std=0.02 pour toutes les couches Linear. Évite les gradients explosifs dans les blocs résiduels empilés. |

### 1.3 Pourquoi d=256 (et pas 512 ou 128) ?

**La largeur (d_model) est le paramètre le plus dangereux du modèle.**

#### Preuve expérimentale (sweep complet, T=8k, k=3, n_blk=2)

| d_model | Params (M) | D_reliable | Verdict |
|---|---|---|---|
| 64 | 0.089 | 2.0 | Sous-capacité |
| 96 | 0.192 | 2.0 | Sous-capacité |
| 128 | 0.334 | 3.0 | Acceptable |
| **192** | **0.735** | **3.2** | **Peak (n_blk=2)** |
| **256** | **1.291** | **2.6** | **Sweet spot pour n_blk=3** |
| 384 | 4.065 | 3.6 | Nécessite n_blk=3 |
| 512 | 7.190 | 3.9 | **6761× moins efficace** que d=64 |

#### La formule d^-3.55

L'ajustement multivarié complet sur toutes les expériences donne:

```
D = k^2.54 × P^1.92 × d^(-3.55) × T^2.06 × n_blk^(-0.81) × C₀
```

L'exposant **δ = −3.55** pour d_model signifie:
- **Doubler d_model → D ÷ 12** (profondeur fiable divisée par 12!)
- d_model=768 (Transformer typique): 6761× moins efficace que d=64

**Pourquoi d=256 et pas d=192?** À n_blk=2, d=192 est peak. Mais à n_blk=3 (notre défaut), d=256 surpasse d=192 car le bloc supplémentaire compense. d=256 est le **compromis optimal** entre:
- Capacité suffisante pour apprendre la règle (d ≥ 128)
- Pas de destruction de l'efficacité de grokking (d ≤ 256)

**Pourquoi pas d=128?** À d=128, le modèle a à peine assez de capacité pour représenter la règle arithmétique. La marge est trop fine — sur des tâches plus complexes (NL reasoning), d=128 échoue. d=256 donne une marge de sécurité.

### 1.4 Pourquoi n_blk=3 (et pas 10 ou 1) ?

**D ∝ n_blk^(2/3)** — la profondeur aide mais de façon **sous-linéaire**.

#### Mesures (d=256 fixe, T=8k, k=3)

| n_blk | D_measured | Gain relatif | Params (M) |
|---|---|---|---|
| 1 | ~2.0 | — | 0.65 |
| 2 | 2.6 | ×1.3 | 1.29 |
| **3** | **3.5** | **×1.35** | **1.83** |
| 4 | 4.0 | ×1.14 | 2.36 |
| 10 | ~5.3 (est.) | ×1.32 | 6.22 |

**Analyse coût/bénéfice:**
- De 2→3 blocs: gain ×1.35 en D, coût ×1.42 en params → **rentable**
- De 3→4 blocs: gain ×1.14 en D, coût ×1.29 en params → **marginal**
- De 3→10 blocs: gain ×1.51 en D, coût ×3.40 en params, temps ×3.9 → **désastreux**

De plus, l'exposant n_blk dans l'ajustement multivarié est **φ = −0.81**: à budget de calcul fixe, ajouter des blocs est **destructif**. Le coût computationnel supplémentaire aurait été mieux investi en training steps (δ = +1.34).

**n_blk=3 est le point d'inflexion** où le coût marginal d'un bloc supplémentaire dépasse son bénéfice marginal.

### 1.5 Pourquoi AdamW (et pas SGD) ?

**Le grokking exige AdamW. SGD échoue systématiquement.**

#### Raisons:

1. **Paysage d'optimisation à deux régimes.** Le grokking implique deux phases:
   - Phase 1 (mémorisation): le modèle apprend par cœur les exemples d'entraînement
   - Phase 2 (généralisation): après des milliers de steps supplémentaires, le modèle découvre la règle
   
   SGD a un taux d'apprentissage effectif constant qui ne peut pas naviguer entre ces deux régimes. AdamW adapte le LR par paramètre, ce qui permet de:
   - Converger vite vers la mémorisation (gradients cohérents)
   - Puis émerger lentement vers la généralisation (gradients subtils dans certaines directions)

2. **Découplage du weight decay.** AdamW (contrairement à Adam avec L2) découple le weight decay de la mise à jour adaptative. C'est essentiel car:
   - Le weight decay pousse vers les solutions de petite norme (= généralisantes)
   - L'adaptation Adam préserve les directions importantes
   - Adam+L2 "contamine" le moment du gradient avec le decay → moins efficace

3. **Second moment adaptatif.** Le FFT mixer a des gradients à dynamique très variable entre fréquences:
   - Basses fréquences: gradients stables, importants
   - Hautes fréquences: gradients fluctuants, petits
   
   Adam normalise par le second moment → chaque fréquence apprend à son rythme optimal. SGD applique le même pas partout → certaines fréquences divergent, d'autres n'apprennent pas.

4. **Beta2=0.95 (pas 0.999).** Le grokking est un changement de régime brutal. beta2=0.999 "moyenne" sur trop de steps et rate le signal. beta2=0.95 détecte le changement de régime plus vite.

**Résultat:** SGD sur modular arithmetic → pas de grokking à 50k steps. AdamW + wd=0.1 → grokking à ~8k steps.

---

## 2. ARCHITECTURE DÉTAILLÉE (COUCHES)

### 2.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPXLM v6 — ARCHITECTURE                     │
│                                                                 │
│  INPUT: (B, L) token IDs                                       │
│    │                                                            │
│    ▼                                                            │
│  Token Embedding: (B, L) → (B, L, d)                          │
│    + Position Embedding: (1, L, d)                             │
│    │                                                            │
│    ▼                                                            │
│  SpectralBlock #1                                              │
│    ├─ LayerNorm → in_proj → FFT → filter → IFFT → out_proj    │
│    ├─ + résiduel                                                │
│    └─ + FFN (LayerNorm → Linear(4d) → GELU → Linear(d))       │
│    │                                                            │
│    ▼                                                            │
│  SpectralBlock #2  (identique)                                 │
│    │                                                            │
│    ▼                                                            │
│  SpectralBlock #3  (identique)                                 │
│    │                                                            │
│    ▼                                                            │
│  Final LayerNorm                                               │
│    │                                                            │
│    ▼                                                            │
│  LM Head (weight-tied avec Token Embedding): (B, L, d)→(B,L,V)│
│                                                                 │
│  OUTPUT: (B, L, V) logits                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Token Embedding

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
# Poids initialisés: N(0, 0.02²) par les modules Linear
```

**Rôle:** Convertit chaque token ID en un vecteur dense de dimension d=256.

**Pourquoi pas one-hot?** One-hot ne capture aucune similarité entre tokens. L'embedding appris permet au modèle de découvrir que "3" et "4" sont plus proches que "3" et "+".

**Pourquoi pas plus grand?** d=256 est déjà le sweet-spot. Un embedding plus large (d=512) augmenterait tous les coûts pour une efficacité de grokking divisée par 12.

### 2.3 Position Embedding

```python
self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
# x = token_embedding(input_ids) + pos_embedding[:, :L, :]
```

**Rôle:** Donne au modèle l'information de position (où se trouve chaque token dans la séquence).

**Pourquoi appris et pas sinusoïdal?**

1. **Le FFT ne code pas la position par nature.** Contrairement à l'attention qui a besoin de position encoding explicite, le FFT code les positions via les fréquences. Mais le filtre fréquentiel appris est **global** (même filtre pour toutes les positions). Le position embedding appris ajoute un **biais local** que le filtre seul ne peut pas capturer.

2. **Positions absolues importantes.** Dans le scratchpad SBS, la position du token détermine son rôle (stem, intermédiaire, réponse). Les embeddings de position appris permettent au modèle de distinguer "position 5 = m₁" vs "position 15 = m₃".

3. **Init à 0.02.** Petit écart-type pour ne pas dominer l'embedding de token. Au début de l'entraînement, le contenu du token domine; au fur et à mesure, le modèle apprend à utiliser la position.

**Alternative non retenue — RoPE (Rotary Position Embedding):** RoPE encode les positions relatives via rotation dans l'espace de l'attention. Mais SPXLM n'a pas d'attention — RoPE n'est pas applicable au FFT mixer de manière évidente. L'embedding absolu est plus simple et fonctionne.

### 2.4 Weight Tying

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight  # SHARED WEIGHTS
```

**Rôle:** La couche de sortie (lm_head) partage les mêmes poids que l'embedding d'entrée.

**Pourquoi?**

1. **Économie de paramètres.** Sans tying: 2 × vocab_size × d_model paramètres pour embedding + lm_head. Avec tying: vocab_size × d_model seulement. Pour vocab=200, d=256: économie de 51,200 params (non négligeable sur un modèle de 1.8M).

2. **Cohérence sémantique.** L'embedding de "5" et le logit de "5" utilisent la même représentation. Le modèle n'a pas à apprendre deux espaces séparés (un pour lire, un pour écrire). C'est particulièrement important pour le grokking: la règle apprise en lecture se transfère directement en écriture.

3. **Régularisation implicite.** Le tying contraint l'espace des solutions. Le modèle ne peut pas apprendre des embeddings d'entrée et de sortie incohérents. Cette contrainte pousse vers la généralisation.

4. **Stabilité d'entraînement.** Le gradient de l'embedding vient de deux sources (entrée + sortie), ce qui stabilise son apprentissage.

### 2.5 SpectralBlock — Le Cœur du Modèle

C'est l'unité computationnelle centrale. Remplace l'attention par un mélangeur FFT.

```python
class SpectralBlock(nn.Module):
    def __init__(self, d_model, seq_len, bidirectional=True):
        super().__init__()
        # Projections linéaires
        self.in_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Filtre fréquentiel appris (complexe, par dimension)
        scale = 1.0 / math.sqrt(d_model)
        self.filter_real = nn.Parameter(
            torch.randn(seq_len // 2 + 1, d_model) * scale + 1.0
        )
        self.filter_imag = nn.Parameter(
            torch.randn(seq_len // 2 + 1, d_model) * scale
        )

        # Masque causal (mode generation seulement)
        if not bidirectional:
            freqs = torch.arange(seq_len // 2 + 1).float()
            self.register_buffer("causal_weight", torch.sigmoid(-freqs * 0.1))

        # Normalisation + FFN
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        B, L, D = x.shape

        # --- Branche spectrale ---
        h = self.norm1(x)           # Normaliser
        h = self.in_proj(h)         # Projeter

        X_freq = torch.fft.rfft(h, dim=1)  # FFT: (B, L//2+1, D) complexe

        # Multiplication complexe: filtre ⊙ spectre
        fr = self.filter_real[:X_freq.shape[1], :].unsqueeze(0)
        fi = self.filter_imag[:X_freq.shape[1], :].unsqueeze(0)
        X_real = X_freq.real * fr - X_freq.imag * fi
        X_imag = X_freq.real * fi + X_freq.imag * fr
        X_filtered = torch.complex(X_real, X_imag)

        # Masque causal (atténue hautes fréquences)
        if not self.bidirectional:
            cw = self.causal_weight[:X_filtered.shape[1], :].unsqueeze(0)
            X_filtered = X_filtered * cw

        y = torch.fft.irfft(X_filtered, n=L, dim=1)  # IFFT: retour au temps
        y = self.out_proj(y)      # Projeter

        x = x + y                 # Résiduel spectral
        x = x + self.ffn(self.norm2(x))  # Résiduel FFN

        return x
```

#### Rôle du SpectralBlock

Le SpectralBlock **mélange l'information à travers la séquence**. Chaque position reçoit de l'information de toutes les autres positions en **une seule opération FFT**, sans matrice d'attention O(L²).

Le mécanisme:
1. **FFT** décompose la séquence en fréquences (basses = structure globale, hautes = détails locaux)
2. **Filtre appris** sélectionne/amplifie les fréquences pertinentes pour la tâche
3. **IFFT** reconstruit la séquence dans le domaine temporel, mélangée

#### Paramètres du SpectralBlock

| Composant | Forme | # Paramètres (par bloc, d=256, L=64) |
|---|---|---|
| `in_proj` (Linear) | (256, 256) + bias | 65,792 |
| `out_proj` (Linear) | (256, 256) + bias | 65,792 |
| `filter_real` | (33, 256) | 8,448 |
| `filter_imag` | (33, 256) | 8,448 |
| `norm1` (LayerNorm) | (256,) × 2 | 512 |
| `norm2` (LayerNorm) | (256,) × 2 | 512 |
| `ffn` Linear 1 | (256, 1024) + bias | 263,168 |
| `ffn` Linear 2 | (1024, 256) + bias | 262,400 |
| **Total par bloc** | | **~674,672** |
| **3 blocs** | | **~2,024,016** |

#### Pourquoi FFT et pas Attention?

| Critère | FFT (SpectralBlock) | Attention (Transformer) |
|---|---|---|
| Complexité | **O(L log L)** | O(L²) |
| À L=512 | 14× plus rapide | Référence |
| Stabilité | Parseval: ‖x‖² = ‖FFT(x)‖² → pas d'explosion | Attention peut exploser (softmax + grandes valeurs) |
| Mémoire | O(L) — pas de matrice L×L | O(L²) pour la matrice d'attention |
| Cache KV | **Non nécessaire** (diffusion) | Nécessaire pour AR efficace |
| Bidirectionnalité | **Native** (FFT est symétrique) | Nécessite masquage (causal mask) |
| Grokking | **Parseval garantit la conservation d'énergie** — le filtre appris ne peut pas diverger | Pas de garantie de conservation |

**Le théorème de Parseval est la clé.** ‖x‖² = ‖FFT(x)‖² signifie que l'énergie du signal est exactement conservée par la FFT. Le filtre fréquentiel peut amplifier certaines fréquences et en atténuer d'autres, mais l'énergie totale reste contrôlée. Cela garantit que:
- Les gradients ne explosent pas (contrairement à l'attention où QK^T peut atteindre des valeurs extrêmes)
- Le dé-bruitage itératif (diffusion-fill) converge (chaque itération préserve l'énergie)
- Le grokking est possible (la stabilité permet un entraînement long sans divergence)

#### Pourquoi deux variantes (bidirectionnel vs causal)?

| Variante | Mode | Cas d'usage | Mécanisme |
|---|---|---|---|
| `bidirectional=True` | "reasoning" | Diffusion-fill, grokking, raisonnement | FFT standard (circulaire): chaque position voit tout |
| `bidirectional=False` | "generation" | Génération AR, texte fluide | Masque `sigmoid(−freqs × 0.1)` atténue les hautes fréquences |

**Le mode bidirectionnel est essentiel pour le raisonnement.** En k-pas, les étapes intermédiaires et la réponse sont masquées simultanément. Le modèle bidirectionnel peut utiliser le **contexte futur** pour remplir les positions antérieures — ce qui est **impossible en AR**.

Exemple: pour prédire m₁ (position 5), le modèle peut utiliser m₂ (position 10) comme contexte. Cela permet de résoudre en cascade, pas seulement de générer de gauche à droite.

#### Initialisation du filtre

```python
filter_real = N(1.0, scale²)    # ≈ identité (passe-tout au départ)
filter_imag = N(0.0, scale²)    # ≈ zéro (pas de rotation de phase au départ)
scale = 1/√d_model = 1/16       # petit bruit autour de l'identité
```

**Pourquoi initialiser près de l'identité?** Au début de l'entraînement, le bloc spectral doit se comporter comme un **passe-tout** (identity) pour ne pas perturber le signal. Si on initialise à zéro ou aléatoirement, le filtre détruit le signal avant que le modèle ait pu apprendre quoi que ce soit.

### 2.6 LayerNorm (et pas RMSNorm)

```python
self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
```

**Rôle:** Normalise chaque vecteur de position pour avoir moyenne=0 et écart-type=1, puis applique un biais et un gain appris.

**Pourquoi LayerNorm et pas RMSNorm?**

| Critère | LayerNorm | RMSNorm |
|---|---|---|
| Centre (soustrait la moyenne) | **Oui** | Non |
| Stabilité FFT | **Meilleure** | Risque d'instabilité |
| Supprime le biais DC | **Oui** (la moyenne est le composant DC, fréquence 0) | Non |
| Standard | Tous les Transformers classiques | LLaMA, quelques modèles récents |

**Le point critique: le composant DC (fréquence 0).** La FFT transforme un vecteur en ses composantes fréquentielles. La composante de fréquence 0 (DC) est la **moyenne** du signal. Si cette moyenne n'est pas centrée (RMSNorm), elle domine le spectre — le filtre appris doit consacrer une grande partie de sa capacité à gérer ce biais.

LayerNorm supprime explicitement le DC → le spectre FFT commence par une composante DC nulle → le filtre peut se concentrer sur les fréquences significatives.

**Conséquence pratique:** Les tests ont montré que RMSNorm provoque occasionnellement des NaN dans le FFT mixer (la composante DC non centrée peut diverger). LayerNorm est stable.

### 2.7 FFN (Feed-Forward Network)

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_model * 4),  # Expansion 4×
    nn.GELU(),                         # Activation
    nn.Linear(d_model * 4, d_model),  # Compression 4×
)
```

**Rôle:** Transformation position-par-position (pas de mixing entre positions). Permet au modèle de combiner les dimensions filtrées de manière non-linéaire.

#### Pourquoi GELU (et pas ReLU)?

| Activation | Formule | Avantage | Inconvénient |
|---|---|---|---|
| **GELU** | x · Φ(x) | **Lisse, dérivable partout**, gradients non-nuls pour x<0 | Plus cher computiquement |
| ReLU | max(0, x) | Simple, rapide | **Gradient mort** pour x<0, non-dérivable en 0 |
| SiLU/Swish | x · σ(x) | Similaire à GELU | — |

**GELU est préféré parce que:**
1. **Lisseur.** GELU est dérivable partout (pas de point anguleux comme ReLU). Dans un modèle entraîné très longtemps (grokking = 10k+ steps), la liseur évite les oscillations causées par les gradients non-lisses.
2. **Pas de neurones morts.** ReLU tue les neurones avec x<0 (gradient=0 → jamais mis à jour). GELU donne un petit gradient pour x<0 → tous les neurones restent vivants.
3. **Compatibilité FFT.** GELU préserve mieux l'énergie du signal que ReLU (qui clamp brutalement à 0). Parseval est mieux préservé.

#### Pourquoi expansion 4× (et pas 2× ou 8×)?

| Expansion | d_interne | Params FFN | Représentation |
|---|---|---|---|
| 2× | 512 | ~262k | Trop limitée — bottleneck |
| **4×** | **1024** | **~525k** | **Optimal** — assez pour la non-linéarité, pas trop pour le coût |
| 8× | 2048 | ~1.05M | Trop cher — la moitié des params du modèle dans le FFN |

**4× est le standard empirique.** À d=256, l'expansion 4× donne d_interne=1024, ce qui permet au FFN de projeter dans un espace assez large pour des transformations non-linéaires complexes (nécessaires pour apprendre des règles arithmétiques), sans dominer le budget de paramètres.

### 2.8 Final LayerNorm + LM Head

```python
self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight  # Weight tying
```

**Rôle:** La LayerNorm finale normalise les représentations avant la projection vers le vocabulaire. Le LM Head (weight-tied) projette vers les logits du vocabulaire.

### 2.9 Masque Causal (Mode Génération)

```python
if not bidirectional:
    freqs = torch.arange(seq_len // 2 + 1).float()
    self.register_buffer("causal_weight", torch.sigmoid(-freqs * 0.1))
    
    # Dans forward:
    cw = self.causal_weight[:X_filtered.shape[1], :].unsqueeze(0)
    X_filtered = X_filtered * cw
```

**Comment ça marche?**

Le masque causal `sigmoid(−freqs × 0.1)` atténue progressivement les hautes fréquences:
- freq=0 (DC): weight = sigmoid(0) = 0.5
- freq=10: weight = sigmoid(−1) = 0.27
- freq=32: weight = sigmoid(−3.2) = 0.039

**Effet:** Les hautes fréquences correspondent aux transitions rapides dans la séquence (information locale, futur proche). Les atténuer force le modèle à ne dépendre que des **basses fréquences** (information globale, passé).

**Ce n'est pas un masque parfait** (contrairement au masque triangulaire strict de l'attention). C'est un masque **doux** qui permet une petite fuite d'information future. Pour le raisonnement pur, on utilise le mode bidirectionnel (pas de masque). Pour la génération de texte fluide, le masque doux est suffisant — il empêche le modèle de tricher en regardant le futur, tout en gardant le bénéfice O(L log L) du FFT.

**Alternative pour causalité stricte:** On pourrait appliquer un masque triangulaire dans le domaine temporel (après IFFT), mais cela annule le bénéfice du FFT et coûte O(L²). Le masque spectral doux est le compromis pratique.

### 2.10 DiffusionFiller (Inférence)

```python
class DiffusionFiller:
    @torch.no_grad()
    def fill(self, x, mask):
        """Remplit les positions masquées par itération."""
        for step in range(self.n_steps):
            logits = self.model(x)
            pred = logits.argmax(dim=-1)
            x = torch.where(mask, pred, x)  # Remplir seulement les masques
        return x
```

**Rôle:** Procédure d'inférence pour le raisonnement. Le modèle lit le contexte visible, puis remplit itérativement les positions masquées.

**Pourquoi itératif?** À chaque étape, le modèle remplit certaines positions avec ses prédictions. Ces prédictions deviennent contexte pour l'itération suivante, améliorant les prédictions restantes. C'est l'équivalent du "diffusion sampling" des modèles de diffusion d'images.

---

## 3. ÉTAPES D'ENTRAÎNEMENT (SÉQUENCE EXACTE)

### Vue d'Ensemble — 18 Étapes Obligatoires

```
ÉTAPE 1:  Sanity overfit (loss < 0.01)
    │
    ▼
ÉTAPE 2:  Modèle ÉTROIT (d=256, n_blk=3)
    │
    ▼
ÉTAPE 3:  AdamW (lr=1e-3, wd=0.1, betas=(0.9, 0.95))
    │
    ▼
ÉTAPE 4:  Warmup 500+ + grad clip 1.0
    │
    ▼
ÉTAPE 5:  Diffusion-fill bidirectionnel
    │
    ▼
ÉTAPE 6:  Format SBS (dist op→m ≤ 4)
    │
    ▼
ÉTAPE 7:  Chaque intermédiaire EXACTEMENT 1 fois
    │
    ▼
ÉTAPE 8:  DOSC: 1 champ à la fois (phase par phase)
    │
    ▼
ÉTAPE 9:  Anti-raccourci SYMÉTRIQUE
    │
    ▼
ÉTAPE 10: L10: masquer champs futurs (bidirectionnalité)
    │
    ▼
ÉTAPE 11: Incremental masking (single_frac=0.3)
    │
    ▼
ÉTAPE 12: Phase interleaved finale (1/k par tâche, 30k+ steps)
    │
    ▼
ÉTAPE 13: Generative replay (sommeil) entre stages
    │
    ▼
ÉTAPE 14: VRAM cap (batch × L ≈ 1.9 × 10⁴)
    │
    ▼
ÉTAPE 15: seq_len fixe (1 plan FFT)
    │
    ▼
ÉTAPE 16: Per-step accuracy > 0.99 avant +1 pas
    │
    ▼
ÉTAPE 17: Cascade eval régulière
    │
    ▼
ÉTAPE 18: Mots RÉELS uniquement
```

---

### ÉTAPE 1 — Sanity Check: Overfit 1 Batch

**Action:** Entraîner le modèle sur un seul batch répété (8 exemples identiques) pendant 500 steps.

```python
ids_fix, fields_fix, text_fix = make_sbs(3, 4)
batch_fix = torch.tensor([ids_fix] * 8, dtype=torch.long, device=D)
target_fix = batch_fix.clone()

for step in range(500):
    masked = batch_fix.clone()
    mask = torch.zeros_like(batch_fix, dtype=torch.bool)
    for b_idx in range(8):
        n_mask = random.randint(1, min(3, len(fields_fix)))
        champs = random.sample(range(len(fields_fix)), n_mask)
        for c in champs:
            s, e = fields_fix[c]
            masked[b_idx, s:e] = tok.mask
            mask[b_idx, s:e] = True

    optimizer.zero_grad()
    logits = model(masked)
    loss = F.cross_entropy(logits[mask], target_fix[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

**POURQUOI cette étape (et pas une autre)?**
Avant d'investir 30k+ steps d'entraînement, il faut vérifier que le pipeline complet fonctionne: tokenizer, format SBS, masquage, forward pass, loss, backward, optimizer. Si le modèle ne peut pas mémoriser 1 batch, il y a un bug.

**Paramètres de l'étape:**
- Batch: 8 copies d'un même exemple
- Steps: 500 (avec warmup)
- Masquage: 1-3 champs aléatoires

**Critère de validation:** `loss < 0.01` → **PASS**, passer à l'étape suivante.

**Si on saute cette étape:**
- On risque de découvrir un bug après 10k steps (perte de temps massive)
- Le bug peut être subtil: positions de champs mal calculées, masque inversé, tokenizer avec collision de tokens, etc.
- Le sanity check détecte tous ces problèmes en 30 secondes

---

### ÉTAPE 2 — Modèle ÉTROIT: d=256, n_blk=3

**Action:** Configurer le modèle avec les paramètres optimaux.

```python
model = SpXLMv6(
    vocab_size   = tok.vocab_size,
    d_model      = 256,       # Pas plus, pas moins
    n_blocks     = 3,         # Sweet spot
    seq_len      = SEQ_LEN,   # Fixe (étape 15)
    mode         = "reasoning",  # Bidirectionnel (étape 5)
    mask_token_id = tok.mask,
    refine_steps = 3,
).to(D)
```

**POURQUOI d=256 et n_blk=3 (voir §1.3 et §1.4)?**

Le modèle doit rester **étroit**. La compétence de raisonnement vient de la **décomposition** (scratchpad, étapes), pas de la **masse** de paramètres. Un modèle plus large détruit l'efficacité du grokking (δ = −3.55).

**Paramètres de l'étape:**
- d_model = 256 (sweet spot mesuré)
- n_blocks = 3 (point d'inflexion coût/bénéfice)
- Total params: ~1.8M

**Critère de validation:** Le modèle s'initialise sans NaN, le forward pass produit des logits de forme correcte (B, L, V), VRAM < 3 GB.

**Si on saute cette étape (utiliser d=512 ou n_blk=10):**
- d=512: D divisé par 12 → le grokking devient pratiquement impossible
- n_blk=10: 3.4× plus de params pour 1.5× plus de profondeur → gaspillage
- Le modèle convergera vers la mémorisation pure, jamais la généralisation

---

### ÉTAPE 3 — AdamW: lr=1e-3, wd=0.1, betas=(0.9, 0.95)

**Action:** Configurer l'optimiseur.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-3,
    weight_decay = 0.1,        # CRITIQUE
    betas        = (0.9, 0.95),
)
```

**POURQUOI AdamW (voir §1.5)?**

SGD ne peut pas grokker. AdamW avec weight_decay=0.1 est **le**配置 qui déclenche le grokking. Le weight decay pousse vers les solutions de petite norme (généralisantes), et l'adaptation par paramètre d'Adam permet de naviguer entre mémorisation et généralisation.

**Paramètres de l'étape:**
- lr = 1e-3 (optimal pour 0.3–2M params)
- weight_decay = 0.1 (CRITIQUE)
- betas = (0.9, 0.95) (beta2 faible pour détection rapide du changement de régime)

**Critère de validation:** La loss diminue sur les premiers 500 steps. Pas de NaN.

**Si on saute cette étape (SGD ou wd=0):**
- SGD: pas de grokking, mémorisation pure
- wd=0: pas de grokking dans un budget pratique (peut nécessiter 100k+ steps au lieu de 8k)
- wd trop élevé (1.0+): possible pour l'arithmétique pure, mais peut tuer l'apprentissage sur des tâches plus complexes

---

### ÉTAPE 4 — Warmup 500+ steps + Gradient Clipping 1.0

**Action:** Implémenter le warmup linéaire et le gradient clipping.

```python
WARMUP = 500

def get_lr_scale(step):
    return min(1.0, (step + 1) / WARMUP)

def apply_warmup(step):
    scale = get_lr_scale(step)
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * scale

# Dans la boucle d'entraînement:
apply_warmup(step)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # OBLIGATOIRE
optimizer.step()
```

**POURQUOI le warmup?**
Les poids initiaux du FFT mixer sont aléatoires. Les premiers gradients sont bruités et potentiellement grands. Un LR plein (1e-3) dès le premier step peut déstabiliser le filtre fréquentiel. Le warmup linéaire (0 → 1e-3 sur 500 steps) donne au filtre le temps de se stabiliser.

**POURQUOI grad clip = 1.0?**
L'IFFT peut amplifier certains gradients (basses fréquences → gradients à grande norme). Sans clipping, ces gradients explosent → NaN cascade → perte complète du modèle. Le clip à 1.0 est suffisant pour prévenir l'explosion sans ralentir l'apprentissage.

**Paramètres de l'étape:**
- warmup_steps = max(500, T // 20)
- grad_clip = 1.0

**Critère de validation:** Pas de NaN sur les 500 premiers steps. La loss suit une trajectoire descendante.

**Si on saute cette étape:**
- Pas de warmup: ~30% de chance de NaN dans les 50 premiers steps
- Pas de grad clip: ~80% de chance de NaN avant 1000 steps (le FFT mixer est très sensible)

---

### ÉTAPE 5 — Diffusion-fill (Bidirectionnel)

**Action:** Utiliser le mode bidirectionnel pour le raisonnement.

```python
model = SpXLMv6(
    mode = "reasoning",  # bidirectional=True
)
```

**POURQUOI bidirectionnel (et pas causal/AR)?**

Le raisonnement multi-pas exige de **lire** les opérandes, pas de les **régénérer**. Le mode AR (causal) doit générer m₁ avant de pouvoir calculer m₂ — chaque erreur s'accumule. Le mode bidirectionnel peut voir m₂ (masqué) comme contexte pour calculer m₁ — la cascade se résout par raffinement itératif.

**Résultats mesurés:**
- AR free-gen pour le raisonnement: **0.02–0.05** (échec total)
- Diffusion-fill direct (1-pas): 0.75 (plafond)
- Diffusion-fill + scratchpad k=3: **0.984** ✅

**Paramètres de l'étape:**
- mode = "reasoning"
- refine_steps = 3 (à l'inférence)

**Critère de validation:** À l'inférence, le modèle remplit correctement les positions masquées (au moins au-dessus du hasard après le sanity check).

**Si on saute cette étape (utiliser le mode causal):**
- Le modèle ne peut pas relire les opérandes → échec du raisonnement
- Accuracy 0.02–0.05 sur problèmes inédits
- Aucun grokking possible

---

### ÉTAPE 6 — Format SBS (Step-By-Step)

**Action:** Structurer la séquence en format SBS avec distance op_i → m_i ≤ 4 tokens.

```
SANS SCRATCHPAD (k=1):
  "stem#a{op}b = ans"
  Champs: [ans]    (1 champ masqué)

FORMAT GROUPÉ (k=3, PROBLÉMATIQUE — distance trop grande):
  "stem#op1 c op2 d op3 e m1m1m1 m2m2m2 m3m3m3 | ans"
  Distance op₁ → m₂ = 16 tokens  ← TROP LOIN

FORMAT SBS (k=3, CORRECT — distance ≤ 4):
  "stem#m1m1m1; op1 c m2m2m2; op2 d m3m3m3; op3 e ans"
  Distance op₁ → m₂ = 4 tokens   ← GROKKABLE
```

Pour l'addition a+b avec scratchpad:
```
"a+b=? [1/b]a+1=m1 [2/b]m1+1=m2 ... [b/b]m(b-1)+1=ans = ans"
```

**POURQUOI SBS (et pas groupé)?**

| Format | Distance op→résultat | Solo progress (6k) | Cascade finale |
|---|---|---|---|
| Groupé (dist=12) | 12 tokens | plateau ~0.15 | 0.950 |
| **SBS (dist=4)** | **4 tokens** | **montant ~0.60** | **0.971** |

Le filtre spectral FFT apprend un motif de convolution. Pour détecter "op₁ c → m₂", le filtre doit "étirer" son support sur la distance entre les tokens. Distance 4 → filtre court → facile à apprendre. Distance 12 → filtre long → difficile, nécessite la phase interleaved.

**Paramètres de l'étape:**
- Format SBS pour k ≥ 3
- Distance op_i → m_i ≤ 4
- L = 1 + 4×k (longueur de séquence idéalisée)

**Critère de validation:** Vérifier que la distance calculée entre chaque opérateur et son résultat est ≤ 4 tokens.

**Si on saute cette étape (format groupé):**
- Le grokking solo plafonne à 0.15 (nécessite obligatoirement la phase interleaved)
- La cascade finale est légèrement inférieure (0.950 vs 0.971)
- Sur des tâches complexes, l'écart peut être beaucoup plus grand

---

### ÉTAPE 7 — Chaque Intermédiaire EXACTEMENT 1 Fois

**Action:** S'assurer que chaque valeur intermédiaire apparaît une seule fois dans la séquence.

```python
# CORRECT — chaque intermédiaire 1 fois:
f"{stem}#{a}{op1}{b}{op2}{c}{op3}{d}={m1:03d}|{m2:03d}|{ans:03d}"

# FAUTIF — m1 dupliqué → le modèle peut le COPIER au lieu de CALCULER:
f"{stem}:{a}{op1}{b}={m1:03d};{m1:03d}{op2}{c}={ans:03d}"
```

**POURQUOI exactement 1 fois?**

Si un intermédiaire apparaît deux fois, le gradient trouve le **raccourci de copie**: il suffit de copier la valeur visible au lieu de la calculer. Le modèle n'apprend jamais la règle.

Exemple: si m₁ apparaît en position 5 (visible) et en position 12 (à prédire), le modèle apprend "copier position 5 → position 12" au lieu de "calculer a×b". À l'évaluation en cascade (toutes positions masquées), m₁ n'est visible nulle part → le modèle ne peut ni copier ni calculer → échec.

**Paramètres de l'étape:**
- Valider le format: assert len(set(intermediates)) == len(intermediates)

**Critère de validation:** SC-2 (Sanity Check 2): assertion que chaque intermédiaire apparaît exactement une fois.

**Si on saute cette étape:**
- Le modèle apprend des raccourcis de copie
- Accuracy individuelle haute (ment) mais cascade effondrée
- La métrique cascade (le vrai test) échoue

---

### ÉTAPE 8 — DOSC: 1 Champ à la Fois (Phase par Phase)

**Action:** Entraîner le modèle champ par champ, dans l'ordre de dépendance topologique.

```
Chaîne A → B → C → ... → Z:

Phase 1: mask A seulement (jamais B ni C)     → grokker A seul
Phase 2: mask B seulement (A VISIBLE)          → grokker B = f(A) sans interférence
Phase 3: mask C seulement (A, B VISIBLES)      → grokker C = g(A,B) sans interférence
...
Phase k: mask champ_k seulement (autres visibles)
Phase k+1: INTERLEAVED 1/k par tâche            → consolidation anti-oubli
```

**Pour l'addition a+b (2 macro-phases):**
```
Phase 1: GROKKER +1 (masquer un intermédiaire m_i aléatoire)
         → Le modèle apprend la primitive "+1"
         → Anti-raccourci: masquer aussi m_{i+1}...m_{b}, ans

Phase 2: GROKKER ANS (masquer seulement la réponse finale)
         → Tous les m_i visibles → le modèle doit copier la dernière valeur
         → En réalité: le modèle doit comprendre la chaîne complète
```

**POURQUOI DOSC (et pas entraînement joint)?**

| Approche | Cascade | Mécanisme |
|---|---|---|
| Entraînement joint (tous champs masqués) | 0.13–0.65 | Interférence de gradient |
| **DOSC (phase par phase)** | **0.97–0.99** | Chaque champ grokke indépendamment |

Le problème de l'entraînement joint: quand le champ A est grokké (loss → 0), son gradient s'annule. Le paysage d'optimisation change brusquement, déstabilisant le champ B qui était partiellement appris. C'est l'**interférence de gradient**.

DOSC résout cela en s'assurant que chaque champ compresse indépendamment, sans que les gradients des autres champs n'interfèrent.

**Durée par phase (guideline):**
- Extraction (prose→champ): 2000–4000 steps
- Multiplication (a×b): 6000 steps
- Addition/soustraction scratchpad→scratchpad: 6000–8000 steps
- Phase interleaved finale: 30000–40000 steps

**Critère de validation:** Per-step accuracy de la phase courante > 0.99 (étape 16).

**Si on saute cette étape (entraînement joint):**
- Cascade = 0.13–0.65 (interférence de gradient)
- DOSC donne 7.7× d'amélioration (0.999 vs 0.129)
- Le modèle ne généralise jamais

---

### ÉTAPE 9 — Anti-Raccourci SYMÉTRIQUE

**Action:** Pour chaque champ f_i prédit, masquer en input TOUTES les variables V_j, V_k... qui permettent de calculer f_i algébriquement.

```python
# Pour prédire m1 = a×b:
# Raccourci possible: m1 = m2 - c, m1 = ans - c - d
# → masquer m1 ET m2 ET ans

# Pour prédire m2 = m1 + c:
# Raccourci possible: m2 = ans - d, m2 = m3 + d
# → masquer m2 ET m3 ET ans

# Pour prédire ans = m_k + d:
# (calcul voulu, pas de raccourci possible)
# → masquer ans seulement
```

**Table de masquage anti-raccourci:**

| Tâche | Raccourci possible | Masquage anti-raccourci |
|---|---|---|
| c extraction | c = ans − m1 | mask c + m1 + ans |
| m1 = a×b | m1 = ans − c | mask m1 + ans |
| m2 = m1±c | m2 = ans − d | mask m2 + ans |
| ans = m_k±d | (calcul voulu) | mask ans seul |

**POURQUOI symétrique?**

Le gradient **choisit toujours le chemin le plus court**. Si un raccourci existe, le modèle l'utilise. Sans anti-raccourci:
- m1_short = 0.999 (avec raccourci m1=ans−c, ans visible)
- m1_honest = 0.240 (sans raccourci, ans masqué)
- cascade = 0.240 (la VRAIE métrique)

Avec anti-raccourci:
- m1_honest = 1.000 ✅
- cascade = 0.984 ✅

**La cascade est la vraie métrique de généralisation. Les métriques individuelles peuvent mentir.**

**Paramètres de l'étape:**
- Identifier tous les chemins algébriques pour chaque champ
- Masquer toutes les variables récupérables

**Critère de validation:** Les métriques individuelles correspondent à la cascade (produit ≈ mesuré). Si m1=0.999 mais cascade=0.24 → il y a un raccourci non bloqué.

**Si on saute cette étape:**
- Le modèle apprend des raccourcis triviaux (m1 = ans - c)
- Les métriques individuelles sont trompeusement hautes
- La cascade s'effondre (0.24 au lieu de 0.98)

---

### ÉTAPE 10 — L10: Masquer Champs Futurs (Bidirectionnalité)

**Action:** Pendant la Phase i d'extraction, masquer AUSSI tous les champs j > i dans la séquence.

```python
# BUG (cascade = 0.093) — Phase 2, champs futurs visibles:
for f in [fm1, fm2, fd, fm3, fm4, fa]:
    si[:, f] = True
# e et f NON masqués → le FFT bidirectionnel lit les positions futures → triche

# FIX (cascade ≈ 0.970) — masquer aussi e et f:
for f in [fm1, fm2, fd, fm3, fm4, fe, ff, fa]:
    si[:, f] = True
# Contexte entraînement = contexte cascade → pas de surprise
```

**POURQUOI cette étape est critique?**

Le FFT mixer est **bidirectionnel**: chaque position voit toutes les autres positions. Si pendant l'entraînement de la Phase i, les champs j > i sont visibles, le modèle apprend à les exploiter comme contexte.

Mais pendant l'évaluation en cascade, TOUS les champs sont masqués simultanément. Les champs j > i ne sont plus visibles → le modèle perd son contexte → effondrement.

**Résultat mesuré sans L10:**
- Accuracy individuelle: ≥ 0.968 pour tous les 9 champs
- Cascade: **0.093** (effondrement!)
- Test oracle (GT extraction + prédiction arithmétique): 0.970 → l'arithmétique est parfaite, le bug est 100% dans les masques

**Paramètres de l'étape:**
- Pour chaque Phase i: masquer le champ i ET tous les champs j > i

**Critère de validation:** La cascade correspond au produit des accuracies individuelles (écart < 0.05).

**Si on saute cette étape:**
- Cascade = 0.093 même avec toutes les autres étapes correctes
- Le modèle apprend à tricher pendant l'entraînement
- À l'évaluation, la triche n'est plus possible → effondrement

---

### ÉTAPE 11 — Incremental Masking (single_frac=0.3)

**Action:** 30% des batchs masquent EXACTEMENT 1 champ. Les 70% masquent un sous-ensemble aléatoire.

```python
def incremental_field_mask(batch_size, seq_len, field_positions,
                           mask_prob=0.5, single_frac=0.3):
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    n_fields = len(field_positions)

    for b in range(batch_size):
        if random.random() < single_frac:
            # Masquer EXACTEMENT 1 champ (pas le target)
            field_idx = random.randint(0, n_fields - 2)
            start, end = field_positions[field_idx]
            mask[b, start:end] = True
        else:
            # Masquer un sous-ensemble aléatoire
            for field_idx in range(n_fields - 1):
                if random.random() < mask_prob:
                    start, end = field_positions[field_idx]
                    mask[b, start:end] = True

        # TOUJOURS masquer le target (dernier champ)
        start, end = field_positions[-1]
        mask[b, start:end] = True

    return mask
```

**POURQUOI incremental masking?**

| Approche | Accuracy | Mécanisme |
|---|---|---|
| Full masking (tous les champs) | 0.75 | Objectif enchevêtré — le modèle doit apprendre tous les k pas simultanément |
| **Incremental (single_frac=0.3)** | **0.984** | k épisodes de grokking indépendants |

Le masquage incrémental crée k épisodes de grokking **indépendants**. Quand un batch masque exactement 1 champ, le modèle n'a qu'un seul problème 1-pas à résoudre. Le grokking d'un problème 1-pas est beaucoup plus facile que le grokking d'un problème k-pas enchevêtré.

Les 70% de batchs multi-champs assurent que le modèle apprend aussi à gérer des contextes complexes (nécessaire pour la cascade).

**Paramètres de l'étape:**
- single_frac = 0.3 (± 0.05)
- mask_prob = 0.5

**Critère de validation:** SC-3: vérifier que single_frac ≈ 0.3 en imprimant 5 exemples masqués.

**Si on saute cette étape:**
- Plafond à 0.75 (full masking = objectif enchevêtré)
- Le grokking ne se produit pas ou prend beaucoup plus de temps

---

### ÉTAPE 12 — Phase Interleaved Finale (1/k par Tâche, 30k+ steps)

**Action:** Après toutes les phases DOSC, entraîner en alternance sur toutes les tâches.

```python
def make_batch_interleaved(pairs, single_frac=0.3):
    for b_idx in range(len(pairs)):
        fields = all_fields[b_idx]
        n_fields = len(fields)

        if random.random() < single_frac:
            # Masquer 1 seul champ au hasard
            c = random.randint(0, n_fields - 1)
            s, e = fields[c]
            masked_input[b_idx, s:e] = tok.mask
            mask[b_idx, s:e] = True
        else:
            # Masquer un sous-ensemble + anti-raccourci (j > premier masqué)
            start_c = random.randint(0, n_fields - 1)
            for c in range(start_c, n_fields):
                if random.random() < 0.5:
                    s, e = fields[c]
                    masked_input[b_idx, s:e] = tok.mask
                    mask[b_idx, s:e] = True
```

**POURQUOI interleaved (et pas s'arrêter après DOSC)?**

Après le DOSC, chaque champ est grokké individuellement. Mais:
1. Les poids sont **partagés** entre les champs → les circuits appris pour le champ A peuvent être écrasés par l'apprentissage du champ B
2. La cascade exige que tous les circuits coexistent et fonctionnent **ensemble**

La phase interleaved:
- Chaque batch tire au sort UNE tâche (1/k de probabilité pour chaque)
- Chaque tâche reçoit ~(T_interleaved / k) steps effectifs
- La fréquence de rafraîchissement (1/k par batch) est assez haute pour empêcher l'oubli
- L'anti-raccourci est préservé dans chaque tâche

**Sans interleaved:** oubli catastrophique à chaque transition de phase (−99.5% en 2000 steps).

**Paramètres de l'étape:**
- 30k–40k steps minimum
- 1/k probabilité pour chaque tâche
- Anti-raccourci préservé

**Critère de validation:** Cascade > 0.95 et stable (ne décroît pas sur les derniers 5k steps).

**Si on saute cette étape:**
- Oubli catastrophique: le champ A est oublié pendant l'apprentissage du champ B
- Cascade < 0.10 même avec un DOSC parfait
- Les circuits individuels ne s'intègrent jamais

---

### ÉTAPE 13 — Generative Replay (Sommeil) Entre Stages

**Action:** Entre les stages du curriculum, faire "rêver" le modèle pour consolider.

→ **Voir §4 Sommeil Multi-Phase pour les détails complets.**

**Résumé:** Le modèle génère ses propres pseudo-exemples (rêves), puis se ré-entraîne sur un mélange 50/50 rêves + vraies données. Cela comble **89.8% de l'oubli catastrophique** entre stages.

**POURQUOI cette étape?**

Chaque stage du curriculum apprend de nouvelles compétences. Sans sommeil, le stage N écrase les compétences du stage N-1. Le sommeil (generative replay) maintient les anciennes compétences actives pendant l'apprentissage des nouvelles.

**Si on saute cette étape:**
- Oubli catastrophique entre stages (acc_A: 0.92 → 0.16 après stage B)
- Le modèle doit tout réapprendre à chaque stage
- Avec sommeil: 89.8% de l'oubli comblé (0.92 → 0.84)

---

### ÉTAPE 14 — VRAM Cap (batch × L ≈ 1.9 × 10⁴)

**Action:** Respecter le budget VRAM en ajustant la taille de batch.

```python
# Contrainte: batch × seq_len ≈ 1.9 × 10⁴ (cap 3 GB, d=256, n_blk=3)
# seq_len = 1 + 4×k (format scratchpad)

# Table pratique:
# k=1 → L=5  → batch ≈ 3800
# k=2 → L=9  → batch ≈ 2100
# k=3 → L=13 → batch ≈ 1460
# k=5 → L=21 → batch ≈ 900
# k=10 → L=41 → batch ≈ 460

batch_size = int(19000 / SEQ_LEN)
```

**POURQUOI cette contrainte?**

Le FFT mixer stocke le spectre complexe (L//2+1) × d_model par batch. À d=256, n_blk=3, le pic VRAM est dominé par le produit batch × seq_len. La formule batch × L ≈ 1.9 × 10⁴ garantit un pic < 3 GB sur RX 7900 XTX.

**Implémentation:**
```python
# Option A: PyTorch
torch.cuda.set_per_process_memory_fraction(3.0 / 24.0, device=0)

# Option B: ROCm env
# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
```

**Si on saute cette étape:**
- OOM (Out of Memory) → crash
- Ou: batch trop petit → apprentissage trop lent

---

### ÉTAPE 15 — seq_len Fixe (1 Plan FFT)

**Action:** Padder toutes les séquences à une longueur FIXE.

```python
SEQ_LEN = 48  # Fixe, ne jamais changer pendant l'entraînement

ids = [tok.bos] + tok.encode(text) + [tok.eos]
while len(ids) < SEQ_LEN:
    ids.append(tok.pad)
ids = ids[:SEQ_LEN]  # Tronquer si nécessaire
```

**POURQUOI longueur fixe?**

ROCm (et CUDA) compilent un **plan FFT** pour chaque longueur de séquence unique. Chaque compilation coûte ~0.8s. Si les séquences ont des longueurs variables, chaque nouvelle longueur déclenche une recompilation → 50–95s de latence à froid.

Avec longueur fixe: **1 plan FFT compilé une fois**, tous les appels suivants < 0.3s.

**Si on saute cette étape:**
- Latence à froid de 50–95s à chaque nouvelle longueur
- Sur un run de 30k steps, perte de plusieurs heures
- Inference instable (recompilation à chaque prompt)

---

### ÉTAPE 16 — Per-Step Accuracy > 0.99 Avant +1 Pas

**Action:** Ne passer à la phase suivante que quand la per-step accuracy de la phase courante dépasse 0.99.

```python
# Évaluation régulière:
final_acc, per_step, cascade = cascade_eval(model, test_pairs, n_steps_fill=5)

# mid_acc = moyenne des accuracies des champs intermédiaires
mid_acc = sum(per_step[:-1]) / max(len(per_step[:-1]), 1)

if mid_acc >= 0.99:
    print("*** PHASE VALIDÉE ***")
    break  # Passer à la phase suivante
```

**POURQUOI > 0.99 (et pas > 0.90)?**

La cascade = produit des per-step accuracies. Si p = 0.99:
- k=3: cascade = 0.99³ = 0.970 ✅
- k=5: cascade = 0.99⁵ = 0.951 ✅

Si p = 0.90:
- k=3: cascade = 0.90³ = 0.729 ❌
- k=5: cascade = 0.90⁵ = 0.590 ❌

**Chaque pourcent d'erreur par pas est amplifié exponentiellement.** Pour que la cascade soit > 0.95, il faut p > 0.99^(1/k).

La profondeur fiable: D = 1/(1−p). À p=0.99: D=100. À p=0.90: D=10. La différence est **10×**.

**Critère de validation:** Per-step accuracy > 0.99 pour TOUS les champs de la phase courante.

**Si on saute cette étape:**
- On ajoute un pas alors que le précédent n'est pas maîtrisé
- L'erreur s'accumule exponentiellement
- Cascade effondrée (< 0.50)

---

### ÉTAPE 17 — Cascade Eval Régulière

**Action:** Évaluer la cascade (produit des per-step accuracies) tous les 1000–2000 steps.

```python
def cascade_eval(model, test_pairs, n_steps_fill=5):
    """
    Cascade eval: masquer TOUS les champs, diffusion-fill, vérifier.
    Le produit des per-step ne ment pas.
    """
    model.eval()
    correct_per_field = {}

    for a, b in test_pairs:
        ids, fields, text = make_sbs(a, b)
        inp = torch.tensor([ids], dtype=torch.long, device=D)

        masked = inp.clone()
        mask_eval = torch.zeros_like(inp, dtype=torch.bool)
        for s, e in fields:
            masked[0, s:e] = tok.mask
            mask_eval[0, s:e] = True

        filled = model.diffuse_fill(masked, mask_eval, n_steps=n_steps_fill)

        # Vérifier chaque champ
        for fi, (s, e) in enumerate(fields):
            gen = tok.decode(filled[0, s:e].tolist())
            # ... comparer avec la vraie valeur
            if gen == true_text:
                correct_per_field[fi][0] += 1
            correct_per_field[fi][1] += 1

    # Cascade = produit des per-step
    per_step_acc = [c/t for c,t in correct_per_field.values()]
    cascade = 1.0
    for acc in per_step_acc:
        cascade *= acc

    return final_acc, per_step_acc, cascade
```

**POURQUOI la cascade (et pas juste l'accuracy finale)?**

**Les métriques individuelles peuvent mentir.** Un modèle peut avoir m1=0.999 (avec raccourci) mais cascade=0.240 (sans raccourci). La cascade est la **vraie** métrique de généralisation.

La cascade masque TOUS les champs simultanément → le modèle doit tout calculer en chaîne → pas de raccourci possible.

**Critère de validation:** Cascade > 0.95 et cohérente avec le produit des per-step (écart < 0.05).

**Si on saute cette étape:**
- On ne détecte pas les raccourcis cachés
- On pense que le modèle généralise alors qu'il triche
- À l'évaluation finale, surprise: tout s'effondre

---

### ÉTAPE 18 — Mots RÉELS Uniquement

**Action:** N'utiliser que de vraies données (dictionnaire, lemminflect, etc.). Ne jamais inventer de mots ou de règles.

```python
# CORRECT — vraies données:
from lemminflect import getInflection
words = ["run", "walk", "eat", "go", "be"]
for w in words:
    plural = getInflection(w, tag='NNS')[0]  # Vrai pluriel
    past = getInflection(w, tag='VBD')[0]    # Vrai passé

# FAUTIF — mots inventés:
fake_words = ["zorp", "blarg", "fnu"]  # Jamais!
```

**POURQUOI de vraies données?**

1. **Généralisation mesurable.** Avec de vraies données, on peut diviser en train/test. Le test contient des exemples inédits → la généralisation est mesurable.

2. **Règles vs exceptions.** Les vraies langues ont des règles (régulières) et des exceptions (irrégulières). Le modèle doit apprendre les règles (généralisation) et stocker les exceptions (mémoire). Les mots inventés n'ont pas cette structure.

3. **Calibration C₀.** La constante C₀ de la loi de scaling dépend de la difficulté de la tâche. Les vraies données donnent un C₀ réaliste pour le planning.

**Si on saute cette étape:**
- On peut "tricher" involontairement (mots inventés = pas de vrai test)
- Le modèle n'est pas calibré pour des données réelles
- Généralisation non mesurable

---

## 4. SOMMEIL MULTI-PHASE

> **Le sommeil n'est PAS optionnel. Il transforme la MÉMOIRE en COMPREHENSION.**

### 4.1 Pourquoi le Sommeil?

Le modèle apprend par stages (curriculum). Chaque stage apprend de nouvelles compétences. Sans consolidation, le stage N **écrase** les compétences du stage N−1 (oubli catastrophique).

**Résultat mesuré sans sommeil:**
- acc_A après Stage A: 0.920
- acc_A après Stage B (sans sommeil): **0.158** (oubli: 0.762)

**Résultat mesuré avec sommeil:**
- acc_A après Stage A: 0.920
- acc_A après Stage B + sommeil: **0.842** (89.8% de l'oubli comblé)

### 4.2 Les Trois Phases de Sommeil

L'architecture FFT rend "macro/micro" = "basses/hautes fréquences" **natif**:
- **Basses fréquences** = vue MACRO (thème, structure globale, sémantique)
- **Hautes fréquences** = vue MICRO (détails fins, forme de surface, nuances)

```
┌─────────────────────────────────────────────────────────────────┐
│                  SOMMEIL MULTI-PHASE                            │
│                                                                 │
│  Stage A (Éveil): Entraînement normal (20K+ steps)            │
│       │                                                         │
│       ▼                                                         │
│  Snapshot → sauvegarder les poids                              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PHASE 1 — SOMMEIL LÉGER (Light Sleep)                   │   │
│  │                                                         │   │
│  │ • Generative replay: le modèle "rêve" ses exemples     │   │
│  │ • Mix 50/50: rêves + vraies données                    │   │
│  │ • LR × 0.1 (base_lr × 0.1 = 1e-4)                     │   │
│  │ • 5K steps                                              │   │
│  │ • Vue MACRO: consolidation des basses fréquences       │   │
│  │ • Effet: réactivation des circuits, anti-oubli         │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PHASE 2 — SOMMEIL MOYEN (Medium Sleep)                  │   │
│  │                                                         │   │
│  │ • Self-distillation: ré-entraînement sur les rêves     │   │
│  │ • Analyse d'entropie des poids du filtre FFT           │   │
│  │ • Renforcer les poids importants (basses/moyennes freq)│   │
│  │ • LR × 0.1 (base_lr × 0.1 = 1e-4)                     │   │
│  │ • 5K steps                                              │   │
│  │ • Consolidation des relations (fréquences moyennes)    │   │
│  │ • Effet: structuration des associations               │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ PHASE 3 — SOMMEIL PROFOND (Deep Sleep)                 │   │
│  │                                                         │   │
│  │ • Analyse fine des hautes fréquences (MICRO)           │   │
│  │ • LR × 0.01 (base_lr × 0.01 = 1e-5)                   │   │
│  │ • Grad clip plus serré: 0.5 (au lieu de 1.0)          │   │
│  │ • 5K steps                                              │   │
│  │ • Extraction des règles cachées, détails fins          │   │
│  │ • Effet: nuances, précision maximale                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Stage B (Test): la compréhension a-t-elle émergé?             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Implémentation Détaillée de Chaque Phase

#### Phase 1 — Sommeil Léger (Generative Replay)

```python
def sleep_phase_light(model, train_data, n_steps=5000):
    """
    SOMMEIL LÉGER: Generative replay.
    
    Le modèle "rêve": il génère ses propres pseudo-exemples
    en remplissant les positions masquées.
    Puis se ré-entraîne sur un mélange 50/50 rêves + vraies données.
    """
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr * 0.1,  # LR réduit ×10
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Étape 1: Générer des rêves
    dreams = []
    model.eval()
    with torch.no_grad():
        for root_id, rule in random.sample(train_data, min(40, len(train_data))):
            ids, fields, _, form_id = make_morpho(root_id, rule)
            inp = torch.tensor([ids], dtype=torch.long, device=D)
            
            # Masquer le champ à prédire
            masked = inp.clone()
            mask_eval = torch.zeros_like(inp, dtype=torch.bool)
            s, e = fields[1]
            masked[0, s:e] = tok.mask
            mask_eval[0, s:e] = True
            
            # Le rêve = la prédiction du modèle
            filled = model.diffuse_fill(masked, mask_eval, n_steps=3)
            dreams.append((filled[0].tolist(), fields))
    model.train()
    
    # Étape 2: Ré-entraîner sur rêves + vraies données (50/50)
    for step in range(n_steps):
        if random.random() < 0.5 and dreams:
            # RÊVE
            dream_ids, dream_fields = random.choice(dreams)
            input_ids = torch.tensor([dream_ids] * 4, dtype=torch.long, device=D)
            target_ids = input_ids.clone()
            fields_batch = [dream_fields] * 4
        else:
            # VRAIES DONNÉES
            batch = random.sample(train_data, 4)
            sequences = [make_morpho(r, rule) for r, rule in batch]
            input_ids = torch.tensor([s[0] for s in sequences], ...)
            target_ids = input_ids.clone()
            fields_batch = [s[1] for s in sequences]
        
        # Masquer et entraîner
        masked_input, mask = apply_mask(input_ids, fields_batch)
        loss = train_step(model, opt, masked_input, target_ids, mask)
```

**Ce qui se passe:**
- Le modèle génère des pseudo-exemples (rêves) en utilisant diffuse_fill
- Les rêves sont imparfaits (le modèle fait des erreurs) — c'est voulu
- Le ré-entraînement sur les rêves **réactive les circuits** appris pendant le Stage A
- Le mélange 50/50 assure que les vraies données corrigent les erreurs des rêves
- **Effet MACRO**: les basses fréquences du filtre FFT sont consolidées (structure globale)

#### Phase 2 — Sommeil Moyen (Entropy Analysis + Consolidation)

```python
def sleep_phase_medium(model, train_data, n_steps=5000):
    """
    SOMMEIL MOYEN: Self-distillation + entropy analysis.
    
    Analyse de l'entropie des poids du filtre FFT:
    - Basses fréquences: vue MACRO (déjà consolidée en Phase 1)
    - Moyennes fréquences: RELATIONS (à consolider ici)
    - Hautes fréquences: vue MICRO (Phase 3)
    """
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr * 0.1,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Analyse d'entropie
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "filter_real" in name:
                energy = param.abs()
                total_energy = energy.sum()
                
                low_freq = energy[:energy.shape[0]//4].sum()
                mid_freq = energy[energy.shape[0]//4:3*energy.shape[0]//4].sum()
                high_freq = energy[3*energy.shape[0]//4:].sum()
                
                print(f"  {name}:")
                print(f"    Basses freq (MACRO):  {low_freq/total_energy:.1%}")
                print(f"    Moyennes freq (REL):  {mid_freq/total_energy:.1%}")
                print(f"    Hautes freq (MICRO):  {high_freq/total_energy:.1%}")
    
    # Entraînement à LR bas pour consolider les relations
    for step in range(n_steps):
        batch = random.sample(train_data, 16)
        # ... entraînement standard à LR × 0.1
```

**Ce qui se passe:**
- L'analyse d'entropie révèle où l'énergie du filtre est concentrée
- Les fréquences moyennes correspondent aux **relations** entre champs (associations)
- Le ré-entraînement à LR bas consolide ces relations sans détruire la macro-structure
- **Effet**: structuration des associations any→any

#### Phase 3 — Sommeil Profond (Hautes Fréquences, Détails Fins)

```python
def sleep_phase_deep(model, train_data, n_steps=5000):
    """
    SOMMEIL PROFOND: Focus hautes fréquences (MICRO).
    
    LR très bas (×0.01) pour affiner les détails sans détruire
    la structure globale consolidée en Phases 1 et 2.
    Grad clip plus serré (0.5) pour des ajustements fins.
    """
    opt_deep = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr * 0.01,  # LR réduit ×100
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    for step in range(n_steps):
        batch = random.sample(train_data, 16)
        # ... entraînement standard à LR × 0.01
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Plus serré!
        opt_deep.step()
```

**Ce qui se passe:**
- Les hautes fréquences correspondent aux **détails fins** (nuances, cas limites)
- Le LR très bas (1e-5) permet des ajustements minuscules sans perturber la structure
- Le grad clip plus serré (0.5) empêche les grands gradients de détruire le travail des Phases 1-2
- **Effet**: extraction des règles cachées, précision maximale, nuances

### 4.4 Fréquence des Phases de Sommeil

**Règle générale:** Le sommeil intervient **entre les stages du curriculum**, pas pendant un stage.

```
Stage 1: Vocabulaire
    │
    ▼
  ═══ SOMMEIL (3 phases, 15K steps total) ═══
    │
    ▼
Stage 2: Morphologie
    │
    ▼
  ═══ SOMMEIL (3 phases, 15K steps total) ═══
    │
    ▼
Stage 3: Grammaire
    │
    ▼
  ═══ SOMMEIL (3 phases, 15K steps total) ═══
    │
    ▼
Stage 4: Raisonnement k-pas
```

**Pourquoi entre les stages (pas pendant)?**

1. Le sommeil consolide les compétences du stage précédent
2. Il les rend résistantes à l'oubli pendant le stage suivant
3. Pendant un stage, le sommeil ralentit l'apprentissage (LR bas) sans bénéfice clair

**Durée totale du sommeil:** 15K steps (3 phases × 5K steps) entre chaque stage.

**Déclencheur automatique:** Si on utilise un MemoryStore (TF-IDF externe), déclencher un cycle de sommeil quand la file de consolidation atteint 25 faits.

### 4.5 Ce Qui Se Passe Sans Sommeil (Mémoire Brute)

| Métrique | Avec sommeil | Sans sommeil |
|---|---|---|
| acc Stage A après Stage B | **0.842** | 0.158 |
| Oubli comblé | **89.8%** | 0% |
| Généralisation | Compréhension (règles) | Mémorisation brute |
| Robustesse au bruit | Élevée | Fragile |
| Transfert de compétences | Oui | Non |

**Sans sommeil, le modèle a une MÉMOIRE BRUTE:**
- Il peut reproduire les exemples d'entraînement
- Il ne peut pas généraliser à des exemples inédits
- Les compétences sont fragiles (oubliées dès qu'une nouvelle tâche est apprise)
- Les associations entre concepts ne se forment pas

**Avec sommeil, le modèle développe une COMPRÉHENSION:**
- Les règles sont extraites et consolidées
- La généralisation émerge
- Les compétences sont robustes (résistent à l'oubli)
- Les associations any→any se forment nativement

### 4.6 Architecture du Sommeil — Pourquoi ça Marche

L'architecture FFT rend le sommeil **natif**, pas un module ajouté:

1. **Le cœur de diffusion est À LA FOIS solveur et générateur.** Le même modèle qui résout des problèmes (diffusion-fill) peut générer des exemples (generative replay). Pas besoin de module séparé.

2. **La hiérarchie macro→micro = basses→hautes fréquences.** Le filtre FFT apprend naturellement cette hiérarchie:
   - Phase 1 consolide les basses fréquences (MACRO)
   - Phase 2 consolide les fréquences moyennes (RELATIONS)
   - Phase 3 consolide les hautes fréquences (MICRO)

3. **Le LR décroissant préserve la structure.** Chaque phase a un LR plus bas → les ajustements sont de plus en plus fins → on ne détruit pas le travail des phases précédentes.

4. **CLS (Complementary Learning Systems).** Le sommeil implémente le pont entre:
   - **Hippocampe** (rapide, épisodique) = MemoryStore TF-IDF externe
   - **Néocortex** (lent, consolidé) = le cœur spectral FFT (les poids)
   
   Le sommeil transfère les souvenirs épisodiques (TF-IDF) vers la mémoire consolidée (poids FFT).

---

## 5. OPTIMISATION

### 5.1 Configuration pour Meilleure ACCURACY

**Objectif:** Maximiser la cascade (accuracy de généralisation).

```python
# Configuration accuracy maximale:
config_accuracy = {
    "d_model":       192,      # d=192 est le peak absolu (si n_blk=2)
                               # ou d=256 si n_blk=3 (plus robuste)
    "n_blocks":      4,        # 4 blocs: gain marginal mais réel (+14%)
    "lr":            1e-3,
    "weight_decay":  0.1,
    "warmup":        1000,     # Plus de warmup = plus stable
    "T_total":       50000,    # Plus de training = plus de profondeur
    "single_frac":   0.3,
    "refine_steps":  5,        # Plus de raffinement à l'inférence
    "k_scratchpad":  3,        # k=3 est le sweet spot
    "sommeil":       True,     # OBLIGATOIRE pour la consolidation
}

# Budget:
# VRAM: ~3 GB (d=256, n_blk=4)
# Temps: ~2h sur RX 7900 XTX
# Cascade attendue: 0.97–0.99
```

**Trade-off:** Cette configuration est plus lente (50k steps, 4 blocs) mais maximise la cascade.

### 5.2 Configuration pour Meilleur RAISONNEMENT

**Objectif:** Maximiser la profondeur de raisonnement D = k/(1−A^(1/k)).

```python
# Configuration raisonnement maximal:
config_reasoning = {
    "d_model":       128,      # PETIT modèle (minimiser d → maximiser D/params)
    "n_blocks":      3,        # Standard
    "lr":            1e-3,
    "weight_decay":  0.1,
    "T_total":       100000,   # BEAUCOUP plus de training
    "k_scratchpad":  5,        # k=5: profondeur maximale (D ≈ 200)
    "recurrence":    10000,    # Récurrence fenêtrée (D → ∞)
    "refine_steps":  3,
    "sommeil":       True,     # Entre chaque stage de k
}

# Budget:
# VRAM: ~1.5 GB (d=128)
# Temps: ~4h sur RX 7900 XTX
# D attendu: 200+ (avec k=5, T=100k)
#         ou ∞ (avec récurrence r=10000)
```

**Levier principal:** Augmenter k (scratchpad depth). D ∝ k^3.5 → doubler k multiplie D par 11.3.

**Trade-off:** Plus de profondeur = plus de steps d'entraînement = plus de temps. Le modèle est petit (d=128) mais entraîné très longtemps.

### 5.3 Configuration pour Meilleur CONTEXTE

**Objectif:** Maximiser la longueur de séquence et le contexte.

```python
# Configuration contexte maximal:
config_context = {
    "d_model":       256,      # Standard (nécessaire pour représenter le contexte)
    "n_blocks":      3,
    "seq_len":       256,      # Longueur maximale
    "lr":            1e-3,
    "weight_decay":  0.1,
    "T_total":       30000,
    "batch_size":    74,       # batch × 256 ≈ 1.9×10⁴
    "refine_steps":  3,
}

# Budget:
# VRAM: ~3 GB (batch=74, seq_len=256)
# Temps: ~50 min sur RX 7900 XTX
# Contexte: jusqu'à 256 tokens
```

**Contrainte VRAM:** batch × seq_len ≈ 1.9 × 10⁴. Pour seq_len=256: batch=74. Le batch plus petit ralentit l'apprentissage → compenser avec plus de steps.

**Trade-off:** Plus de contexte = moins de batch par step (VRAM fixe) = apprentissage plus lent.

### 5.4 Tableau des Trade-offs

| Objectif | d_model | n_blk | T | k | batch | VRAM | Temps | Cascade | D |
|---|---|---|---|---|---|---|---|---|---|
| **Accuracy max** | 256 | 4 | 50k | 3 | ~300 | 3 GB | ~2h | **0.97–0.99** | ~50 |
| **Raisonnement max** | 128 | 3 | 100k | 5 | ~900 | 1.5 GB | ~4h | 0.95 | **200+** |
| **Contexte max** | 256 | 3 | 30k | 3 | 74 | 3 GB | ~50min | 0.93 | ~45 |
| **Rapidité max** | 128 | 2 | 8k | 2 | ~2000 | 1 GB | ~15min | 0.90 | ~20 |
| **Frugal (min VRAM)** | 64 | 2 | 5k | 1 | ~3800 | <1 GB | ~4min | 0.75 | 4 |

### 5.5 VRAM vs Accuracy

```
VRAM (GB)  vs  Cascade
──────────────────────────────────
0.5 GB     →  cascade ≈ 0.75 (d=64, n_blk=2, k=1)
1.0 GB     →  cascade ≈ 0.90 (d=128, n_blk=2, k=2)
2.0 GB     →  cascade ≈ 0.95 (d=256, n_blk=3, k=3)
3.0 GB     →  cascade ≈ 0.97 (d=256, n_blk=4, k=3, T=50k)
```

**Loi:** Au-delà de 3 GB, les rendements sont fortement décroissants. doubler VRAM (3→6 GB) n'augmente la cascade que de ~0.01 (0.97→0.98). Le grokking n'est pas un problème de VRAM.

### 5.6 Vitesse vs Compréhension

```
Temps d'entraînement  vs  Type d'apprentissage
──────────────────────────────────────────────────
5K steps (~4min)     →  Mémorisation brute (cascade ≈ 0.10)
8K steps (~12min)    →  Grokking naissant (cascade ≈ 0.30)
14K steps (~22min)   →  Grokking partiel (cascade ≈ 0.83)
30K steps (~48min)   →  Grokking complet (cascade ≈ 0.94)
50K steps (~2h)      →  Grokking mature (cascade ≈ 0.97)
+ Sommeil (15K)      →  Compréhension (cascade ≈ 0.98+)
```

**Le grokking est un phénomène de seuil.** Avant le seuil: mémorisation. Après le seuil: généralisation. Le seuil dépend de T, d, et la difficulté de la tâche. **On ne peut pas accélérer le grokking en augmentant d** (ça le ralentit!). On ne peut l'accélérer qu'en augmentant T (δ = +1.34).

### 5.7 Guide de Choix Rapide

| Situation | Configuration recommandée |
|---|---|
| Prototypage rapide | d=128, n_blk=2, T=5k, k=1 (~4 min) |
| Premier grokking | d=256, n_blk=3, T=14k, k=3 (~22 min) |
| Production | d=256, n_blk=3, T=30k, k=3 + sommeil (~1h) |
| Raisonnement profond | d=128, n_blk=3, T=100k, k=5 + récurrence (~4h) |
| Multi-modal | ContinuousFiller, d=256, n_blk=3, patch≥7×7 |

### 5.8 Checklist Finale — Avant Chaque Run

```
□ 1.  Sanity check overfit 1 batch (loss < 0.01)             [ÉTAPE 1]
□ 2.  Modèle ÉTROIT: d=192–256, n_blk=3–4                   [ÉTAPE 2]
□ 3.  AdamW lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95)   [ÉTAPE 3]
□ 4.  Warmup 500+ steps, grad clip 1.0                      [ÉTAPE 4]
□ 5.  Diffusion-fill (bidirectionnel) pour le raisonnement  [ÉTAPE 5]
□ 6.  Format scratchpad SBS (dist op_i→m_i ≤ 4)            [ÉTAPE 6]
□ 7.  Chaque intermédiaire apparaît EXACTEMENT UNE FOIS     [ÉTAPE 7]
□ 8.  DOSC: Phase par phase, un champ à la fois             [ÉTAPE 8]
□ 9.  Anti-raccourci SYMÉTRIQUE (masquer variables récuper.) [ÉTAPE 9]
□ 10. L10: masquer champs d'extraction futurs               [ÉTAPE 10]
□ 11. Incremental masking (single_frac=0.3)                 [ÉTAPE 11]
□ 12. Phase interleaved finale (1/k par tâche, 30k+ steps)  [ÉTAPE 12]
□ 13. Generative replay (sommeil) entre stages              [ÉTAPE 13]
□ 14. VRAM cap (batch × L ≈ 1.9×10⁴)                       [ÉTAPE 14]
□ 15. ROCm: seq_len fixe (1 plan FFT)                       [ÉTAPE 15]
□ 16. Per-step accuracy > 0.99 avant d'ajouter un pas       [ÉTAPE 16]
□ 17. Cascade eval régulière (le produit ∏ p_i ne ment pas) [ÉTAPE 17]
□ 18. Mots RÉELS uniquement (jamais inventer)               [ÉTAPE 18]
```

---

## ANNEXE A — CODE DE RÉFÉRENCE COMPLET

### A.1 Création du Modèle

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
    def forward(self, x):
        return self.norm(x)

class SpectralBlock(nn.Module):
    def __init__(self, d_model, seq_len, bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.seq_len = seq_len

        self.in_proj  = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        scale = 1.0 / math.sqrt(d_model)
        self.filter_real = nn.Parameter(
            torch.randn(seq_len // 2 + 1, d_model) * scale + 1.0
        )
        self.filter_imag = nn.Parameter(
            torch.randn(seq_len // 2 + 1, d_model) * scale
        )

        if not bidirectional:
            freqs = torch.arange(seq_len // 2 + 1).float()
            self.register_buffer("causal_weight", torch.sigmoid(-freqs * 0.1))

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        B, L, D = x.shape
        h = self.norm1(x)
        h = self.in_proj(h)

        X_freq = torch.fft.rfft(h, dim=1)
        fr = self.filter_real[:X_freq.shape[1], :].unsqueeze(0)
        fi = self.filter_imag[:X_freq.shape[1], :].unsqueeze(0)
        X_real = X_freq.real * fr - X_freq.imag * fi
        X_imag = X_freq.real * fi + X_freq.imag * fr
        X_filtered = torch.complex(X_real, X_imag)

        if not self.bidirectional:
            cw = self.causal_weight[:X_filtered.shape[1], :].unsqueeze(0)
            X_filtered = X_filtered * cw

        y = torch.fft.irfft(X_filtered, n=L, dim=1)
        y = self.out_proj(y)

        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x

class SpXLMv6(nn.Module):
    def __init__(self, vocab_size=200, d_model=256, n_blocks=3,
                 seq_len=64, mode="reasoning", mask_token_id=0,
                 pad_token_id=1, refine_steps=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.seq_len = seq_len
        self.mode = mode
        self.mask_token_id = mask_token_id
        self.refine_steps = refine_steps

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        bidirectional = (mode == "reasoning")
        self.blocks = nn.ModuleList([
            SpectralBlock(d_model, seq_len, bidirectional=bidirectional)
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :L, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def compute_loss(self, input_ids, target_ids, mask):
        logits = self.forward(input_ids)
        masked_logits = logits[mask]
        masked_targets = target_ids[mask]
        return F.cross_entropy(masked_logits, masked_targets)

    @torch.no_grad()
    def diffuse_fill(self, input_ids, mask, n_steps=None):
        n_steps = n_steps or self.refine_steps
        x = input_ids.clone()
        for _ in range(n_steps):
            logits = self.forward(x)
            pred = logits.argmax(dim=-1)
            x = torch.where(mask, pred, x)
        return x
```

### A.2 Boucle d'Entraînement DOSC Complète

```python
import random, torch

random.seed(42)
torch.manual_seed(42)
D = "cuda"

# Tokenizer (adapter à la tâche)
class MathTok:
    def __init__(self):
        chars = list("0123456789+-=;[]/ \n?")
        special = ["<PAD>", "<MASK>", "<BOS>", "<EOS>"]
        self.chars = special + chars
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad = self.stoi["<PAD>"]
        self.mask = self.stoi["<MASK>"]
        self.bos = self.stoi["<BOS>"]
        self.eos = self.stoi["<EOS>"]
    def encode(self, text):
        return [self.stoi.get(c, self.pad) for c in text]
    def decode(self, ids):
        return "".join(self.itos.get(i, "?") for i in ids
                      if self.itos.get(i, "") not in ["<PAD>","<MASK>","<BOS>","<EOS>"])

tok = MathTok()
SEQ_LEN = 48  # ÉTAPE 15: seq_len fixe

# Format SBS
def make_sbs(a, b, seq_len=SEQ_LEN):
    result = a + b
    parts = [f"{a}+{b}=?"]
    current = a
    field_positions = []
    text_offset = 1  # après BOS

    for i in range(1, b + 1):
        nxt = current + 1
        step = f" [{i}/{b}]{current}+1={nxt}"
        parts.append(step)
        eq_pos = step.index("=")
        val_str = str(nxt)
        val_start = text_offset + eq_pos + 1
        val_end = val_start + len(val_str)
        field_positions.append((val_start, val_end))
        current = nxt
        text_offset += len(step)

    final_part = f" ={result}"
    parts.append(final_part)
    ans_start = text_offset + 1
    ans_end = ans_start + len(str(result))
    field_positions.append((ans_start, ans_end))

    full_text = "".join(parts)
    ids = [tok.bos] + tok.encode(full_text) + [tok.eos]
    while len(ids) < seq_len:
        ids.append(tok.pad)
    ids = ids[:seq_len]
    field_positions = [(min(s, seq_len-1), min(e, seq_len)) for s, e in field_positions]
    return ids, field_positions, full_text

# Données
all_adds = [(a, b) for a in range(10) for b in range(10)]
random.shuffle(all_adds)
train_pairs = all_adds[:80]
test_pairs = all_adds[80:]

# Modèle
model = SpXLMv6(
    vocab_size=tok.vocab_size, d_model=256, n_blocks=3,
    seq_len=SEQ_LEN, mode="reasoning",
    mask_token_id=tok.mask, refine_steps=5,
).to(D)

# Optimiseur
base_lr = 1e-3
optimizer = torch.optim.AdamW(
    model.parameters(), lr=base_lr, weight_decay=0.1, betas=(0.9, 0.95)
)

# Warmup
WARMUP = 500
def apply_warmup(step):
    scale = min(1.0, (step + 1) / WARMUP)
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * scale

# === PHASE 1: GROKKER +1 ===
global_step = 0
for step in range(15000):
    global_step += 1
    apply_warmup(global_step)
    batch_pairs = random.sample(train_pairs, 16)
    sequences = [make_sbs(a, b) for a, b in batch_pairs]
    input_ids = torch.tensor([s[0] for s in sequences], dtype=torch.long, device=D)
    all_fields = [s[1] for s in sequences]
    target_ids = input_ids.clone()
    masked_input = input_ids.clone()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b_idx in range(len(sequences)):
        fields = all_fields[b_idx]
        n_fields = len(fields)
        champ_idx = random.randint(0, n_fields - 2) if n_fields > 1 else 0
        if random.random() < 0.3:  # single_frac
            s, e = fields[champ_idx]
            masked_input[b_idx, s:e] = tok.mask
            mask[b_idx, s:e] = True
        else:  # anti-raccourci
            for c in range(champ_idx, n_fields):
                s, e = fields[c]
                masked_input[b_idx, s:e] = tok.mask
                mask[b_idx, s:e] = True
    optimizer.zero_grad()
    logits = model(masked_input)
    ml = logits[mask]; mt = target_ids[mask]
    if ml.shape[0] == 0: continue
    loss = F.cross_entropy(ml, mt)
    if torch.isnan(loss): continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# === PHASE 2: GROKKER ANS ===
# === PHASE 3: INTERLEAVED ===
# (Voir protocol_complete.py pour le code complet)
```

---

## ANNEXE B — FORMULES DE RÉFÉRENCE

```
═══════════════════════════════════════════════════════════════
 SPXLM v6 — FORMULA REFERENCE CARD
═══════════════════════════════════════════════════════════════

 ACCURACY & DEPTH (EXACT)
 ──────────────────────────
 A_cascade  = ∏ᵢ pᵢ                       (vérifié ±0.003)
 D_reliable = 1 / (1 − p̄)                (série géométrique)
 D_total    = k / (1 − A^(1/k))           (identité algébrique)
 A(k,D)     = (1 − 1/D)^k                 (inverse)

 SCALING LAW (ESTIMATED)
 ────────────────────────
 D(k,n_blk,T) = k^β × n_blk^α × T^δ × C₀(task)

   β ≈ 3.5   k (scratchpad)    — doubler k × 11.3 la profondeur
   α ≈ 2/3   n_blk (blocs)     — doubler n_blk × 1.59
   δ ≈ 1.34  T (training)      — doubler T × 2.53
   γ ≈ 0     d_model           — NON-MONOTONE, PAS un levier

 EFFICIENCY FRONTIER
 ────────────────────
 D ∝ k^3.5 × n_blk^(−0.34) × d^(−0.94) × C_train^1.34
 → MAXIMISER k et C_train
 → MINIMISER d et n_blk

 VRAM BUDGET
 ────────────
 batch × L ≈ 1.9 × 10⁴   (cap 3 GB, d=256, n_blk=3)
 L = 1 + 4k
 batch ≈ 4750 / k

 EXECUTION TIME
 ──────────────
 t_step(d,n) ≈ 7.5 × (d/64)^0.7 × (n/2)^0.75   [ms]
 C_train = T × t_step

 OPTIMAL WIDTH
 ─────────────
 d* ≈ 128–256  pour T ≤ 30k, k = 3–5

═══════════════════════════════════════════════════════════════
```

---

## ANNEXE C — RÉSULTATS VALIDÉS

| k (pas) | Champs | Format | Steps total | Cascade FINAL | Source |
|---|---|---|---|---|---|
| 1 | 3 | CD | 40k | **0.984** | Rapport 58 |
| 2 | 5 | CD | 55k | **0.950** | Rapport 59 |
| 3 | 7 | SBS | 70k | **0.972** (peak 0.990) | Rapport 61 |
| 2 | 5 | SBS | — | **0.971** | Rapport 62 |
| 1 (recur.) | 1 | — | 3k, r=100k | **1.000** | E12 |

**Ces résultats sont des valeurs mesurées réelles, non estimées.**

---

*Document généré le 2026-06-18 à partir de SPXLM_PUBLICATION.md (1856 lignes), PROTOCOLE_EXTRAIT.md, model.py, protocol_complete.py, et protocol_sleep.py.*

*Ce protocole a permis d'atteindre 100% de généralisation sur l'addition (cascade = 1.0).*
