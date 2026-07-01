# REPRODUCIBILITÉ OCM-26400 — Guide complet de reconstruction

> **Objectif** : recréer un modèle OCM-26400 de zéro, sans se perdre. Ce document est **auto-suffisant** : il contient TOUT (paradigme, lois, règles, exigences, formules, architecture, hyperparamètres, données, protocole). Lis ça, suis l'ordre, tu reconstruis.
>
> Théorie détaillée : `SOLUTION_OCM26400.md`. Exigences brutes : `Besoins/` + `CAHIER_DES_CHARGES_FINAL.md`.

---

## TABLE DES MATIÈRES
1. [Paradigme](#1-paradigme)
2. [Les 6 lois + loi unifiée](#2-les-6-lois--loi-unifiée)
3. [Les règles (gates, training, licensing)](#3-les-règles)
4. [Les formules](#4-les-formules)
5. [Architecture exacte (code)](#5-architecture-exacte)
6. [Hyperparamètres canoniques](#6-hyperparamètres-canoniques)
7. [Données](#7-données)
8. [Protocole d'entraînement (étape par étape)](#8-protocole-dentraînement)
9. [Protocole d'évaluation (rigoureux)](#9-protocole-dévaluation)
10. [Résultats attendus](#10-résultats-attendus)
11. [Checklist de reproduction](#11-checklist)

---

## 1. PARADIGME

**Apprendre → Comprendre → Raisonner → Générer**, noyau unique = SpectralCoreBlock (FFT).

| Phase | Mécanisme | Outil |
|---|---|---|
| Apprendre | Capture simultanée (toutes modalités en 1 passe) → IDs numériques | encodeurs modality-specific |
| Comprendre | **Grok** (phase transition) | loss **1-cos** (cœur) |
| Raisonner | Boucle **LSRA** itérative + gate | `v(t+1)=Block(v(t))`, stop meta[0]≥τ |
| Générer | Crown-jewel inversé | opération apprise rejouée |

**Principe fondateur (L4)** : *raisonner = ajouter des ÉTAPES, pas des PARAMÈTRES.*

---

## 2. LES 6 LOIS + LOI UNIFIÉE

| Loi | Énoncé |
|---|---|
| **L1** | Décomposition > Scale (compétence = étapes, pas masse) |
| **L2** | Masquage incrémental (sous-ensemble visible → cascade inférence) |
| **L3** | `depth_max ≈ 1/(1−per_step_err)` → per-step exact ⇒ profondeur ∞ |
| **L4** | Récurrence ⊥ Longueur ⊥ Params (raisonner = étapes) |
| **L5** | `L = 1 + 4·D` (scratchpad ; `batch·L ≈ 1.9e4`) |
| **L6** | Association (1-source direct ; multi-source = décomposer) |

**Loi unifiée** :
```
D = k^1.98 · P^1.06 · d^−2.38
```
- `D` = profondeur fiable, `k` = cste tâche, `P` = params, `d` = dimension
- **γ=1.06** (D∝P), **δ=−2.38** (élargir d détruit D — scale inverse)
- **Conséquence design** : d=64 + 1 bloc = optimum (d-min pour P≤27, maximise D, L4)

---

## 3. LES RÈGLES

### 3.1 Gates (seuils de validation par niveau)
```
L1 ≥ 0.99   (primitives grokkées)
L2 ≥ 0.95   (composition)
L5 ≥ 0.90   (scratchpad)
L6 ≥ 0.85   (association)
τ_grok = sigmoid(meta[0]) ≥ 0.9   (stop LSRA)
CONF_TARGET = 4.0   (sigmoid(4)≈0.98)
```

### 3.2 Règles d'entraînement (procédure canonique `train_binary_block`)
```
loss      = 1 − cos(ent, target)      # 1-COS (pas CE) pour le CŒUR
optimizer = Adam (lr=3e-3)            # PAS AdamW pour le crown-jewel
seed      = 0
batch     = 64
n_steps   = 1500 (primitives) → 100000 (audio/perception)
```

### 3.3 Règle de répartition des loss (CRITIQUE)
| Composant | Loss | Pourquoi |
|---|---|---|
| **Cœur de raisonnement** (SCB, déterministe) | **1-cos** | grok, généralisation exacte |
| **Lobes sensoriels** (audio, image, perception) | **CE** (cross-entropy) | perception stochastique, CE optimal |

> ⚠️ **Ne pas inverser** : 1-cos sur perception plafonne (audio 1-cos = 58% vs CE 94%). CE sur le cœur casse le grok.

### 3.4 Principe IDs numériques
- TOUT convertir en IDs numériques discrets avant le SpectralCoreBlock.
- Audio : invariant IDs introuvable par extraction (3 méthodes testées, non-invariantes) → le lobe audio (M5) produit les features, le cœur raisonne dessus.
- Crown-jewel : one-hot dans `slot = d//2` (P ≤ slot).

### 3.5 Lobe Licensing (commercial)
- Lobes sensoriels = open-source (CE).
- Cœur de raisonnement (SCB + 1-cos + LSRA) = commercial.

### 3.6 Sommeil (3 phases, non-optionnel) — VALIDÉ + CONTRÔLÉ (session 2026-07-01)
> **Thèse** : le sommeil transforme la MÉMOIRE en COMPREHENSION. Le bassin de mémorisation est un min local aigu que le SGD pur ne quitte PAS — seul le filtrage spectral le casse.

**Mécanisme neural (validé, +73pt vs pur)** :
1. **Léger (low-pass)** — `keep_frac=0.5` : FFT des poids (dim=0), zero HF → détruit la mémorisation "spiky". **Ingrédient actif** (replay_seul=+0pt, low-pass_seul=+17pt).
2. **Profond (high-pass)** — `keep_frac=0.3` : zero BF → affine détails (high-pass_seul=-2pt, mais synergique avec low-pass).
3. **Replay** — 200 steps/phase entre les filtres : retrain sur données.
4. **Cycles** : répéter léger→profond→replay. **5 cycles optimal** (1→60%, 2→94%, 5→99%).

**Config canonique** (test_sleep_neural.py) : low-pass 0.5 → replay 200 → high-pass 0.3 → replay 200, ×5 cycles. Résultat : **99.1% ± 0.7** (vs éveil 25.7%, vs pur +2000 steps 26.0%). `keep_frac` pic ÉTROIT à 0.5 (0.3→19%, 0.7→37%).

**Symbolique** (sleep_phases.py) : extraction de règle (α,β) des faits épisodiques (N faits → 1 règle, compression N×) + paradoxal (analogies/composites).

### 3.7 Gate → composition arbitraire (VALIDÉ, session 2026-07-01)
> **Thèse** : la capacité compositionnelle arbitraire émerge d'UNE primitive grokkée + la gate qui certifie.

- **Gate** = alignement (cosinus) de la sortie au dictionnaire canonique (L_align). ≥ τ → étape certifiée.
- Grok 1 primitive (op(a,b), 1-cos) → 100%. Compose en cascade, **gate certifie chaque étape → 100% à profondeur 50** (gate≥0.987).
- **Le grok = la gate, PAS le scale.** (grok_gate_composition.py)

---

## 4. LES FORMULES

| Grandeur | Formule |
|---|---|
| Loi unifiée | `D = k^1.98 · P^1.06 · d^−2.38` |
| Profondeur max | `depth_max ≈ 1/(1−per_step_err)` |
| Scratchpad | `L = 1 + 4·D` |
| Loss crown-jewel | `L = 1 − cos(ent, target)` |
| LSRA | `v(t+1) = ReasonerBlock(v(t))`, stop `sigmoid(meta[0]) ≥ 0.9` |
| AMV-256 | `[ent(64) \| prop(64) \| op(64) \| meta(64)]`, meta[0]=confiance |
| Opération test | `op(a,b) = (3a + 5b) mod P` |
| SCB FFT | `y = irfft(rfft(LN(Linear(x))) ⊙ filtre_compleme_appris) → résiduel → FFN` |

---

## 5. ARCHITECTURE EXACTE

### 5.1 SpectralCoreBlock (crown-jewel, noyau unique)
```python
class SpectralCoreBlock(nn.Module):
    # d_model, seq_len, bidirectional=True
    # in_proj, out_proj : Linear(d,d)
    # filter_real, filter_imag : Parameter(seq_len//2+1, d)  # filtre fréquentiel appris
    # norm1, norm2 : LayerNorm ; ffn : Linear(d,4d)→GELU→Linear(4d,d)
    def forward(x):  # (B,L,d) ou (B,d)
        h = norm1(x); h = in_proj(h)
        X = rfft(h, dim=1)                    # FFT (fp32 sous AMP !)
        X = X * (filter_real + i·filter_imag) # filtre complexe appris
        y = irfft(X, n=L, dim=1); y = out_proj(y)
        x = x + y                              # résiduel
        x = x + ffn(norm2(x))                  # FFN
```
**Complexité O(L log L).** Patch AMP : FFT en fp32 (rfft ne supporte pas bf16).

### 5.2 Architecture unifiée (recommandée, LEAN)
```
Cœur : d=64, 1 SpectralCoreBlock + ReasonerBlock itérable (LSRA) + gate_head(meta[0])
       ↕ P ≤ 27 IDs (slot = d//2 = 32)
Lobes (1 par modalité) : texte, audio(M5 Conv1d), image, vidéo, 3D
       chacun : signal → features → P IDs → projette dans le cœur d=64
```

### 5.3 ReasonerBlock (raisonnement latent, itérable)
```python
class ReasonerBlock(nn.Module):  # d, hidden=max(2d,128)
    def forward(x): return x + fc2(relu(fc1(LayerNorm(x))))  # MLP résiduel
# entraîné via train_reasoner_with_confidence : 1-cos(ent→canonical) + (meta[0]→4.0)²
```

### 5.4 Lobe audio (M5 + SCB) — référence
```python
# Conv1d(1,32,k80,s16)+BN+ReLU+MaxPool4 → Conv1d(32,32,k3)+BN+ReLU+MaxPool4
# → Conv1d(32,64,k3)+BN+ReLU → Conv1d(64,64,k3)+BN+ReLU  (T≈62 frames)
# → SpectralCoreBlock(d=64,seq_len=62,bidir) → mean pool → Linear(64,35) + CE
```

---

## 6. HYPERPARAMÈTRES CANONIQUES

| Param | Crown-jewel (cœur) | Audio (lobe) |
|---|---|---|
| d | 64 | 64 (sweep 64-256) |
| n blocs | **1** (L4) | 1-7 |
| Loss | **1-cos** | **CE** |
| Optimizer | Adam lr=3e-3 | AdamW lr=1e-3 wd=5e-4 fused |
| Steps | 1500-2000 | 100000 (cosine LR) |
| Batch | 64-128 | 32 |
| Seed | 0 (+1,2 pour stats) | 0,1,2 |
| AMP | — | bf16 (FFT fp32) |
| Augmentation | — | speed-perturb [0.9,1.1] (bruit additif N'AIDE PAS) |

---

## 7. DONNÉES

| Tâche | Dataset | Split |
|---|---|---|
| Crown-jewel | synthétique `op(a,b)=(3a+5b) mod P`, P∈{3..27} | P² paires train, triples test jamais vus |
| Audio | SpeechCommands v0.02 (35 mots) | `testing_list.txt` officiel (test) + 10% train→val |
| — | SpeechCommands path | `/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02` |
| Bruits (aug) | `_background_noise_/` (6 fichiers officiels) | SNR 10-40 dB (NB : l'augmentation par bruit additif dégrade) |

---

## 8. PROTOCOLE D'ENTRAÎNEMENT

### Phase 1 — Crown-jewel (cœur de raisonnement)
1. `train_binary_block` : ReasonerBlock sur op(a,b)→m, **1-cos**, Adam 3e-3, seed 0, 1500 steps. Gate L1≥0.99.
2. **Sommeil** 3 phases (obligatoire).
3. Cascade scratchpad (intermédiaire PUIS final) → composition profonde, gate L2≥0.95.

### Phase 2 — Lobes sensoriels
4. Lobe audio : M5+SCB, **CE**, AdamW, cosine 100k. Sélection sur **val** (pas test !).
5. Pour chaque lobe : encoder → P IDs → projection vers cœur d=64.

### Phase 3 — Intégration
6. Capture simultanée (3 vues texte+phon+audio → même canonical, 1-cos).
7. LSRA + gate à l'inférence (itérer jusqu'à confiance).

---

## 9. PROTOCOLE D'ÉVALUATION (RIGOREUX — ne pas tricher)

```
1. Split : test officiel + 10% train → val
2. Sélection du meilleur checkpoint : SUR VAL (jamais sur test)
3. Test : évalué UNE SEULE FOIS (pas de max-sur-test)
4. ≥3 seeds {0,1,2} → report mean ± std
5. Un template fixe par sweep (OFAT : une variable à la fois)
6. Front-end figé si on teste d (c3/c4 constants + projection)
```
> ⚠️ Tricher (sélection sur test, 1 seed) gonfle les scores de ~1pt et invalide toute comparaison. [validé par DA+Juges]

---

## 10. RÉSULTATS ATTENDUS

| Tâche | Métrique | Résultat |
|---|---|---|
| Crown-jewel décomposition | acc triples non-vus | **100%** (tout d 64-512, P 3-27) |
| Crown-jewel chaînes | acc k-step | **100% à k=500** (L3, profondeur ∞) |
| Blocs vs étapes | 1 bloc vs 8 blocs | **identiques** (L4) |
| Audio (d=256, rigoureux) | acc test | **93.91% ± 0.08%** (SpeechCommands) |
| Audio (d=64, rigoureux) | acc test | **93.52% ± 0.05%** (sweet spot efficace) |
| 1-cos sur audio (perception) | acc test | **58%** (confirme : 1-cos = cœur, CE = lobes) |
| **Gate → composition** | acc cascade profondeur D | **100% à D=50** (1 primitive + gate, 2026-07-01) |
| **Sommeil spectral** | acc test après sommeil | **99.1% ± 0.7** (vs éveil 25.7%, vs pur 26.0%, +73pt) |
| **Grok pur (extrapolation)** | acc paires tenues secrètes | **0-8%** (tous setups) — le grok Besoins = composition, pas extrapolation |

---

## 11. CHECKLIST DE REPRODUCTION

- [ ] Implémenter `SpectralCoreBlock` (FFT, filtre complexe appris, résiduel+FFN, AMP-safe)
- [ ] Implémenter `ReasonerBlock` (MLP résiduel) + `train_reasoner_with_confidence` (1-cos + gate)
- [ ] Crown-jewel : op(a,b) mod P, 1-cos, Adam 3e-3, seed 0 → 100% décomposition
- [ ] Vérifier L3 : chaînes k=50 à 100%
- [ ] Vérifier L4 : 1 bloc = 8 blocs
- [ ] Lobe audio : M5+SCB, CE, AdamW cosine 100k, val-split, 3 seeds → ~93.9% (d=256)
- [ ] Cœur d=64 + 1 bloc + LSRA + lobes par modalité
- [ ] **Gate + composition** : grok 1 primitive + gate (alignement canonique) → cascade 100% à D=50
- [ ] **Sommeil spectral** : low-pass 0.5 + high-pass 0.3 + replay, 5 cycles → 99% (contrôle vs pur : +73pt)
- [ ] Éval : val-split + select-on-val + test-1× + ≥3 seeds (PAS de sélection-sur-test)

---

## FICHIERS CLÉS (code canonique)
- `ocm26400/spectral_core.py` — SpectralCoreBlock
- `ocm26400/reasoner.py` — ReasonerBlock + LSRA + train_reasoner_with_confidence
- `ocm26400/verifier.py` + `amv.py` — SymbolicDict, AMV-256, op mod P
- `ocm26400/audio_unified_m5scb.py` — lobe audio M5+SCB
- `ocm26400/audio_sweep_d_rigorous.py` — sweep d propre
- `ocm26400/crown_jewel_*.py` — tests crown-jewel (décomposition, depth, min-d)
- `ocm26400/grok_gate_composition.py` — gate + composition arbitraire (VALIDÉ, 100% D=50)
- `ocm26400/test_sleep_neural.py` + `optimize_sleep.py` + `control_sleep_vs_puresteps.py` — sommeil spectral (VALIDÉ, 99%, +73pt)
- `ocm26400/sleep_phases.py` — sommeil symbolique (extraction règle)
- `ocm26400/verify_scale_antigrok.py` — vérif formule scale=anti-grok

*Si un seul doc à lire pour recréer : celui-ci. Pour la théorie : SOLUTION_OCM26400.md.*
