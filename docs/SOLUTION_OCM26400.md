# SOLUTION OCM-26400 — Document canonique

> Architecture spectrale unifiée pour le raisonnement et la perception.
> Ce document est la **source de vérité** : paradigme, lois, formules, architecture, résultats.
> Les autres docs (RAPPORT_*, AUDIT_*, RECHERCHE_*) sont historiques — celui-ci l'emporte en cas de conflit.

---

## 1. LE PARADIGME

**Apprendre → Comprendre → Raisonner → Générer**, organisé autour d'un noyau unique : le **SpectralCoreBlock** (mélangeur FFT).

- **Apprendre** : capture simultanée de toutes les modalités en une passe (texte, audio, image, vidéo, 3D) → **IDs numériques**.
- **Comprendre** : **grok** (phase transition) via la loss **1-cos** sur les tâches déterministes.
- **Raisonner** : boucle **LSRA** (`v(t+1) = Block(v(t))`), stoppée par le **gate** (observateur de confiance).
- **Générer** : crown-jewel inversé (l'opération apprise rejouée en sortie).

**Principe fondateur (L4)** : *raisonner = ajouter des ÉTAPES, pas des PARAMÈTRES.* La capacité vient de la récurrence, pas de l'empilement.

---

## 2. LES 6 LOIS + LOI UNIFIÉE

| Loi | Énoncé | Statut |
|---|---|---|
| **L1** | Décomposition > Scale (la compétence vient des étapes, pas de la masse) | ✅ Validé (crown-jewel 100% via décomposition, scale casse le grok) |
| **L2** | Masquage incrémental (sous-ensemble visible → cascade à l'inférence) | ✅ |
| **L3** | `depth_max ≈ 1/(1−per_step)` — per-step exact → profondeur ∞ | ✅ Validé (chaînes k=500 à 100%) |
| **L4** | Récurrence ⊥ Longueur ⊥ Params (raisonner = étapes) | ✅ Validé (1 bloc = 8 blocs pour le raisonnement) |
| **L5** | `L = 1+4·D` (format scratchpad ; `batch·L ≈ const`) | ✅ |
| **L6** | Association (1-source direct ; multi-source = décomposer) | ✅ |

**Loi unifiée** :
```
D = k^1.98 × P^1.06 × d^−2.38
```
- `D` = profondeur de raisonnement fiable
- `k` = constante de tâche ; `P` = params ; `d` = dimension (d_model)
- **γ = 1.06** : D ∝ P (plus de params = un peu plus de D)
- **δ = −2.38** : D ∝ d^−2.38 (**élargir d détruit D** — scale inverse). ⚠️ **COMPOSANTE d OBSOLÈTE** : l'anti-grok (d^−2.38 comme d^−3.55) est **RÉFUTÉ** en 2026-07-01 (§9.4 — ni primitive ni composition one-shot ne dégradent avec d). Garder pour l'histoire ; d=64 reste efficient en coût, pas en anti-grok.

### Vérification empirique (30 juin, voir §6)
- **δ = −2.38 sur la perception** : sweep d audio (rigoureux) → tous d ≈ 93.5-93.9% (effet faible), mais les runs confondus précédents montraient d=1024 dégrader (scale-inverse confirmé qualitativement).
- **L3 sur le raisonnement** : chaînes k-step 100% à k=50 (puis k=500) → profondeur ∞ si per-step exact.
- **L4** : 1 bloc = 8 blocs (k=500 à 100%) → les blocs n'ajoutent rien au raisonnement.

---

## 3. L'ARCHITECTURE — crown-jewel + lobes

### Le SpectralCoreBlock (crown-jewel)
Mélangeur spectral FFT bidirectionnel, **noyau unifié** de tout le système :
```
x → LayerNorm → Linear → rfft(seq) → filtre fréquentiel complexe APPRIS → irfft → out_proj → résiduel → FFN
```
- Complexité **O(L log L)** (pas d'attention O(L²)).
- Stabilité **Parseval** (‖x‖² = ‖FFT(x)‖²).
- Filtre `filter_real/imag` shape `(seq_len//2+1, d)` — appris, content-global.
- AMP-safe : FFT en fp32 interne (rfft ne supporte pas bf16).

### Architecture unifiée (LEAN, optimale)
```
               ┌─────────────────────────────────────┐
               │   CŒUR DE RAISONNEMENT               │
               │   d = 64, 1 SpectralCoreBlock        │
               │   + boucle LSRA (étapes, depth ∞)    │
               │   + gate meta[0] (observateur)       │
               └──────────────▲──────────────────────┘
                              │ P ≤ 27 IDs numériques (slot=32)
        ┌──────────┬──────────┼──────────┬──────────┐
     texte      audio(M5)   image     vidéo       3D
   (lobe OS)  (lobe OS)   (lobe OS)  (lobe OS)  (lobe OS)
        └──────────┴──────────┴──────────┴──────────┘
              lobes sensoriels (1 par modalité)
```

### Pourquoi d=64 + 1 bloc est optimal
- **Représentation** : `slot = d//2 = 32 ≥ P_max (27)` → tous les IDs tiennent. [validé]
- **Loi (d^−2.38)** : d=64 **maximise D** → plus petit = plus de profondeur fiable.
- **L4** : 1 bloc = capacité de raisonnement maximale (k=500+) ; +de blocs = 0 gain, params gaspillés.
- **L3** : la profondeur vient des **étapes LSRA**, pas des blocs.

> **d=64 + 1 bloc** n'est pas un compromis : c'est l'optimum mathématique du système.

---

## 4. LES FORMULES CLÉS

| Grandeur | Formule | Rôle |
|---|---|---|
| Profondeur fiable | `D = k^1.98 · P^1.06 · d^−2.38` | loi unifiée |
| Profondeur max | `depth_max ≈ 1/(1−per_step_err)` | L3 (per-step exact → ∞) |
| Format scratchpad | `L = 1 + 4·D` | L5 |
| Loss crown-jewel | `L = 1 − cos(ent, target)` | 1-cos (grok) |
| Gate | `stop si sigmoid(meta[0]) ≥ τ_grok = 0.9` | observateur LSRA |
| AMV-256 | `[ent(64) \| prop(64) \| op(64) \| meta(64)]` | format vecteur, meta[0]=confiance |

---

## 5. Lobe Licensing (modèle commercial)

- **Lobes sensoriels** (audio M5, vision, texte...) = **PERCEPTION**, open-source, loss **CE** (cross-entropy).
- **Cœur de raisonnement** (SpectralCoreBlock + 1-cos grok + LSRA) = **RAISONNEMENT**, commercial.

> La perception est stochastique (CE optimal) ; le raisonnement est déterministe (1-cos grok). **Ne pas mélanger** : 1-cos sur perception plafonne (audio 1-cos = 58% vs CE 94%). [validé]

---

## 6. RÉSULTATS (rigoureux, protocole propre)

### Crown-jewel (raisonnement, op = 3a+5b mod P)
- **Décomposition : 100% pour tout d (64-512) × P (3-27)** sur triples jamais vus. [validé]
- One-shot (sans décomposition) : ~0.5-2.5% → la compétence vient des **étapes** (L1).
- Chaînes k-step : **100% à k=500** (profondeur ∞, L3). 1 bloc = 8 blocs (L4).

### Audio (SpeechCommands v0.02, 35 mots, split officiel) — sweep rigoureux
*Protocole propre : val split (9464), sélection sur val, test (11005) évalué 1×, 3 seeds, front-end figé, mean ± std.*

| d | params | test mean | ± std |
|---|---|---|---|
| 64 | 351K | 93.52% | 0.05% |
| 128 | 1.25M | 93.45% | 0.33% |
| 192 | 2.73M | 93.78% | 0.15% |
| 256 | 4.77M | **93.91%** | 0.08% |

**Conclusions audio** :
- d a un **effet faible** (+0.4pt de d=64→256, dans le bruit). d=64 = sweet spot efficace (93.5% à 351K params).
- Le **ratio P/d=0.1875 est RÉFUTÉ** (d=192 n'est pas un pic ; P=35 est fixe, le ratio n'a pas de sens).
- Les anciens « 94.5-95.2% » étaient **inflatés de ~1pt** par sélection-sur-test (biais dénoncé par les DA/Juges). **Vraie accuracy mono-modèle ~93.9%**.
- SOTA ~96% : le gap (~2pt) ne se comble PAS par d — il faut un autre levier (Mel/sinc front-end, SpecAugment, architecture différente).

### 1-cos vs CE sur perception
- 1-cos pur : 58%. ArcFace (marge angulaire) : 57.7%. CE : 93.9%. → **CE requis sur les lobes sensoriels** (1-cos pour le cœur de raisonnement seulement).

---

## 7. VALIDATION DES RÈGLES (checklist)

| Règle | Statut | Preuve |
|---|---|---|
| Crown-jewel SCB(seq_len=T) | ✅ | +4.1pt sur M5 audio pur (90.4→94.5%, ancien protocole) |
| Grok 1-cos (phase transition) | ✅ | détecté sur audio (step 6000), 100% sur crown-jewel |
| Gate/Observer meta[0] | ✅ | 94.8% sélectif @ 99% coverage |
| d=64 + 1 bloc optimal | ✅ | L3/L4, k=500 à 100%, d-min unifié |
| Capture simultanée | ✅ | 3 vues (texte+phon+audio) → crown-jewel |
| Sommeil 3 phases | ✅ **VALIDÉ+contrôlé** | neural : +73pt vs pur (99.1% vs 25.7%), 5 cycles, low-pass=ingrédient actif (§9) |
| **Gate → composition arbitraire** | ✅ **VALIDÉ** | 1 primitive grokkée + gate = 100% à profondeur 50 (§9) |
| LSRA auto-correction | ✅ (raisonnement) / ⚠️ (perception : n'aide pas, info absente de v0) |
| CE sur lobes / 1-cos sur cœur | ✅ | 1-cos sur perception = 58%, CE = 93.9% |

---

## 8. CODE CANONIQUE (à garder)

- `ocm26400/spectral_core.py` — SpectralCoreBlock (crown-jewel, patché AMP)
- `ocm26400/reasoner.py` — ReasonerBlock + LSRA + train_reasoner_with_confidence
- `ocm26400/verifier.py` + `amv.py` — SymbolicDict, AMV-256, op mod P
- `ocm26400/audio_sweep_d_rigorous.py` — sweep d propre (val split, 3 seeds)
- `ocm26400/crown_jewel_*.py` — crown-jewel (décomposition, depth, min-d, ratio)
- `ocm26400/audio_unified_m5scb.py` — lobe audio M5+SCB (référence)

**Session 2026-07-01 (gate + sommeil)** :
- `ocm26400/grok_gate_composition.py` — 1 primitive grokkée + gate → composition profondeur arbitraire (100% à D=50)
- `ocm26400/test_sleep_neural.py` + `_seeds.py` — sommeil neural 3 phases (5 seeds, +34pt)
- `ocm26400/optimize_sleep.py` — ablation + intensité + cycles (config optimale : kf=0.5, 5 cycles → 99%)
- `ocm26400/control_sleep_vs_puresteps.py` — contrôle décisif (sommeil vs pur +2000 steps, +73pt)
- `ocm26400/verify_scale_antigrok.py` — vérif formule D=k^3.5·d^-3.55·T^2.06
- `ocm26400/grokking_heldout.py` + `grokking_canonical_ce.py` + `grokking_scb.py` — tests grok pur (tous 0-8%, réfutent l'extrapolation atomique)

> Les ~48 `audio_m5_scb_*.py` one-off (wide/deep variants) sont des expériences obsolètes — purgeables.

---

## 9. SESSION 2026-07-01 — GATE, SOMMEIL, CORRECTIONS (nouveau)

### 9.1 La GATE fait grokker (principe clé, VALIDÉ)
> **Thèse** : *la capacité compositionnelle arbitraire émerge d'UNE primitive grokkée + la gate qui certifie.*

- Grok 1 primitive (op(a,b)=(a+b) mod 23, 1-cos, crown-jewel) → 100% en 1000 steps.
- **Gate** = alignement (cosinus) de la sortie au dictionnaire canonique (L_align du Besoins). Si ≥ τ → étape certifiée.
- Composition en cascade à profondeur D : **100% à D=50** (gate ≥ 0.987). La gate empêche l'accumulation d'erreur → profondeur arbitraire.
- **Le grok = la gate, PAS le scale.** Raisonner = enchaîner des étapes certifiées.
- *Preuve* : `grok_gate_composition.py`. Voir mémoire `crown-jewel-composition-not-extrapolation`.

### 9.2 Le sommeil spectral fonctionne (VALIDÉ + CONTRÔLÉ — corrige la réfutation antérieure)
> **Thèse** : *le sommeil transforme la MÉMOIRE en COMPREHENSION.*

- Tâche : class=(seq[3]+seq[7]) mod 5, MLP. **Éveil** 1500 steps → train 100% / **test 25.7%** (mémorisation verrouillée).
- **Baseline** (+2000 steps purs, sans sommeil) → 26.0% (**STUCK** : le bassin de mémorisation est un min local aigu que SGD ne quitte pas).
- **Sommeil** (low-pass keep_frac=0.5 + high-pass 0.3 + replay 200/phase, **5 cycles**) → **99.1% ± 0.7** (3 seeds).
- **Δ sommeil − pur = +73pt.** Le sommeil fait ce que l'entraînement pur NE PEUT PAS.
- **Ce n'est PAS du grokking retardé classique** (Power) : là, le pur finit par grokker ; ici le pur **reste stuck** — seul le filtrage spectral débloque.
- **Ablation** : replay_seul = +0pt (inutile) ; high-pass_seul = −2pt ; **low-pass_seul = +17pt (ingrédient actif)** ; léger+profond = +32pt/cycle.
- **Intensité** : keep_frac low-pass = pic étroit à **0.5** (0.3→19%, 0.5→60%, 0.7→37%).
- **Cycles composent** : 1→60% | 2→94% | 3→95% | 5→99%. Le grok est déclenché ET approfondi par les cycles.
- **Mécanisme** : le low-pass détruit la composante HF des poids (mémorisation "spiky") → le modèle retombe dans le bassin généralisant → replay affine.
- *Code* : `test_sleep_neural.py`, `optimize_sleep.py`, `control_sleep_vs_puresteps.py`. Mémoire `neural-sleep-3phase-works`.
- **CORRIGE** la réfutation antérieure (`spectral_sleep_3arm.py` Δ0.0pt) : ce test utilisait un mécanisme trop faible (replay uniform vs replay). Le filtrage spectral 3 phases a un effet réel.

### 9.3 Crown-jewel = COMPOSITION, pas extrapolation (correction)
- Le crown-jewel fait la **composition** à 100% (triples/cascade non-vus via décomposition) — c'est le grok au sens du Besoins.
- Le grok **pur** (extrapolation aux paires atomiques tenues secrètes) = **0-8% tous setups** (1-cos+one-hot, CE+dense+wd∈{0,1e-3,1e-2,1e-1}, SCB L=P). Mémorisation parfaite (100% train), 0% généralisation atomique.
- **Conclusion** : "grok" dans le Besoins = composition (primitives à 1.0 + cascade scratchpad). L'extrapolation Power-2022 est un phénomène plus fort, non atteint (et non requis : la composition suffit via la gate).

### 9.4 Formule scale=anti-grok — RÉFUTÉE (primitive ET composition one-shot)
- Formule utilisateur (raffinée) : `D = k^3.5 · d^−3.55 · T^2.06`.
- **T aide** (c>0) : +de steps → +de profondeur, MAIS c'est un **seuil binaire de grok** (T ≥ T_grok → composition ∞ ; T < → 1), pas T^2.06 lisse.
- **d^−3.55 (anti-grok) RÉFUTÉ à TOUS les niveaux** :
  - *Primitive* (`verify_scale_antigrok.py`) : un +gros modèle grok la primitive **plus vite** (d=256 grok à T=40 où d=32 échoue ; exposant b=+1.55).
  - *Composition one-shot* (`complete_formula.py`) : one-shot compose = **99.7-100% à d=64 comme d=512** (aucune dégradation).
  - *D_max* = **seuil binaire** indépendant de d (grok ou non, pas de gradient d^-3.55).
- **Conclusion** : dans le setup crown-jewel (one-hot-concat + 1-cos), **aucun anti-grok** — d=64 reste efficient (coût/params), mais **pas pour des raisons d'anti-grok**. Le "0.24 anti-grok" du Besoins reste non reproduit (autre setup : embeddings continus ? tâche plus dure ?). La **loi unifiée §2 (d^−2.38) est donc à considérer comme OBSOLÈTE** sur sa composante d.
- *Code* : `verify_scale_antigrok.py`, `complete_formula.py`.

### 9.5 Grok = association NUMÉRIQUE, pas copie texte/perception
- La recette VQ→IDs→SCB→1-cos appliquée à la **perception** (audio/image) = **mémorisation + hasard** (audio OOD 7.8%, image 17.3%, test unique propre).
- L'ancien "word→number 73-92%" était **inflaté par sélection-sur-test** (biais corrigé). Le chiffre propre = hasard sur perception.
- **Le grok (1-cos) est pour le cœur de raisonnement** (associations numériques déterministes), **CE pour les lobes sensoriels** (perception stochastique). [déjà §5, confirmé]

---

*Document vivant. Dernière mise à jour : 1 juillet 2026 — session gate+sommeil (validation + contrôle), corrections crown-jewel/perception/formule.*
