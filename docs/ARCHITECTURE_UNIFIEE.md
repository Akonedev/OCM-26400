# ARCHITECTURE UNIFIÉE OCM-26400 — décision canonique (B')

> **Décision validée (1 juillet 2026, panel experts + littérature + biologie + user).**
> Le modèle est **UNIFIÉ** (pas Frankenstein) via l'architecture **B' : lobes spectraux minces → cœur SCB unique + mentalese AMV → lobes inverses**.
> 100% spectral (FFT/DCT/STFT/SH/Laplacien), **pas de transformers**.

---

## La décision : B' (lobes minces → cœur unifié)

**Pourquoi pas (A) pur-unifié ?** Chaque signal a une **base naturelle** différente (audio=FFT, image=DCT, 3D=harmoniques sphériques, texte=discret). Un encodeur unique devrait réinventer ces bases → gaspille la capacité. La biologie a refusé l'unification périphérique (rétine ≠ cochlée). Mirage computationnel.

**Pourquoi pas Frankenstein ?** Encodeurs profonds spécialisés (ViT/ResNet) + fusion superficielle = Late-fusion. Shukor et al. 2025 (Apple, ICCV) : *"no inherent advantage to late-fusion"* — l'early-fusion bat à échelle modérée (notre régime).

**B' = le cerveau** : capteurs spécialisés minces (rétine/cochlée) → cortex associatif unifié. L'unification est **au mentalese (AMV) + cœur**, pas au capteur.

---

## Le pipeline architectural

```
[MODALITÉ IN] → [LOBE SPECTRAL MINCE] → [AMV-256] ──┐
  audio   → STFT (FFT native)        → conv  → AMV   │
  image   → DCT 2D / patch-FFT       → proj  → AMV   │
  vidéo   → FFT 3D (spatial×temps)   → proj  → AMV   ├→ [CŒUR SCB UNIQUE] → [DÉCODEUR AMV] → [LOBE INVERSE] → [MODALITÉ OUT]
  3D      → harmoniques sphériques   → proj  → AMV   │   (early-fusion,      (mentalese)     (iFFT/iDCT/vocodeur)
  texte   → tokenizer (discret)      → embed → AMV   │    O(L log L))
  world   → Laplacien spectral       → proj  → AMV   ┘
                                                       mentalese partagé [ent|prop|op|meta]
```

### Lobe spectral mince = transducteur (rétine/cochlée)
- **2-4 couches max**, transduction signal→IDs, **sans raisonnement sémantique** (ça = le cœur).
- Exploite la **base native** du signal (FFT/DCT/SH/wavelet).
- Émet l'**AMV-256 partagé** (pas d'espace privé).
- **Test lobe-vs-Frankenstein** : remplaçable sans retrain le cœur (émet AMV standard) → lobe. Sinon → Frankenstein.

### Cœur SCB = cortex unifié
- **1 SpectralCoreBlock** (d=64, crown-jewel) + **boucle LSRA** (raisonnement, profondeur ∞ L3).
- **Amodal** : opère sur l'AMV, ignore la modalité d'origine.
- **Loss 1-cos** (grok, raisonnement déterministe) — 100% décomposition, chaînes k=500.
- **Gate/observer** (alignement canonique L_align, τ=0.9) — **la gate fait grokker** : 1 primitive grokkée + gate qui certifie → composition à profondeur arbitraire (100% à D=50, `grok_gate_composition.py`). La gate = **signal de compréhension** (0.88 stuck → 0.995 grok).

### Décodage omni-out
- AMV → **lobe inverse spectral** par modalité (iFFT audio→waveform, iDCT→pixels, tokens→texte).
- **Chat** = sortie texte (AMV → décodeur texte). Une des sorties omni-out.

---

## La répartition des loss (validée empiriquement)

| Composant | Loss | Rôle |
|---|---|---|
| **Cœur SCB** (raisonnement) | **1-cos** (grok) | décomposition, généralisation exacte |
| **Lobes sensoriels** (perception) | **CE** | transduction signal→IDs, perception stochastique |

> Ne pas inverser : 1-cos sur perception plafonne (audio 58% vs CE 94%) ; CE sur le cœur casse le grok.

---

## Le pipeline d'entraînement (comprehension → chat) — pipeline unifié VALIDÉ

```
ÉVEIL    (train → mémoire, souvent stuck sur les règles dures)
  ↓
SOMMEIL  (3 phases spectrales : low-pass débruit + high-pass affine + replay, 5 cycles)
         → grok : la gate ÉMERGE (0.88 stuck → 0.995 compris). +73pt vs pur (pur stuck).
  ↓
GATE     (alignement canonique ≥ τ → certifie chaque étape)
  ↓
COMPOSE  (cascade gate-certifiée, profondeur arbitraire — 100% à D=50)
```

**Assemblé et validé end-to-end** (`assemble_pipeline.py`, 2026-07-01) :
- Le sommeil spectral **fonctionne** (≠ réfutation antérieure) : le low-pass détruit la composante HF de mémorisation → le modèle sort du bassin de mémorisation (min local aigu que le SGD pur ne quitte pas) → replay affine. 5 cycles → 99%.
- La **gate = signal de compréhension** : détecte le grok sans accuracy test → utilisable comme stop-criterion (`sommeil jusqu'à gate ≥ τ`).

Phases cognitives (macro) :
```
Phase 1 : COMPRENDRE  — grok primitives → compositions (éveil + sommeil jusqu'à gate≥τ)
Phase 2 : MÉMORISER   — engrames épisodiques + sommeil (replay + filtrage spectral, VALIDÉ)
Phase 3 : RAISONNER   — LSRA + gate + compose (cascade certifiée)
Phase 4 : RESTITUER   — génération → chat (sortie omni-out)
```

**Méthode** (Besoins) : *pré-entraîner les PRIMITIVES jusqu'au grok, PUIS les compositions → la maîtrise émerge.* Le sommeil accélère/permute ce grok sur les tâches où l'éveil reste stuck.

---

## Références (décision étayée)
- **Shukor et al. 2025** (Apple, ICCV) — early-fusion bat late-fusion, *"no inherent advantage to late-fusion"*. [arXiv:2504.07951](https://arxiv.org/abs/2504.07951)
- **Chameleon/Transfusion/Show-O** (Meta 2024) — early-fusion, un seul backbone (mais transformers ; nous = spectral).
- **GFNet/FNet** — mixeur FFT remplace l'attention (prouve le cœur spectral viable).
- **Biologie** — rétine/cochlée spécialisées → cortex unifié (Sci. Adv. 2025, inverse effectiveness).
- **Trou de littérature** : aucun multi-modal spectral pur n'existe → **OCM-26400 occupe cette niche** (opportunité publication).

---

## Décisions réfutées / validées (état mis à jour 2026-07-01)

**✅ VALIDÉ (était à tort réfuté)** :
- **Sommeil spectral** — l'ancien test 3-bras (Δ0.0pt) utilisait un mécanisme trop faible. Le **filtrage spectral 3 phases** (low-pass débruit + high-pass affine + replay, 5 cycles) donne **+73pt** vs pur (pur stuck 26%, sommeil 99%). Le low-pass est l'ingrédient actif (replay_seul=+0pt). → **sommeil = filtrage spectral + replay** (PAS replay brut). Voir `SOLUTION_OCM26400.md` §9.2.

**❌ Réfuté (ne pas re-coder)** :
- Entropie active learning — test 3-bras : −1.6pt overall = défavorable. → sampling uniforme.
- **Scale=anti-grok (d^−3.55 / d^−2.38)** — **RÉFUTÉ** dans le setup crown-jewel (one-hot-concat + 1-cos) : `complete_formula.py` montre D_max = **seuil binaire de grok** (indépendant de d), one-shot compose 100% à d=64 comme d=512. d AIDE la primitive (grok + vite). L'anti-grok n'apparaît ni sur la primitive ni sur la composition one-shot. (Le "0.24" du Besoins = autre setup, non reproduit.) → d=64 reste efficient mais **pas pour des raisons d'anti-grok**.
- Empiler blocs (12/24) — L4 : 1 bloc = 8 blocs pour le raisonnement. → 1 bloc.
- Ratio P/d=0.1875 — réfuté (P=35 fixe, pas un pic). → d=64 fixe.

---

*Source : `SOLUTION_OCM26400.md` (théorie), `REPRODUCIBILITE_OCM26400.md` (re-création). Ce doc = l'architecture officielle.*
