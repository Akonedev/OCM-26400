# Session 2026-07-01 — GATE, SOMMEIL, CORRECTIONS

> Consolidation des découvertes de la session (pour ne rien perdre). Source de vérité mise à jour : `SOLUTION_OCM26400.md` §9. Ce doc = récap rapide + code + principes corrigés.

---

## 1. CE QUI A ÉTÉ VALIDÉ (3 piliers)

### 1.1 La GATE fait grokker → composition arbitraire ✅
**Thèse** : *la capacité compositionnelle arbitraire émerge d'UNE primitive grokkée + la gate qui certifie.*

- Grok 1 primitive `op(a,b)=(a+b) mod 23` (1-cos, crown-jewel) → 100% en 1000 steps.
- **Gate** = alignement cosinus de la sortie au dictionnaire canonique (= L_align du Besoins). `≥ τ` → étape certifiée.
- Cascade à profondeur D, gate certifie chaque étape : **100% à D=50** (gate ≥ 0.987).
- **Le grok = la gate, PAS le scale.** Raisonner = enchaîner des étapes certifiées (pas empiler des params).
- Code : `grok_gate_composition.py`.

### 1.2 Le sommeil spectral fonctionne ✅ (corrige la réfutation antérieure)
**Thèse** : *le sommeil transforme la MÉMOIRE en COMPREHENSION.*

| Bras | test acc |
|---|---|
| Éveil (1500 steps, mémorisation) | 25.7% ± 2.4 |
| Baseline (+2000 steps PURS, sans sommeil) | 26.0% ± 2.5 (**STUCK**) |
| **Sommeil** (5 cycles spectraux + replay) | **99.1% ± 0.7** |

- **Δ sommeil − pur = +73pt.** Le sommeil fait ce que l'entraînement pur NE PEUT PAS.
- **PAS du grokking retardé classique** : là, le pur finit par grokker ; ici le pur **reste stuck** — seul le filtrage spectral débloque (mécanisme distinct).
- **Ablation** : replay_seul = +0pt (inutile) ; high-pass_seul = −2pt ; **low-pass_seul = +17pt (ingrédient actif)** ; léger+profond = +32pt/cycle.
- **Intensité** : keep_frac low-pass = pic **étroit à 0.5** (0.3→19%, 0.5→60%, 0.7→37%).
- **Cycles composent** : 1→60% | 2→94% | 3→95% | 5→99%.
- **Mécanisme** : le low-pass (FFT des poids, zero HF) détruit la mémorisation "spiky" → le modèle retombe dans le bassin généralisant → replay affine.
- Code : `test_sleep_neural.py`, `optimize_sleep.py`, `control_sleep_vs_puresteps.py`.

### 1.3 Grok = association NUMÉRIQUE (pas perception/texte) ✅
- Recette VQ→IDs→SCB→1-cos sur **perception** (audio/image) = mémorisation + hasard (audio OOD 7.8%, image 17.3%, test propre unique).
- L'ancien "word→number 73-92%" = **inflaté par sélection-sur-test** (biais corrigé). Chiffre propre = hasard.
- **1-cos = cœur de raisonnement** (associations déterministes) ; **CE = lobes sensoriels** (perception stochastique). Ne pas mélanger.

---

## 2. CE QUI A ÉTÉ CORRIGÉ / RÉFUTÉ

### 2.1 Crown-jewel = COMPOSITION, pas extrapolation
- Composition (triples/cascade non-vus via décomposition) = **100%** → c'est le grok au sens Besoins.
- Grok **pur** (extrapolation aux paires atomiques tenues secrètes) = **0-8% TOUS setups** (1-cos+one-hot, CE+dense+wd{0,1e-3,1e-2,1e-1}, SCB L=P). Mémorisation parfaite (100% train), 0% généralisation atomique.
- **"Grok" dans le Besoins = composition** (primitives à 1.0 + cascade scratchpad). L'extrapolation Power-2022 est plus forte, non atteinte — et **non requise** (la composition suffit via la gate).

### 2.2 Formule scale=anti-grok — PARTIELLEMENT vérifiée
- Formule raffinée (utilisateur) : `D = k^3.5 · d^−3.55 · T^2.06`.
- **T aide** (c>0, direction confirmée) : +de steps → +de profondeur.
- **d^−3.55 RÉFUTÉ au niveau primitive** : un +gros modèle grok la primitive **plus vite** (d=256 grok à T=40 où d=32 échoue ; exposant mesuré b=+1.55). L'effet anti-grok du Besoins ("élargir casse le grok, 0.24") = composition **one-shot** → à tester séparément.
- Code : `verify_scale_antigrok.py`.

---

## 3. PRINCIPES MIS À JOUR (résumé exécutif)

| Principe | Avant session | Après session |
|---|---|---|
| **Grok** | phase transition 1-cos | = **gate qui certifie** + composition. Extrapolation atomique = non atteinte/non requise |
| **Composition** | décomposition 100% triples | **100% à profondeur arbitraire (D=50)** via gate |
| **Sommeil** | "implémenté" (3 phases symboliques) | **VALIDÉ neural : +73pt, essentiel** (le pur stuck). low-pass=ingrédient actif, 5 cycles → 99% |
| **Scale** | anti-grok (d^−2.38) | anti-grok = composition one-shot uniquement ; **primitive : d AIDE** |
| **Perception** | recette IDs→SCB→1-cos | = mémorisation sur perception ; **CE requis** sur lobes, 1-cos sur cœur |
| **Pipeline** | Apprendre→Comprendre→Raisonner | **Éveil(mémoire) → Sommeil(grok) → Gate → Compose** |

---

## 4. FORMULES (à jour)

```
Loi unifiée (perception/raisonnement)  : D = k^1.98 · P^1.06 · d^−2.38   (anti-grok = composition one-shot)
Formule raffinée (utilisateur)         : D = k^3.5  · d^−3.55 · T^2.06    (T confirmé, d réfuté sur primitive)
Profondeur max (L3)                    : depth_max ≈ 1/(1 − per_step_err)
Gate                                   : certifié si cos(sortie, canonique) ≥ τ   (= L_align)
Loss cœur                              : L = 1 − cos(ent, target)
Sommeil (config optimale)              : low-pass keep_frac=0.5 → replay 200 → high-pass 0.3 → replay 200, ×5 cycles
```

---

## 5. CODE DE LA SESSION (tous dans ocm26400/)

| Script | Rôle | Résultat |
|---|---|---|
| `grok_gate_composition.py` | gate + cascade profondeur D | 100% à D=50 |
| `test_sleep_neural.py` + `_seeds.py` | sommeil 3 phases (5 seeds) | 99.1% ± 0.7 |
| `optimize_sleep.py` | ablation + intensité + cycles + courbe grok | config optimale trouvée |
| `control_sleep_vs_puresteps.py` | contrôle sommeil vs pur +2000 steps | +73pt (pur stuck) |
| `verify_scale_antigrok.py` | vérif formule D=k^3.5·d^-3.55·T^2.06 | T ok, d réfuté (primitive) |
| `grokking_heldout.py`, `grokking_canonical_ce.py`, `grokking_scb.py` | tests grok pur | 0-8% (réfutent extrapolation) |
| `sleep_phases.py` | sommeil symbolique (extraction règle) | règle (α,β) extraite |

---

## 6. PROCHAIN PAS — ASSEMBLAGE

Pipeline unifié `Apprendre → Comprendre → Raisonner` (noyau SCB) :
```
ÉVEIL (train, mémoire) → SOMMEIL (5 cycles spectraux → grok, gate émerge)
                       → GATE (certifie chaque étape) → COMPOSE (cascade arbitraire)
```
Tâche cible : primitive arithmétique grokkée imparfaitement à l'éveil (stuck), consolidée par sommeil (gate→0.99), puis composée à profondeur arbitraire via gate. Démontre end-to-end mémoire→sommeil→compréhension→raisonnement dans UNE boucle.

### ✅ ASSEMBLAGE RÉALISÉ (`assemble_pipeline.py`)
Pipeline `Éveil → Sommeil → Gate → Compose` validé end-to-end :

| Démo | éveil (acc/gate) | sommeil (acc/gate) | compose |
|---|---|---|---|
| 1 crown-jewel (raisonnement) | 100% / 0.994 | 100% / 0.998 | D=20 → 100% (gate 0.996) |
| 2 seq-rule (sleep-nécessaire) | **28% / 0.882 (stuck)** | **99% / 0.995 (grok)** | — |

**Finding clé** : la **gate = SIGNAL DE COMPRÉHENSION**. Avant sommeil (mémoire, 28%) → gate 0.882 (incertain). Après sommeil (grok, 99%) → gate 0.995 (certifié). La gate détecte la compréhension **sans regarder l'accuracy test** → utilisable dans la boucle d'entraînement (`sommeil jusqu'à gate ≥ τ`). Le pipeline gère les deux régimes : primitives faciles (grok à l'éveil, sleep affine la gate) et règles dures (stuck → sleep débloque). Compose certifié à profondeur arbitraire.

---

---

*Mémoires associées : `crown-jewel-composition-not-extrapolation`, `neural-sleep-3phase-works`, `1cos-grok-perception-vs-reasoning`.*
