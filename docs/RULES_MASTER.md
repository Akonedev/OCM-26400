# RULES_MASTER — Checklist vérifiée (audit exhaustif des Besoins + PROCEDURES + CODE)

**Date :** 27 juin 2026 | **Audit :** Training.md, Formule_Lois_Grokking.md, Grokking.md,
Besoins.md, Ressources.md, Besoins_Tests.md, Besoins_Documents.md, PROCEDURES.md,
CODE_EXACT_SPECTRALM.md, FORMULAS_AND_DISCOVERIES.md, GROKKING_CONDITIONS.md, CAPABILITIES.md.

> Toute étape d'entraînement DOIT être cochée contre cette liste. **Rien ne doit être oublié.**

---

## A. PRINCIPE ABSOLU
- [x] **Compréhension > Mémoire, TOUJOURS**, tous modes et disciplines.
- [x] Le modèle doit COMPRENDRE les règles puis GÉNÉRALISER/GÉNÉRER — pas mémoriser des instances.
- [x] **Pas de transformers, pas de Mamba, pas de Frankenstein, pas de réutilisation de modèles existants.**
- [x] **Ne JAMAIS mentir** sur les résultats (honnêteté SOTA obligatoire).
- [x] **Pas de training aveugle** (test sur échantillon ≤5 min avant gros run).

---

## B. ARCHITECTURE (constantes vérifiées dans le code)
- [x] `D_MODEL=256`, `PART=64` (4 partitions ×64), `seq_len=64`, **675K params FIXES**.
- [x] **AMV-256** = `[ent(64)|prop(64)|op(64)|meta(64)]` contigu.
  - `meta[0]`(192) = confidence (gate/observateur), `meta[1]`(193) = source, `meta[2]`(194) = consist.
  - **NE JAMAIS partager meta[0] entre deux losses.**
- [x] **SpectralCoreBlock** (FFT) : `norm1→in_proj→rfft→filtre complexe appris→irfft→out_proj→résiduel→FFN(norm2)`.
  - Filtre `filter_real=randn·(1/√D)+1.0`, `filter_imag=randn·(1/√D)` (init ≈ identité).
  - Stabilité Parseval `‖x‖²=‖FFT(x)‖²`. Complexité `O(L log L)`.
  - Robuste aux longueurs variables (pad/slice du filtre).
- [x] **LSRA** : `v(0)=encode(a,b)` ; `v(t+1)=Block(v(t))` ; stop quand `sigmoid(meta[0])≥TAU_GROK=0.9`.
  - `CONF_TARGET=4.0` (sigmoid(4)≈0.98), `max_iter=8`. Sinon `ANOMALIE_CAUSALE` → abstention.

---

## C. PROCÉDURE CANONIQUE DE GROK (`train_binary_block`) — LA SEULE pour le grok binaire
| Condition | Valeur |
|---|---|
| Fonction | `ocm26400/experiment_composition.py::train_binary_block` |
| Loss | **(1 − cos)** sur `ent[0:64]` vs `canonical(op(a,b))` — PAS MSE |
| Optimizer | **Adam** (PAS AdamW) |
| lr | **3e-3** |
| batch | **64** (128 pour refinement / omni_rules) |
| n_steps | **1500** (mono), **2000** (multi-op) |
| seed | **0** (`torch.manual_seed(0)` à chaque palier) |
| tirage | `randint(0,121,(batch,))` WITH replacement |

- [x] `op(a,b)=(3a+5b) mod 11` (NON-commutatif, NON-associatif, `A_COEF=3, B_COEF=5, P_MOD=11`).
- [x] **JAMAIS** `train_with_acsp` pour le grok binaire (donne 0.95 vs 1.00).
- [x] **JAMAIS** AdamW pour ocm26400 (AdamW = spxlm_v6, autre config : lr=1e-3 wd=0.1).

---

## D. ACCEPTATION DU GROK (critères de vérification)
- [x] `binary_acc ≥ 0.99` sur données NON vues.
- [x] `decomposition_acc ≥ 0.95` sur triples NON vus (vs oneshot ~0.5%).
- [x] **gap(decomp − oneshot) ≥ +95 pts** (crown-jewel mesuré +99.5pt).
- [x] survie du grok après one-hot → dense (+100pt, P2).
- [x] `SymbolicDict.decode` retourne `valid=True` (purity `atol=1e-3`).

---

## E. LES 6 LOIS (L1–L6) + loi unifiée
1. **L1 Décomposition > Scale** — la compétence vient des ÉTAPES, pas de la masse (0.75→0.98→100%). Scale inverse (élargir casse le grok 0.24).
2. **L2 Masquage incrémental** — masquer un SOUS-ENSEMBLE des intermédiaires → chaque étape = op 1-pas en contexte → cascade à l'inférence.
3. **L3 depth_max ≈ 1/(1−per_step)** — per-step exact → profondeur ∞ (vérifié à 100000).
4. **L4 Récurrence ⊥ Longueur ⊥ Params** — raisonner = ajouter des ÉTAPES, pas des PARAMS.
5. **L5 L = 1+4·D** — format scratchpad ; `batch·L ≈ 1.9e4` (3 Go).
6. **L6 Association** — 1-source direct (p_step>0.99) ; multi-source = DÉCOMPOSER.
- **Loi unifiée** : `D = k^1.98 × P^1.06 × d^−2.38` (γ≈1 D∝P ; δ<0 inverse du scale).
  - **VÉRIFIÉ 30/06** (voir `DECOUVERTES_LOI_SCALING.md`) : δ=−2.38 confirmé sur perception (audio d=1024→87%≪d=128→92.6%) ; L3 confirmée (chaînes k-step 100% à k=50, profondeur ∞ en raisonnement exact) ; décomposition 100% universelle (d,P quelconques) vs one-shot ~1%.

---

## F. CURRICULUM v4 (ADR-0030) — `primitive_grok_curriculum.py`
- [x] **Phase 1 SOLO** : chaque opérateur grok INDIVIDUELLEMENT à **gate L1≥0.99** (2k-6k steps, `train_binary_block`).
- [x] **Phase 2 SOMMEIL** : **OBLIGATOIRE** (transforme mémoire→compréhension, `sleep_phases`, comble oubli catastrophique 89.8%).
- [x] **Phase 3 CASCADE** : scratchpad (intermédiaire PUIS final) → composition profonde 100%.
- [x] **Gates** : **L1≥0.99, L2≥0.95, L5≥0.90, L6≥0.85**.
- [x] **Anti-shortcut asymétrique** : masquer TOUTES les variables algébriquement récupérables depuis la cible ; JAMAIS masquer les 18/18 ; chaque tâche garde son anti-shortcut en interleaved.

---

## G. SOMMEIL — 3 PHASES (NON optionnel)
- [x] **Léger (macro)** : rétrospection, basses fréquences, relations grossières.
- [x] **Moyen (consolidation)** : entropie des poids, fréq. moyennes, régularités, règles.
- [x] **Profond (micro)** : hautes fréquences, nuances, substitués, équivalences.
- [x] Analyse d'entropie : identifier info concentrée / redondante / manquante → consolider.

---

## H. IDs NUMÉRIQUES (PRINCIPE FONDATEUR)
- [x] **TOUT convertir en IDs numériques discrets avant le SpectralCoreBlock.**
- [x] Le grokking marche parce que c'est une **ASSOCIATION entre NOMBRES**, pas une copie de signal.
- [x] ✅ texte→word_ID, phonème→phoneme_ID, video→frame_ID, 3D→voxel_ID → grok 100%.
- [x] ❌ audio→Mel float → pas de grok (signal continu stochastique) → utiliser IDs phonétiques + génération depuis règles.

---

## I. CAPTURE SIMULTANÉE (associations, en UNE passe)
- [x] Capturer en une fois : grammaire, vocabulaire, phonèmes, lexique, phonologie, morphologie, étymologie, affixes, morphèmes, syntaxe, conjugaison, sens, synonymes, nuances + audio + image + video + 3D + world.
- [x] **40+ niveaux linguistiques/conceptuels** simultanés (word_id, plural_id, tense_id, ..., syntax_role_id).
- [x] Champ partagé diffusion-fill (mécanisme validé 6/8 niveaux, multi-source=décomposer).

---

## J. GATES + LEAN + OBSERVATEUR
- [x] **Gate** `meta[0]` → entraîné vers `CONF_TARGET=4.0`. Loss = `(1-cos) + (out[192]-4.0)²`.
- [x] **Observateur** : confiant ET correct sur NON-vus = COMPRÉHENSION ✓ ; confiant ET faux = surconfiance ✗ ; non-confiant = abstention.
- [x] **LEAN** : peu d'exemples (10-200) + peu de params (675K) + peu de steps (1500) + grokking. (Preuve : 105k samples→31% vs 200 triples→100%.)

---

## K. GÉNÉRATION DEPUIS COMPRÉHENSION (crown-jewel inversé)
- [x] 3 étapes simultanées : (1) COMPRENDRE primitives→concept (1-cos), (2) GÉNÉRER concept→signal (MSE), (3) VÉRIFIER signal→concept.
- [x] **Loi d'asymétrie** : GÉNÉRER depuis règles (78-100%) >> RECONNAÎTRE depuis signal (0.5-43%).
- [x] Flow-matching : `x_t=(1-t)x_0+tx_1`, `v=x_1-x_0`, MSE sur vélocité, Euler 8 steps.

---

## L. ACSP (complémentaire, PAS pour le grok binaire)
- [x] `L = α·L_align + β·L_step + γ·L_sparse + δ·L_consist` avec α=1.0, β=1.0, γ=1e-3, δ=0.0.
- [x] **L_step différentiable via Gumbel straight-through** (`diff_decode.py`) — `decode_gumbel` hard=True.
- [x] Sanity RED : sur batch parfait, loss < 0.05.
- [x] InfoNCE `tau=0.07` (CLIP/SigLIP).

---

## M. SANITY CHECKS (avant tout long training)
- [x] **SC-1** : overfit 1 batch (100 steps), loss < 0.01.
- [x] **SC-2** : validation du format (champs alignés, chaque intermédiaire EXACTEMENT 1 fois).
- [x] **SC-3** : masque incrémental (`single_frac ≈ 0.3 ± 0.05`).
- [x] **SC-4** : VRAM (`assert mem_allocated < 3e9`).
- [x] **SC-5** : per-field acc à step 500 (chaque champ > 0.05).
- [x] **SC-6** : cascade eval à step 1000 (`A_cascade > 0.1`).
- [x] Générer du texte/artefacts tous les **500 steps** pour détecter les problèmes tôt.

---

## N. DOMAINS / MODES À ENTRAÎNER (sur VRAIES données)
- [x] **D1=Maths, D2=Code, D3=Science** (TRAINING_PROTOCOL étape 3) sur vraies données.
- [x] Arithmétique, logique, morphologie (EN/FR), phonétique, composition.
- [x] Physique, chimie, biologie, médecine, histoire, géo, finance, stat, théorie des jeux, crypto, info.
- [x] Audio (SpeechCommands 39 mots), Vision (tinyimagenet/MNIST), Video, 3D, World.
- [x] GSM8K (NL→CoT→arithmétique), CommonSenseQA.
- [x] **Benchmarks cibles** : HLE, AIME 2026, GPQA-Diamond, SWE-bench, Terminal Bench, MCP-Atlas, Tool-Decathlon, BLiMP.

---

## O. AGENTS / VALIDATION / DEVIL'S ADVOCATE
- [x] Orchestrateur + sous-agents : communiquent, se challengent, valident ensemble.
- [x] **Devil's Advocate** : BRUTAL mais juste, ne laisse RIEN passer, classifie (CRITIQUE/MIEUX/GONFLER/REJETER), 3 priorités vraies.
- [x] **Juges** : vérifient tous les besoins implémentés, best practices, tests Unit+Integration.
- [x] Boucle : recherche → tests → DA → juges → validation ; si No-Go, corriger/recommencer.

---

## P. ERREURS À ÉVITER (PROCEDURES §5 + CODE §19)
1. Pas de conjugaison par flat-map (décomposer stem+affixe).
2. Pas de morphologie char-level dans le FFT (IDs numériques + SBS).
3. Gate stricte à V=120/dim=64 = problème de sharpening, pas de correction (relaxer/augmenter dim/sharpen).
4. `compose(a,b,op_id=0)` signature UNIFIÉE partout.
5. ACSP : passer `consist_term` en kwarg, jamais `l_consist()` sans args.
6. Init explicite `std=0.02` (pas LazyLinear). Sous ROCm : pad à `seq_len` fixe.
7. Un seul espace (cosinus unit-norm) pour `delta_m`. Decode `cos1≥0.85 AND cos1-cos2≥0.05`.
8. Loss d'uniformité anti-collapse (test de rang insuffisant).
9. Pin SymbolicDict one-hot en regression test.
10. P3 = gate calibrée + abstention (PAS "TTC improves accuracy").
11. Anti-shortcut asymétrique seulement (jamais masquer tout).
12. NE JAMAIS citer "92% v6" (vérité : best_avg=0.695, single-forward 50-60%).
13. Masquer `c+d+m1+m2+ans` (récupérables depuis c), PAS le stem.
14. Phases arithmétiques : **≥6000 steps** (pas 4000) ; prose : 2000.
15. v6 `causal_weight` doit être 2D `(F,1)` ; v6 = raisonneur, pas générateur.
16. `d_model=256` optimum ; `d=768` → 0% compréhension (NON-monotone).
17. ASCII `-` (pas U+2212).
18. FFT mixer per-dimension filter (pas un filtre global unique).
19. Loss non-différentielle = décorative (Gumbel ST obligatoire pour L_step).

---

## Q. LICENCE
- [x] **Commerciale "Lobe Licensing"** : Sensory Lobes open-source, Reasoning Core + Semantic Memory COMMERCIAL, crown-jewel prior art défensif. (PAS MIT.)

---

## R. ÉTAT HONNÊTE ACTUEL (CAPABILITIES.md, à améliorer)
| Domaine | Score | SOTA | Gap | Status |
|---|---|---|---|---|
| Arithmétique crown-jewel | 100% | 100% | 0 | ✅ SOTA |
| Logique | 100% | 100% | 0 | ✅ SOTA |
| Morphologie EN | 100% | 100% | 0 | ✅ SOTA |
| Composition (non-vus) | 100% | 100% | 0 | ✅ SOTA |
| Image classification (synth) | 89.5% | ~90% | -0.5 | ✅ ~SOTA |
| Audio génération | 97% | N/A | — | ✅ |
| **Audio reconnaissance** | **42.7%** | **96%** | **-53** | ⚠️ gap |
| **GSM8K** | **3.5%** | **95%** | **-91** | ❌ frontier |
| Tests | 1137/1137 | — | — | ✅ |

**Frontières réelles à pousser** : (1) GSM8K NL→CoT, (2) audio reconnaissance stochastique, (3) vision réelle, (4) AIME.
