# Conditions de GROKKING — OCM-26400

**Le grokking est NON-NÉGOCIABLE.** Sans ses conditions, le modèle mémorise (one-shot 0.5%)
au lieu de généraliser (décomposition 100%). Ce doc liste les conditions exactes et la
conformité de chaque module d'entraînement.

## Conditions canoniques (PROCEDURES.md §2)

Pour produire le **crown-jewel** (grok binaire → composition → généralisation 100% sur
hold-out), le pre-training DOIT utiliser `train_binary_block` :

| Condition | Valeur | Rôle |
|---|---|---|
| Fonction | `train_binary_block` | l'unique procédure de grok binaire |
| Loss | **(1 − cos)** sur `ent[0:64]` vs `canonical(op(a,b))` | alignement, PAS MSE |
| Optimizer | **Adam** (PAS AdamW) | canonique ocm26400 |
| lr | **3e-3** | (AdamW spxlm_v6 = 1e-3 wd=0.1 — NE PAS confondre) |
| batch | **64** | (sauf refinement/omni_rules = 128) |
| n_steps | **1500** (multi-op = 2000) | assez long pour le grok |
| seed | **0** (`torch.manual_seed(0)`) | reproductibilité |
| tirage | `randint(0,121,(batch,))` WITH replacement | couverture aléatoire des paires |

## Vérification du grok (critères d'acceptation)

Le grok a eu lieu ssi :
1. `binary_acc ≥ 0.99` sur données NON vues
2. `decomposition_acc ≥ 0.95` sur triples NON vus (vs one-shot ~0.5%)
3. `gap(decomp − oneshot) ≥ +95 points` (crown-jewel, mesuré +99.5pt)
4. survie du grok après passage one-hot → dense (+100pt, P2)

## Conformité des modules d'entraînement

| Module | Procédure | Conformité grok | Rôle |
|---|---|---|---|
| `experiment_composition.train_binary_block` | CANONIQUE (1-cos, Adam 3e-3, seed 0) | ✅ **RÉFÉRENCE** | produit le grok 100% |
| `train.py` stage 1 | appelle `train_binary_block` | ✅ | orchestrateur conforme |
| `neural_multihop.py` | appelle `train_binary_block` (8 opérateurs) | ✅ | compétence neurale 100% |
| `diff_decode.train_with_acsp` | ACSP différentiable (Gumbel ST), Adam 3e-3 | ⚠️ complément | loss ACSP pour L_step (ne PAS utiliser pour le grok binaire — voir audit) |
| `text_decoder.CharGenerator` | Adam 3e-3, cross-entropy | ⚠️ contexte différent | génération de TEXTE (CE adapté, pas le grok mod-n) |
| `continual_learning.demo_ewc` | Adam **3e-3** (corrigé), MSE + pénalité Fisher | ⚔️ orthogonal | EWC = régularisation anti-oubli, ne remplace PAS le grok |
| **`primitive_grok_curriculum`** | **ADR-0030 v4** : SOLO (train_binary_block §2) + sommeil + **scratchpad cascade** | ✅ **NOUVEAU** | curriculum v4 complet, cascade 100% depth 3 |
| `gsm8k_spectral` | SpectralCoreBlock encoder (noyau FFT, PAS transformer) | ✅ archi corrigée | NL→CoT (toujours en amélioration) |

## Les 6 LOIS du projet (Besoins/Formule_Lois_Grokking.md)

1. **L1 DÉCOMPOSITION > SCALE** : la compétence vient de la STRUCTURE du calcul (étapes),
   pas de la masse. Scratchpad (calcule l'intermédiaire m=(a∘b) PUIS le final r=(m∘c))
   fait bondir la généralisation 0.75→0.98→100%. Scale inverse (élargir casse le grok 0.24).
2. **L2 MASQUAGE INCRÉMENTAL** : masquer un SOUS-ENSEMBLE des intermédiaires (partiellement
   visibles) apprend chaque étape comme une op 1-pas en contexte → cascade à l'inférence.
3. **L3 DEPTH_MAX ≈ 1/(1−per_step)** : per-step exact (0 erreur) → profondeur ∞ (vérifié à 100000).
4. **L4 RÉCURRENCE ⊥ LONGUEUR ⊥ PARAMS** : raisonner = ajouter des ÉTAPES, pas des PARAMS.
5. **L5 L = 1+4·D** : format de séquence ; batch ≈ 1.9e4/L (3 Go).
6. **L6 ASSOCIATION** : 1-source direct (p_step > 0.99) ; multi-source = DÉCOMPOSER.

**Loi unifiée empirique** : D = k^1.98 × P^1.06 × d^−2.38 (γ≈1: D∝P ; δ<0: inverse du scale).

## Curriculum v4 (ADR-0030, primitive_grok_curriculum.py)

```
Phase 1 SOLO     : chaque opérateur grok INDIVIDUELLEMENT à gate L1≥0.99 (2k-6k steps,
                   train_binary_block §2 : 1-cos, Adam 3e-3, seed 0).
Phase 2 SOMMEIL  : OBLIGATOIRE (transforme mémoire→compréhension, sleep_phases).
Phase 3 CASCADE  : scratchpad (intermédiaire puis final) → composition profonde 100%.
Gates            : L1≥0.99, L2≥0.95, L5≥0.90, L6≥0.85.
```

## Règles à respecter (leçons)

1. **Le grok binaire = `train_binary_block` UNIQUEMENT.** Toute autre loss/procédure →
   mémorisation, pas généralisation (leçon audit + expérience : train_with_acsp donnait
   0.95 vs train_binary_block 1.00).
2. **Adam PAS AdamW** pour ocm26400. (AdamW = spxlm_v6, autre config.)
3. **seed 0** à chaque palier (reproductibilité absolue).
4. **Les techniques annexes** (EWC, text-decoder CE, flow-matching) sont ORTHOGONALES au
   grok : elles s'appliquent par-dessus ou sur d'autres tâches, jamais ne le remplacent.
5. **Vérifier le gap decomp−oneshot** : s'il s'effondre, le grok n'a pas eu lieu (raccourci).
6. **Le sommeil n'est PAS optionnel** — c'est ce qui transforme la MÉMOIRE en COMPREHENSION.

## Reproduction du grok (commande)

```bash
python3 -m ocm26400.experiment_composition     # crown-jewel canonique (grok + composition)
python3 -m ocm26400.neural_multihop            # compétence neurale hold-out (8 opérateurs)
python3 -m ocm26400.train --full --stages 0,1  # pipeline conforme (grok 1.00)
```
