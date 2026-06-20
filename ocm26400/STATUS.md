# OCM-26400 — Statut & Validation

**Date:** 19 Juin 2026
**Package:** `ocm26400/` (construit en TDD, 75 tests verts)

## CE QUI EST CODÉ ET VALIDÉ

Le joyau spec (Besoins_Maths.md) — auparavant **markdown seulement** — est maintenant **implémenté + démontré**. **75 tests verts.**

| Composant | Spec | Fichier | Tests |
|---|---|---|---|
| **AMV-256** | v=[ent64\|prop64\|op64\|meta64], **partition meta 3 rôles** (conf LSRA / source / consist) | `amv.py` | 7 |
| **SymbolicDict + Verifier** | V(ent,prop,op)→{0,1}, compose pluggable, **compose(a,b,op_id) dispatch** | `verifier.py` | 9 |
| **ACSP loss** | α·L_align+β·L_step+γ·L_sparse+δ·L_consist, **consist_term kwarg (contrat unique)** | `acsp.py` | 8 |
| **InfoNCE (L_consist core)** | §2.4, stable logsumexp, symétrique + multimodal | `infonce.py` | 7 |
| **ReasonerBlock + LSRA** | v(t+1)=Block(v(t)), boucle pleine (gate τ_grok + anomalie) | `reasoner.py` | 9 |
| **Contrats partagés** | compose op_id + partition meta (anti-Frankenstein) | `test_contracts.py` | 6 |
| **LearnedVocab (P2)** | dictionnaire DENSE identité-préservant, V>64, anti-collapse cosinus | `learned_vocab.py` | 14 |
| **Crown-jewel arithmétique** | décomp >> one-shot sur Z₁₁³ | `experiment_composition.py` | démontrée |
| **Crown-jewel linguistique** | décomp >> one-shot sur dérivation anglaise | `experiment_linguistic.py` | démontrée |
| **Survie dense (P2)** | crown-jewel survit one-hot→dense (ortho+random, +100pt) | `experiment_linguistic_dense.py` | démontrée |
| **Scaling V>64 (P2)** | LearnedVocab sur Z₁₂₀ (impossible one-hot), grok règle 99% | `experiment_vocab_scale.py` | démontrée |
| **Gate calibrée + abstention (P3)** | lsra_loop refuse l'OOD (ANOMALIE), 1-step hallucine | `experiment_refinement.py` | démontrée |
| **Alignement amodal** | f_T~f_A~f_V~v_C (§A1.3), InfoNCE + ancrage LearnedVocab, retrieval 100% | `concept_amodal.py` | 5 |
| **Conjugaison multi-temps (op_id)** | MorphologyVerifier dispatch PAST/GERUND/THIRD par op_id, 1 block 3 temps | `morphology.py` | 5 |
| **Récurrence fenêtrée profonde** | op^k récursif (profondeur 2-5), 100% sur chaînes non vues (raisonner longuement) | `experiment_recursion.py` | 3 |

## ROADMAP (juge experts/DA — voir `EXPERT_PANEL_VERDICT.md`)

Ordre imposé par le juge (DAG, pas de collage) :
1. ✅ **P1 InfoNCE pur + fix seam ACSP** — FAIT (consist_term contrat établi).
2. ✅ **Contrats partagés** (compose op_id + partition meta) — FAIT.
3. ✅ **P2 LearnedVocab** (dense identity-preserving, V>64, anti-collapse) — FAIT. Survie crown-jewel prouvée empiriquement (+100pt sous dense). Claim MRR sémantique RETIRÉ (falsifié). Scaling V>64 démontré (Z₁₂₀, grok règle 99% brut).
4. ✅ **P3 Test-time compute REFRAMÉ** (gate calibrée + abstention) — FAIT. Claim d'ingénierie honnête : le 1-step hallucine toujours, la boucle LSRA + gate calibrée REFUSE l'OOD (ANOMALIE 100%, AUROC 1.0). Le TTC-rafine-accuracy original était tautologique (enterré par le DA).
5. ✅ **P4 Pont v6→AMV** — CONSTRUIT (résultat honnête en deux faces). Prémices levées : P2 V>64 ✅, partition meta ✅, single-forward mesuré ✅ (96.4% ≥ diffuse 91.3%), L_step **retiré** de la loss du pont (juge b). `spxlm_v6/experiment_v6_bridge.py` : v6 gelé single-forward → tête linéaire → `LearnedVocab(80, V>64)` → AMV (meta[1]=source_conf). **(+) Interface marche sur dictionnaire fixe : 100% accuracy (raw+gated), gate dense valide, AMV consommable. (−) Ne généralise PAS vers symboles non vus (0%) — cibles denses arbitraires ⇒ pas de structure linéaire, exactement la critique du DA (« pas de 3ᵉ état stable »). Le pont est un encodeur à dictionnaire fixe, pas généralisant.** 3 tests contractuels (`spxlm_v6/test_v6_bridge.py`).

## P2 LEARNEDVOCAB — résultats honnêtes (20/06)

**Dictionnaire dense E∈R^{V×64} préservant l'identité, classe SÉPARÉE de SymbolicDict**
(les 6 fichiers one-hot existants restent verts inchangés). Decode par plus proche
voisin cosinus + marge de pureté (cos1≥0.85 ET cos1−cos2≥0.05). Anti-collapse sérieux
(uniformity_loss en cos² + garde-fou rang). **Claim sémantique MRR RETIRÉ** (falsifié :
aucun terme distributionnel dans la loss).

### 1. Survie crown-jewel : one-hot → dense (`experiment_linguistic_dense.py`)
Même tâche morpho (60 primitives), même split (20 train / 12 test jamais vus), même
block 263K params, même loss. On ne change QUE le dictionnaire (one-hot → dense gelé).

| Variant | cos inter-paires | DÉCOMP test | ONE-SHOT test | ÉCART | valid-rate |
|---|---|---|---|---|---|
| one-hot (réf.) | 0 (axes) | 100% | 0% | **+100pt** | — |
| ortho-dense (gelé) | 0 (rotated) | 100% | 0% | **+100pt** ✅ | 100% |
| random-dense (gelé) | 0.001 | 100% | 0% | **+100pt** ✅ | 100% |

**Conclusion** : le grok compositionnel ne dépend PAS de l'alignement sur les axes,
seulement de la séparabilité pairwise. Le crown-jewel survit parfaitement au dense.

### 2. Scaling V>64 (`experiment_vocab_scale.py`, Z₁₂₀)
SymbolicDict ne peut pas (assert n≤64). LearnedVocab le peut : roundtrip identité 120/120.
op(a,b)=(3a+5b) mod 120, non-associative. Block binaire entraîné sur un ÉCHANTILLON de
paires (doit grokker la RÈGLE, pas mémoriser) :

| Métrique | gated (gate stricte) | raw (argmax) |
|---|---|---|
| Grok binaire (paires jamais vues) | 40% | **99.7%** |
| DÉCOMP test (600 triples neufs) | 13% | **97.7%** |
| ONE-SHOT test | 0% | 2.7% |
| ÉCART | +13pt (gate trop stricte) | **+95pt** ✅ |

**Honnêteté** : le mécanisme crown-jewel SURVIT au niveau représentation (+95pt brut,
règle modulaire grokkée à 99.7% sur paires jamais vues). MAIS la gate stricte
(cos1≥0.85) rejette beaucoup d'outputs corrects-mais-impurs au packing V=120/dim=64.
Diagnostic compute→gate : validité 34%→54%→65% à 3000/6000/10000 pas (raw stable ~99%),
donc c'est un problème de **netteté (sharpening)**, pas de correction — la gate bute
géométriquement sur le packing 120/64. Leviers honnêtes : relaxer la gate à grand V,
monter dim, ou affûter le block. NON p-hacké (on garde le config standard 3000 pas).

## P3 GATE CALIBRÉE + ABSTENTION — résultats honnêtes (20/06)

**Reframeage honnête** : le design original de P3 (supervision d'une trajectoire
géométrique λ<0.5) était TAUTOLOGIQUE (1-pas faux et convergence par construction,
enterré par le DA). De plus, itérer le ReasonerBlock résiduel ne *raffine* pas vers
la cible — il *recompose* (prop reste b). Donc « TTC-rafine-accuracy » est incohérent
ici. La valeur HONNÊTE de la boucle LSRA (spec §3) est la **gate de confiance calibrée** :
le 1-step hallucine toujours (retourne un idx même sur du garbage) ; la boucle +
gate calibrée **REFUSE** l'OOD ([ANOMALIE_CAUSALE], confident=False).

Block calibré (Z₁₁, `experiment_refinement.py`) : entrées valides → ent=op(a,b) +
meta[0] haut ; OOD (ent=bruit) → meta[0] bas. On réutilise `lsra_loop` SANS modifier
sa signature.

| Métrique | Valeur |
|---|---|
| conf moyenne valides | 0.981 (> τ=0.9) |
| conf moyenne OOD | 0.018 (≪ τ) |
| séparation (AUROC proxy valid>OOD) | **1.000** |
| valides → accepte (confident=True) | 100% (200/200 corrects+confiants) |
| OOD → refuse (ANOMALIE) | **100%** |
| itérations moyennes (valides) | 1.00 (gate adaptative, stop anticipé) |

**Conclusion** : claim d'INGÉNIERIE validé (incertitude épistémique / abstention),
PAS crown-jewel. La gate calibrée donne à la boucle LSRA un stop conditionnel +
un signal d'anomalie que le forward fixe n'a pas.

## DÉMONSTRATION CROWN-JEWEL (validée, honnête)

### 1. Arithmétique Z₁₁ (`experiment_composition.py`)
Tâche : `op(a,b)=(3a+5b) mod 11` (non-commutative, non-associative), `r=op(op(a,b),c)`.
Train : 200 triples. Test : **1131 triples jamais vus**. Params constants (263K).

```
Bloc binaire grokké (op(a,b))      : 100.0%
ONE-SHOT train                     : 100.0%   (mémorise)
ONE-SHOT test (jamais vus)         :   0.5%   ❌ mémorisation pure
DÉCOMP LSRA test (jamais vus)      : 100.0%   ✅ crown jewel
Écart                              : +99.5 points   (33s GPU)
```

### 2. Linguistique — dérivation anglaise (`experiment_linguistic.py`)
Tâche : `prefix+stem+suffix` (un+kind+ness), composition 2-étapes sur 8 stems × 2 pref × 2 suff.
Train : 20 triples. Test : **12 triples jamais vus**. Params constants (263K).

```
ONE-SHOT train                     : 100.0%
ONE-SHOT test (triples neufs)      :   0.0%   ❌
DÉCOMP    train                    : 100.0%
DÉCOMP    test (triples neufs)     : 100.0%   ✅ crown jewel linguistique
Écart                              : +100.0 points  (27s GPU)
```

**Cause racine** (pourquoi ça marche) : one-shot doit apprendre une mapping sur l'espace produit complet (sans tout voir = mémorisation). Décomposition n'apprend que les sous-fonctions (binaire / un seul affixe) — fonctions simples grokkées — puis **compose par application**. La composition d'une fonction grokkée est exacte → généralisation gratuite.

⚠️ **Honnêteté** :
- Les écarts (0.5%/0% vs 100%) sont plus nets que le `0.75→0.98` spec, car le one-shot voit peu de données et les sous-fonctions sont simples (grok parfait). Le **principe** (décomp >> one-shot à params constants) est validé ; le chiffre exact dépend de la tâche.
- Certains mots linguistiques sont des formes régulières bien formées mais non attestées (ex. "refairful"). Objet = valider le MÉCANISME compositionnel sur structure linguistique, pas la couverture lexicale.

## PROCHAINES ÉTAPES (roadmap honnête)

POC validé sur tâche contrôlée. Pour atteindre la spec complète :
1. **LSRA pleine** : boucle jusqu'à `confidence ≥ τ_grok` avec gate réelle stop/anomalie (actuel = 2-step fixe).
2. **Dictionnaire réel** : primitives = mots/lemmes (pas Z₁₁), encodeurs ent/prop/op appris.
3. **L_consist multimodale** : InfoNCE cross-modal (texte/audio/image → même AMV).
4. **Intégration v6** : v6 (FFT spectral) comme encodeur de contexte + OCM comme cœur de raisonnement (TESTING.md : "complémentaire à SPXLM").
5. **Scale langage** : phrases, règles grammaticales comme opérateurs vérifiables.

## COMMENT REPRODUIRE
```bash
cd MathsBase
python3 -m pytest ocm26400/ -q                      # 62 tests
python3 -m ocm26400.experiment_composition          # crown-jewel arithmétique (~33s)
python3 -m ocm26400.experiment_linguistic           # crown-jewel linguistique (~27s)
python3 -m ocm26400.experiment_linguistic_dense     # survie one-hot→dense P2 (~64s)
python3 -m ocm26400.experiment_vocab_scale          # scaling V>64 (Z_120) P2 (~90s)
python3 -m ocm26400.experiment_refinement           # gate calibrée + abstention P3 (~45s)
```
Résultats : `ocm26400/{crown_jewel,linguistic,linguistic_dense,vocab_scale,refinement}_results.json`.
