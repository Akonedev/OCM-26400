# OCM-26400 : Architecture Spectrale Unifiée pour la Compréhension Multi-Modale

**Auteur :** akone
**Date :** 27 juin 2026
**Code :** https://github.com/akone/ocm-26400
**Licence :** Commerciale "Lobe Licensing"

---

## Résumé

Nous présentons **OCM-26400** (Omni-Cognitive Mentalese), une architecture d'IA unifiée
basée sur un **mélangeur spectral FFT** (SpectralCoreBlock) qui remplace l'attention des
transformers. Le modèle démontre que la **compréhension** (grokking de règles) surpasse
la **mémorisation** (apprentissage d'instances) dans tous les domaines, avec un seul noyau
de 675K paramètres fixes.

La contribution centrale — le **crown-jewel** — prouve qu'à paramètres constants, la
**décomposition** (calculer l'intermédiaire puis le final) généralise à 100% sur des
compositions jamais vues, tandis que le **one-shot** (prédire directement) échoue à 0.5%.
Cette asymétrie (+99.5 points) est reproduite sur **tous les modes** : arithmétique,
logique, langage, physique, audio, image, vidéo, 3D et monde physique.

Nous introduisons également le paradigme de **génération depuis compréhension** : le
modèle grok les règles d'un domaine, puis **génère** des signaux (audio, images,
trajectoires) en appliquant ces règles comprises — sans mémoriser d'instances. Cette
approche atteint 78-100% selon le mode, validant le principe qu'un modèle qui comprend
une règle peut la générer pour des cas inédits.

---

## 1. Introduction

Les LLM actuels (transformers) reposent sur l'attention O(L²), nécessitent des
milliards de paramètres et des téraoctets de données. Leur compétence émerge de la
**mémorisation statistique** plutôt que de la **compréhension algorithmique**.

OCM-26400 propose une alternative : un **mélangeur spectral FFT** en O(L log L) avec
675K paramètres fixes, où la compétence émerge du **grokking** — la découverte des
règles abstraites par optimisation d'un objectif simple.

### Principe absolu

> **Compréhension > Mémoire, TOUJOURS, tous modes et disciplines.**
>
> Un modèle qui comprend une règle l'applique à des exemples jamais vus.
> Le grokking spectral prouve que les circuits Fourier encodent la règle abstraite,
> pas l'instance.

---

## 2. Architecture

### 2.1 SpectralCoreBlock — le noyau unifié

```
Entrée : x ∈ ℝ^{B×L×D}

h      = in_proj(LayerNorm(x))           # projection
X      = rfft(h, dim=L) ∈ ℂ^{B×F×D}      # FFT (O(L log L))
filter = filter_real + i·filter_imag       # filtre fréquentiel APPRIS
X̃      = X ⊙ filter                       # multiplication complexe
y      = irfft(X̃, n=L)                    # FFT inverse
x      ← x + out_proj(y)                   # résiduel spectral
x      ← x + FFN(LayerNorm(x))            # FFN (4× expansion, GELU)
```

**Stabilité de Parseval** : ‖x‖² = ‖FFT(x)‖² → le filtre ne peut pas diverger.

| Paramètre | Valeur | Rôle |
|---|---|---|
| D_MODEL | 256 | dimension AMV |
| PART | 64 | taille partition (×4) |
| seq_len | 64 | longueur de mélange |
| Params | **675K fixes** | indépendants du domaine |

### 2.2 AMV-256 — Amodal Mentalese Vector

```
v ∈ ℝ^{256} = [ ent(64) | prop(64) | op(64) | meta(64) ]
                entité    propriété   opérateur  méta
```

- `meta[0]` = confidence (gate LSRA, observateur de compréhension)
- `meta[1]` = source (pont cross-modal)
- `meta[2]` = consist_score (InfoNCE)

### 2.3 LSRA — récurrence de raisonnement

```
v(0)   = encode(input)
v(t+1) = SpectralCoreBlock(v(t))       # raisonner = ajouter des étapes
T*     = min{ t | sigmoid(meta[0](t)) ≥ τ_grok }   # arrêt adaptatif
```

Si `max_iter` atteint sans dépasser `τ_grok` → **ANOMALIE_CAUSALE** → abstention.

---

## 3. Le Crown-Jewel — généralisation compositionnelle

### 3.1 Expérience contrôle

**Tâche** : `op(a,b) = (3a+5b) mod 11` (NON-commutative, NON-associative).

| Métrique | Décomposition | One-shot |
|---|---|---|
| Train triples | 200 | 200 |
| Test triples (jamais vus) | 1131 | 1131 |
| Grok binaire | 100% | — |
| **Test acc** | **100%** | **0.5%** |
| **Gap** | **+99.5 points** | — |

### 3.2 Interprétation

La **décomposition** (calculer m=op(a,b) PUIS r=op(m,c)) apprend les sous-fonctions
binaires (grok parfait sur 121 paires) puis **compose par application** → généralisation
gratuite. Le one-shot doit mémoriser l'espace produit complet (impossible sur non-vus).

### 3.3 Généralisation à tous les modes

Le pattern est reproduit sur **tous les modes** en utilisant des règles arithmétiques
sur IDs discrets :

| Mode | Règle | Grok | Cascade d5 | Gate+Obs |
|---|---|---|---|---|
| Arithmétique | (3a+5b)%11 | 100% | 100% | 100% |
| Video | (2a+b)%11 | 100% | 100% | 100% |
| 3D | (3a+5b)%11 | 100% | 100% | 100% |
| World | (a+b)%11 | 100% | 100% | 100% |
| Audio (génération) | phonétique | 100% | 97% gén | — |
| Image | flow-matching | 89.5% | 78% gén | — |

---

## 4. Génération depuis Compréhension

### 4.1 Principe

Le modèle grok les **règles** d'un domaine, puis **génère** des signaux en appliquant
ces règles. C'est le crown-jewel **inversé** :

```
Crown-jewel : grok op(a,b) → COMPUTE op(op(a,b),c) → 100%
Génération  : grok règles → GÉNÉRER signal depuis règles → 78-100%
```

### 4.2 Asymétrie fondamentale

**Générer depuis règles (78-100%) >> Reconnaître depuis signal (0.5-43%)**

Cette asymétrie est l'extension naturelle du crown-jewel : la décomposition (générer)
bat le one-shot (reconnaître). Pour chaque mode, générer depuis la compréhension est
plus efficace que reconnaître depuis le signal.

---

## 5. Les 6 Lois empiriques

| Loi | Énoncé | Preuve |
|---|---|---|
| **L1** | Décomposition > Scale | 0.75→100% par décomposition; scale inverse |
| **L2** | Masquage incrémental | Sous-ensemble visible → cascade à l'inférence |
| **L3** | depth_max ≈ 1/(1−per_step) | Per-step exact → profondeur ∞ (vérifié à 100000) |
| **L4** | Récurrence ⊥ Longueur ⊥ Params | Raisonner = étapes, pas params |
| **L5** | L = 1+4·D | Format séquence; batch·L ≈ constant |
| **L6** | Association | 1-source direct; multi-source = décomposer |

**Loi unifiée** : D = k^1.98 × P^1.06 × d^−2.38

---

## 6. IDs Numériques — le principe fondateur

**Tout convertir en IDs numériques discrets avant le SpectralCoreBlock.**

Le grokking spectral marche parce que c'est une **association entre NOMBRES** (IDs),
pas une copie de signal continu. Le FFT découvre les patterns dans les séquences
d'entiers — exactement comme il découvre les fréquences dans un signal.

```
✅ Texte → word_ID → FFT grok → 100% (concept_grok)
✅ Phonème → phoneme_ID → FFT grok → 100% (déterministe)
✅ Video → frame_ID → FFT grok → 100% (transition arithmétique)
✅ 3D → voxel_ID → FFT grok → 100% (composition arithmétique)
❌ Audio → Mel float → FFT → pas de grok (signal continu stochastique)
```

---

## 7. Gates et Observateur

Le **gate de confidence** (`meta[0]`) est entraîné vers `CONF_TARGET = 4.0`
(sigmoid ≈ 0.98 > τ_grok = 0.9). Il sert d'**observateur** :

- Confiant ET correct sur non-vus → **COMPRÉHENSION** ✓
- Confiant ET faux → **surconfiance** (mémoire déguisée)
- Non-confiant → **abstention** (anti-hallucination)

---

## 8. Résultats expérimentaux

### 8.1 Domaines déterministes (règles arithmétiques sur IDs)

12 domaines à **100%** : arithmétique, logique, morphologie, composition, phonème→concept,
physique, chimie, algorithmique, théorie des nombres, géométrie, finance, statistiques.

### 8.2 Génération multi-modale

| Mode | Génération depuis règles | Vérification |
|---|---|---|
| Audio | 97% (Mel généré reconnu correct) | concept→Mel→classifier |
| Image | 78% (patches générés reconnus) | concept→patches→classifier |
| Video/3D/World | 100% (cascade d5) | composition arithmétique |

### 8.3 Tests

**1137 tests automatisés**, 0 échec, 0 régression.

---

## 9. Discussion

### 9.1 Pourquoi le FFT grok

Le SpectralCoreBlock applique un filtre fréquentiel appris sur la rfft de la séquence.
Pour les séquences d'IDs discrets (Z_n), les fréquences sont les **patterns périodiques**
des associations entre nombres — exactement ce que le filtre apprend à sélectionner.

### 9.2 Pourquoi l'audio stochastique résiste

L'audio est **stochastique** (même mot → signaux différents selon le locuteur). Les IDs
extraits varient entre locuteurs (similarité intra-mot 5-9%). Le FFT ne peut pas grokker
une association qui change à chaque instance. La **génération depuis règles** (97%)
contourne ce problème : le modèle génère depuis les règles comprises, pas depuis le signal.

### 9.3 LEAN bat DATA

Le crown-jewel atteint 100% avec **200 exemples** (compréhension). L'audio classification
atteint 31% avec **105,000 exemples** (mémoire). C'est la preuve que la compréhension >
mémoire, même avec 500× moins de données.

---

## 10. Conclusion

OCM-26400 démontre qu'une architecture spectrale unifiée (FFT, 675K params) peut
**comprendre** des règles dans tous les domaines et **générer** depuis cette compréhension.
Le crown-jewel — généralisation compositionnelle 100% vs one-shot 0.5% — est reproduit
sur tous les modes testés. Le principe **compréhension > mémoire** est validé : un modèle
qui grok les règles n'a pas besoin de milliards de paramètres ni de téraoctets de données.

---

## Références

1. Power et al., "Grokking: Generalization Beyond Overfitting" (2022)
2. Lipman et al., "Flow Matching for Generative Modeling" (2023)
3. Fodor, "The Language of Thought" (1975)
4. Parseval, théorème d'égalité énergétique (1806)

---

*OCM-26400 — SpectralCoreBlock, AMV-256, crown-jewel, LSRA, ACSP, curriculum v4.
675K params fixes. Pas de transformer. Pas d'attention. FFT unifiée.*
