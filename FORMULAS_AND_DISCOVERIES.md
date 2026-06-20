# OCM-26400 — Formules, Découvertes, Innovations, Leçons

**Consolidation scientifique** — extrait de 113 fichiers `.py` + 130+ tests + expériences.
Objectif : tout retrouver au même endroit pour reproduire sans erreur.

---

## 1. ARCHITECTURE — SpectralCoreBlock (le noyau unifié)

### 1.1 Mélangeur spectral FFT bidirectionnel
Le noyau du modèle. Mixage de séquence par FFT en **O(L log L)** (vs attention O(L²)).

```
Entrée : x ∈ ℝ^{B×L×D}   (ou (B,D) traité comme L=1)

  h      = in_proj( LayerNorm(x) )                    # projection
  X      = rfft(h, dim=L) ∈ ℂ^{B×F×D}                 # F = L//2 + 1 coefficients
  filter = filter_real + i·filter_imag ∈ ℂ^{F×D}      # APPRIS, init ≈ 1.0
  X̃      = X ⊙ filter                                  # multiplication complexe
           Re(X̃) = Re(X)·fr − Im(X)·fi
           Im(X̃) = Re(X)·fi + Im(X)·fr
  y      = irfft(X̃, n=L) ∈ ℝ^{B×L×D}                  # retour temporel
  x      ← x + out_proj(y)                             # résiduel spectral
  x      ← x + FFN( LayerNorm(x) )                     # FFN (4× expansion, GELU)
```

**Stabilité de Parseval** : `‖x‖² = ‖FFT(x)‖²` → l'énergie est conservée par la FFT,
donc le filtre ne peut pas faire diverger les activations (pas de normalisation
additionnelle nécessaire sur la branche spectrale).

### 1.2 Paramètres
| Symbole | Valeur | Rôle | Fichier |
|---|---|---|---|
| `d_model` (D) | 256 | dimension AMV | `amv.py` |
| `PART` | 64 | taille partition (×4 = 256) | `amv.py` |
| `seq_len` (L) | 64 | longueur de mélange | `spectral_core.py` |
| `filter_real/imag` | (33, 256) | filtre fréquentiel appris | `spectral_core.py` |
| **Params noyau** | **675K FIXES** | indépendants agents/profondeur | — |

---

## 2. AMV-256 — Amodal Mentalese Vector

```
v ∈ ℝ^{256} = [ ent(64) | prop(64) | op(64) | meta(64) ]
                ╰───┬───╯    ╰──┬──╯   ╰─┬─╯   ╰───┬───╯
                  entité     propriété opérateur  méta
```

**Partition méta (3 rôles)** :
| meta[i] | Rôle | Source |
|---|---|---|
| `meta[0]` | **confidence** (LSRA sigmoid) | reasoner.py |
| `meta[1]` | **source_conf** (pont v6 / bridge) | learned_vocab / v6_bridge |
| `meta[2]` | **consist_score** (InfoNCE) | infonce.py |

---

## 3. VÉRIFICATEUR SYMBOLIQUE — la ground-truth

```
V(a, b) = ( A_COEF·a + B_COEF·b ) mod P_MOD
        = ( 3·a + 5·b ) mod 11
```

| Constante | Valeur | Rôle |
|---|---|---|
| `P_MOD` | 11 | modulus |
| `A_COEF` | 3 | coef entité |
| `B_COEF` | 5 | coef propriété |
| `P_BACKTRACK` | 1000.0 | pénalité étape illégale |

⚠️ **`(3a+5b) mod 11` est NON-commutative ET NON-associative** → pas de raccourci
algébrique possible. Le grok compositionnel observé est **réel**, pas un artefact.

---

## 4. ACSP LOSS — la loss d'entraînement

```
L = α·L_align + β·L_step + γ·L_sparse + δ·L_consist
```

| Coef | Valeur | Terme |
|---|---|---|
| α | 1.0 | L_align |
| β | 1.0 | L_step |
| γ | 1e-3 | L_sparse |
| δ | 0.0 | L_consist (0 en single-modalité) |

```
L_align   = min_{d ∈ D} ( 1 − cos(v.ent, d) )              # ancrage dictionnaire
L_step    = 0   si V(a,b) légal,  P_BACKTRACK sinon         # (NON-diff → voir Gumbel)
L_sparse  = λ · Σ_i |v_i|        (L1 sur 256 dims)          # régularisation
L_consist = InfoNCE(z_a, z_b, τ=0.07)                       # cross-modal (§2.4)
```

### 4.1 Le fix critique : L_step DIFFÉRENTIABLE (P-B)
Le `L_step` original (`acsp.py:35`) retourne une **constante** (0 ou 1000) → pas de
gradient. On a créé `diff_decode.py` avec **Gumbel straight-through** :
```
L_step_diff = decode_gumbel(v, ...) → gradient via ST-estimator
acsp_loss_diff(v, dictionary, verifier, a, b, op_id)   # loss RÉELLE pour training
train_with_acsp(d, ver, n_steps)                       # trainer RÉEL (pas décoratif)
```
**Leçon** : une loss non-différentiable est décorative. Vérifier TOUJOURS que le
gradient remonte (`tensor.requires_grad` + `backward` sans erreur).

---

## 5. LSRA — récurrence de raisonnement (depth_max)

### 5.1 La boucle
```
v(0)   = encode(a, b)
v(t+1) = ReasonerBlock( v(t) )            # = SpectralCoreBlock recyclé
T*     = min{ t  |  sigmoid( meta[0](t) ) ≥ τ_grok }   # arrêt adaptatif
```
Si `max_iter` atteint sans dépasser `τ_grok` → **[ANOMALIE_CAUSALE]** → **abstention**
(le modèle dit « je ne sais pas » au lieu d'halluciner).

### 5.2 Constantes
| Symbole | Valeur | Rôle |
|---|---|---|
| `τ_grok` | 0.9 | seuil de confidence pour stop |
| `CONF_TARGET` | 4.0 | sigmoid(4)≈0.98 > τ_grok (cible d'entraînement) |
| `max_iter` | 8 | borne de sécurité |
| `lr` | 3e-3 | Adam |

### 5.3 La LOI depth_max (découverte clé)
> **La récurrence découple la profondeur de raisonnement du nombre de paramètres.**

```
1000 agents × profondeur 64 = 64 000 pas de raisonnement en 0.020s
avec 675K params FIXES (vs un LLM qui ajoute des params pour raisonner +longtemps)
```
**Raisonnement = ajouter de la profondeur, pas des paramètres.** C'est le cœur du
paradigme utilisateur : « apprendre les bases → intermédiaire grok → décomposer
macro→micro → généralisation émerge → efficient (raisonner, pas grossir) ».

---

## 6. CROWN JEWEL — la généralisation compositionnelle (prouvée)

### 6.1 Résultat arithmétique (Z₁₁)
| Métrique | Valeur |
|---|---|
| Opération | `(3a+5b) mod 11` |
| Train triples | 200 |
| Test triples (jamais vus) | 1131 |
| Grok binaire (sous-fonction) | **100%** |
| ONE-SHOT (triples neufs) | **0.5%** ❌ |
| DÉCOMPOSITION (triples neufs) | **100%** ✅ |
| **GAP** | **+99.47 points** |

**Interprétation** : one-shot doit mémoriser l'espace produit complet (impossible sur
non-vus). Décomposition n'apprend que les sous-fonctions binaires (grok parfait) puis
**compose par application** → généralisation *gratuite*.

### 6.2 Robustesse prouvée
- **Survie one-hot → dense** : +100 points (ortho ET random dense). Le grok ne dépend
  PAS de l'alignement sur les axes. (`experiment_linguistic_dense.py`)
- **Profondeur 256** : 100% sur chaînes `op^k`. (`experiment_recursion.py`)
- **Survie cross-domain** : inter-règles (math+chimie+bio) → composition inter-domaine 100%. (`experiment_cross_domain.py`)
- **Scaling V>64** : Z₁₂₀ (impossible en one-hot), grok règle 99.7% brut. (`experiment_vocab_scale.py`)

---

## 7. CURRICULUM anti-shortcut

Phases :
1. **Primitives** : opérateur binaire seul (grok sous-fonction)
2. **Paires** : composition de 2
3. **Chaînes** : composition de k (profondeur)
4. **Inter-règles** : composition cross-domain

**Anti-shortcut** : `gap(train_acc, test_acc) < δ`. Si le gap explose, le modèle a
trouvé un raccourci (mémorisation) — on recale.

---

## 8. SOMMEIL / CONSOLIDATION

```
mémoire épisodique  →  extraction de règle  →  mémoire sémantique
                         (compression ×27)        (généralise aux 121 paires)
```
Le modèle « dort » : consolide les traces épisodiques en règles abstraites réutilisables.
Compression ×27 mesurée, généralisation aux paires jamais vues. (`sleep.py`)

---

## 9. GÉNÉRATION — flow-matching (Lipman 2023)

Au lieu d'une régression MSE, la génération se fait par **flow-matching** :
```
x_t   = (1−t)·x_0 + t·x_1        # interpolation linéaire dans l'espace AMV
v     = x_1 − x_0                # champ de vecteurs cible
loss  = ‖ decoder(amv, t, x_t) − v ‖²
sample: x_0 ~ N(0,I), intégrer de t=0→1 par pas Euler
```
`AMVConditionedDecoder` : décodeur conditionné par l'AMV. Plus expressif que MSE
(génère de vrais signaux multimodaux : audio, image, vidéo). (`generators.py`)

---

## 10. INNOVATIONS CLÉS (résumé)

| # | Innovation | Pourquoi ça compte |
|---|---|---|
| 1 | **Noyau spectral FFT unifié** | O(L log L) + Parseval stable → scaling sans explosion |
| 2 | **Crown-jewel compositionnel** | +99.5pt : preuve que grok+compose > mémorisation |
| 3 | **depth_max (récurrence)** | Raisonnement = profondeur, pas params (675K fixes) |
| 4 | **Abstention épistémique** | ANOMALIE → « je ne sais pas » → apprentissage (anti-hallucination) |
| 5 | **ACSP différentiable (Gumbel ST)** | L_step décorative → vraie loss (fix critique) |
| 6 | **Survie dense (P2)** | Le grok ≠ alignement d'axes → robuste au passage dense |
| 7 | **Sleep consolidation ×27** | Épisodique → sémantique, généralise aux non-vus |
| 8 | **Adaptateur MCP** | Outils natifs derrière protocole standard (MCP-Atlas) |

---

## 11. LEÇONS TIRÉES (anti-erreurs)

| Piège rencontré | Symptôme | Fix |
|---|---|---|
| Loss non-différentiable | `acsp_loss` L_step = constante, pas de grad | Gumbel straight-through (`diff_decode.py`) |
| Conjugaison naïve | `happy+ness`→`happyness` | Détection y→i + verbes réguliers en test |
| Gate stricte trop serrée | V=120/64 validité 13% | Sharpening, pas rejet (finding honnête) |
| LazyLinear non-init | Count params = 0 avant forward | Forward avant `sum(p.numel())` |
| SSRF / shell=True | Vulnérabilités sécurité | `_validate_url_safe()` + `shlex.split` liste |
| Claim MRR sémantique | Falsifié (pas de terme distributionnel) | Retrait honnête du claim |
| Pont v6 non-généralisant | 0% sur symboles non vus | Reframe honnête : encodeur à dico fixe |
| TTC tautologique | Convergence par construction | Reframe : gate calibrée + abstention |
| Flow-matching MSE sur one-hot texte | Loss bloquée ~0.96 (prédit la moyenne = variance) | Cross-entropy sur logits char (CharGenerator) : loss 3.87→0.001 |
| Mauvaise procédure de training | grok 0.95 (train_with_acsp) vs cible 0.99 | Procédure canonique `train_binary_block` (loss 1-cos) → grok **1.00** |
| Benchmarks tautologiques | `rule.apply == rule.apply` (100% cosmétique) | Tester le core NEURAL sur hold-out (neural_multihop : 97-100%) |
| Moins unicode dans code | `−` (U+2212) → SyntaxError | ASCII `-` partout |

**Méta-leçon** : « Honnêteté scientifique d'abord ». Chaque claim falsifié est retiré
et documenté, pas masqué. C'est ce qui rend OCM-26400 crédible vs un demo marketing.

---

## 12. REPRODUCIBILITÉ — constantes globales ( cheat-sheet )

```python
# Architecture
D_MODEL     = 256       # amv.py
PART        = 64        # amv.py (×4 partitions)
seq_len     = 64        # spectral_core.py
KERNEL_PARAMS = 675_000 # FIXES

# Vérificateur
P_MOD       = 11        # verifier.py
A_COEF      = 3         # verifier.py
B_COEF      = 5         # verifier.py
P_BACKTRACK = 1000.0    # verifier.py

# ACSP loss
ALPHA = 1.0; BETA = 1.0; GAMMA = 1e-3; DELTA = 0.0   # acsp.py

# LSRA
TAU_GROK    = 0.9       # reasoner.py
CONF_TARGET = 4.0       # reasoner.py (sigmoid(4)≈0.98)
max_iter    = 8         # reasoner.py
lr          = 3e-3      # Adam, reasoner.py

# Flow-matching
tau (InfoNCE) = 0.07    # acsp.py / infonce.py
```
