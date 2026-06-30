# TESTING.md — Journal d'expériences scientifiques SPXLM

**Projet**: SPXLM (SpectraLM / Positional Reasoning / Arithmetic Generalization)
**GPU**: AMD RX 7900 XTX 24GB (ROCm)
**Période**: Session v3 → v7
**Format**: `ID | Date | Hypothèse | Méthode | Résultat | Conséquence/Insight`

---

## ANNEXE 1: Formules complementaires et decouvertes additionnelles

### A1.1 Loi empirique alternative (dataset dur)

Sur un dataset plus difficile (3000 compositions, 5 regles), la formule mesuree est:

```
D = k^1.98 * P^1.06 * d^-2.38
```

| Param | Mesure | Theorie (akone) | Interpretation |
|-------|--------|-----------------|----------------|
| beta  | 1.98   | 3.5             | Effet k sous-estime (quadratique) sur dataset dur |
| gamma | 1.06   | ?               | NOUVELLE MESURE: D proportionnel aux params (D ~ P) |
| delta | -2.38  | -3.55           | NEGATIF: d augmente -> D diminue (inverse du scale) |

Interpretations:
1. gamma = 1.06 ~ 1: D est proportionnel aux params. Plus de params = plus de profondeur fiable.
2. delta = -2.38 < 0: INVERSE DU SCALE! Augmenter d_model diminue D. Conforme a la these centrale.
3. beta = 1.98: effet k sous-estime sur dataset dur. Le scratchpad est moins efficace quand la tache est plus complexe.

Sweet spot sur dataset dur:
  d=64, P=67K: A=86.7%, D=69931
  d=128, P=237K: A=80.0%, D=44864
  d=256, P=883K: A=73.3%, D=32292
  d=512, P=3.4M: A=73.3%, D=32292 (saturation)

### A1.2 Interference de gradient vs DOSC (preuve majeure)

Decouverte critique: le DOSC resout l'interference de gradient.

  Phase 2 (mask ans seulement, m1 visible):
    - 2000 steps suffisent pour grokker ans a 0.964
  vs joint training (nl_hybrid):
    - ans max 0.652 puis decline vers 0.214
  vs nl_2step:
    - ans=0.129 meme avec m1 visible

CONCLUSION: L'interference de gradient etait BIEN la cause principale
de l'echec du joint training. Le curriculum sequentiel (DOSC) resout le probleme.

### A1.3 Formule OCM-26400 — Amodal Mentalese Vector

Architecture complementaire: le Mentalese Vector (AMV-256).

v = [v_ent || v_prop || v_op || v_meta]  (4 x 64 dimensions)
  v_ent:  entity/root primitive (Cat, Mass)
  v_prop: properties/modifiers (Plural, Velocity)
  v_op:   causal operators (compose, derive)
  v_meta: metadata (confidence, modality source)

Amodal alignment:
  f_T(C) ~ f_A(C) ~ f_V(C) ~ f_S(C) ~ v_C
  (toutes les modalites convergent vers le meme vecteur)

Causal Rigor Loss (ACSP):
  L = alpha*L_align + beta*L_step + gamma*L_sparse + delta*L_consist

  L_align: penalise les vecteurs qui ne correspondent a aucun concept connu
  L_step:  penalise les operations illegales (P_backtrack >> 1)
  L_sparse: L1 sur les 256 dimensions (anti-memoire noise)
  L_consist: InfoNCE pour la coherence multimodale

Test-time compute (LSRA):
  v(t+1) = TransformerBlock(v(t), Context)
  Stop quand v(t)_confidence >= tau_grok
  Si T* > MaxIterations -> ANOMALIE_CAUSALE -> Sleep Mode

Cette architecture est COMPLEMENTAIRE a SPXLM:
  - SPXLM = le moteur spectral (FFT + diffusion-fill + grokking)
  - OCM-26400 = la structure du mentalese (vecteur partitionne + causal rigor)
  Les deux peuvent etre combines pour un modele cognitif complet.

| ID | Date | Hypothèse | Méthode | Résultat | Conséquence / Insight |
|----|------|-----------|---------|----------|-----------------------|
| E01 | Sess-v3 | Un Transformer large (d=768, 268M) peut apprendre le raisonnement mathématique par dump aveugle de données | Entraînement Transformer d=768 268M sur dump non structuré | **0% compréhension** — le modèle mémorise sans généraliser | Le scaling brut ne produit pas de raisonnement. La capacité est gaspillée en mémorisation. **Il faut un format structuré et un scratchpad.** |
| E02 | Sess-v4a | Un xLSTM compact (d=256, 8M) peut grokker l'addition sans scratchpad | xLSTM d=256 8M, addition, sans scratchpad | **0% généralisation** | L'addition nécessite un espace de calcul intermédiaire. Sans scratchpad, le modèle ne peut pas décomposer l'opération. |
| E03 | Sess-v4b | Un scratchpad simple (comptage +1) permet le grokking | xLSTM d=256, scratchpad comptage +1 | **0% généralisation** (comptage +1 partiel) | Le scratchpad naïf ne suffit pas — il faut une décomposition algorithmique explicite. |
| E04 | Sess-v4c | Un scratchpad indexé [N] améliore la généralisation | xLSTM d=256, scratchpad [N] (indexation positionnelle) | **20% généralisation** | L'indexation positionnelle aide le modèle à structurer le calcul. Premier signal de grokking. |
| E05 | Sess-v4d | Un scratchpad bidirectionnel [i/b] permet l'arrêt propre | xLSTM d=256, scratchpad [i/b] (init/borrow) | **60% arrêt**, **0% généralisation** (bug eval) | Le format [i/b] améliore la terminaison mais un bug d'évaluation masque la généralisation réelle. |
| E06 | Sess-v4e | Un scratchpad long (50K tokens) permet le grokking complet | xLSTM d=256, scratchpad long 50K tokens | **85% généralisation** | **Le scratchpad long est crucial.** Le modèle a besoin d'espace suffisant pour dérouler l'algorithme. Confirme O(k^3.5) sur D. |
| E07 | Sess-v4f | La copie de texte est possible en char-level AR | xLSTM char-level (4 variantes: char/word/BPE/mixed) | **0% × 4** — copie impossible | **La copie de texte est impossible en char-level sans attention.** L'AR séquentiel ne peut pas回头看. |
| E08 | Sess-v5 | Les FastWeights ajoutent suffisamment de capacité pour le raisonnement | xLSTM + FastWeights (mémoire externe différentiable) | **0%** — capacité insuffisante | Les FastWeights seuls ne comblent pas le gap. Il faut changer de paradigme (FFT bidirectionnel). |
| E09 | Sess-v6a | Le FFT bidirectionnel permet l'addition sans scratchpad | FFT bidirectionnel, addition, sans scratchpad | **0%** — next-token dégénéré | Sans scratchpad, le FFT prédit le token le plus fréquent. Le contexte bidirectionnel ne suffit pas sans structure de calcul. |
| E10 | Sess-v6b | Un scratchpad SBS (step-by-step) dans le FFT permet le grokking | FFT bidirectionnel, scratchpad SBS | **25%** — FFT trop simple | Le SpectralBlock de base est trop simple pour capturer les dépendances arithmétiques multi-étapes. |
| E11 | Sess-v6c | Le SpectralBlock amélioré augmente la généralisation | FFT + SpectralBlock amélioré + scratchpad SBS | **25%** (plateau) | L'amélioration du SpectralBlock seule ne suffit pas. Le problème est ailleurs (format, anti-raccourci). |
| E12 | Sess-v6d | Le protocole complet anti-raccourci (18/18 cas) force le calcul | FFT + SpectralBlock + anti-raccourci complet 18/18 | **0%** — anti-raccourci trop agressif | **Masquer TOUT emprend l'apprentissage.** L'anti-raccourci doit être asymétrique: masquer un sous-ensemble seulement. |
| E13 | Sess-v6e | Le format STRUCTURE (séparateur `\|`, padding) du SpectraLM permet le grokking exact | FFT bidirectionnel, format SpectraLM exact (sep `\|`, padding aligné), addition k=1 | **100% GÉNÉRALISATION** 🎉 | **PREMIER SUCCÈS COMPLET.** Le format structuré est la clé. Grok démarre step 18000, complété step 44000. Le FFT a besoin de délimiteurs syntaxiques. |
| E14 | Sess-v6f | La morphologie char-level dans le FFT transfère à la syntaxe | FFT, morphologie char-level | **0%** — padding domine | Le padding domine le signal spectral. La morphologie char-level n'est pas compatible avec le FFT. |
| E15 | Sess-v6g-v1 | La morphologie par IDs numériques avec anti-raccourci fonctionne | FFT, morphologie IDs numériques, anti-raccourci v1 | **0%** — anti-raccourci masque la racine, instable | L'anti-raccourci masque trop de signal (root), rendant l'apprentissage instable. |
| E16 | Sess-v6g-v2 | Ignorer le rule tag dans la morphologie IDs stabilise l'entraînement | FFT, morphologie IDs numériques v2 (rule tag ignoré) | **0%** — output constant offset ~260 | Le modèle ignore le rule tag et produit un offset constant. La morphologie IDs ne fonctionne pas non plus. |
| E17 | Sess-v7 | Le sommeil multi-phase consolide l'apprentissage | Architecture multi-phase avec sommeil (consolidation macro→micro) | **EN COURS** | Le sommeil doit transférer les basses fréquences (macro) vers les hautes fréquences (micro). |

---

## Données SpectraLM (rapports de référence)

| ID | Config | k (généralisation) | p_step | A (accuracy) | Notes |
|----|--------|---------------------|--------|--------------|-------|
| S-E6a | Sans scratch | k=1 | 0.750 | 0.750 | Baseline sans scratchpad |
| S-E6b | Avec scratch | k=3 | 0.995 | 0.984 | Le scratchpad multiplie la portée de généralisation par 3 |
| S-B4a-14k | 14K steps | k=3 | 0.939 | 0.827 | Apprentissage intermédiaire |
| S-B4a-30k | 30K steps | k=3 | 0.978 | 0.937 | Convergence proche du plateau |
| S-B2c | Cascade | k=3 | 0.677 | 0.311 | La cascade (masquer sous-ensemble) sous-performe |

---

## Insights théoriques du user

### 1. Scratchpad = Normalisation de position (hippocampe)
Le scratchpad n'est pas juste un espace de calcul — c'est un **mécanisme de normalisation de position**, similaire au rôle de l'hippocampe qui "place" les informations dans un repère spatial cohérent. Sans scratchpad, le modèle ne sait pas où placer chaque étape intermédiaire.

### 2. Phase 3 = Réactivation, pas réapprentissage
La phase de sommeil/consolidation ne réapprend pas depuis zéro. Elle **réactive** les patterns déjà vus en les réorganisant. C'est une replay différent de l'entraînement initial.

### 3. c_short < c_honest = Distribution OOD
Quand le coût du raccourci (c_short) est inférieur au coût honnête (c_honest), le modèle apprend une distribution **Out-of-Distribution** — il optimise un proxy au lieu de la vraie fonction. L'anti-raccourci doit rendre c_short ≥ c_honest.

### 4. Oubli catastrophique = Dilemme stabilité/plasticité
L'oubli catastrophique n'est pas un bug — c'est la manifestation fondamentale du **dilemme stabilité/plasticité**. Trop de plasticité → oubli. Trop de stabilité → pas d'apprentissage. Le sommeil vise à équilibrer les deux.

### 5. Savings (re-apprendre 3-5× plus rapide)
Même après oubli catastrophique, le modèle re-apprend **3-5× plus vite** la tâche oubliée. Ce phénomène de "savings" prouve que **une trace latente persiste** dans les poids, même si la performance de surface disparaît. Le sommeil peut exploiter ces traces.

### 6. Anti-raccourci asymétrique
L'anti-raccourci ne doit pas masquer toutes les variables de sortie — seulement celles qui sont **algébriquement récupérables** à partir des autres (anti-raccourci symétrique → casse l'apprentissage). L'anti-raccourci **asymétrique** masque un sous-ensemble stratégique qui force le calcul sans détruire le signal.

### 7. DOSC appliqué à la syntaxe
Le principe DOSC (Delayed Output Structured Constraint) peut être appliqué à la syntaxe: au lieu de forcer la sortie finale, on contraint la **structure du chemin** (séparateurs, padding, ordre des tokens). Le format SpectraLM (E13) est une instance de DOSC syntaxique.

### 8. Grokking séquentiel (curriculum de difficulté)
Le grokking peut être piloté par un **curriculum séquentiel de contraintes**:
```
c = 1.0  (couverture complète, apprentissage facile)
  → c = 0.02  (couverture minimale, force la généralisation)
    → d = 0.999  (decay élevé, force la rétention long-terme)
```
Chaque phase prépare la suivante. Le passage direct à c=0.02 échoue; la séquence réussit.

---

## Synthèse des lois empiriques découvertes

| Loi | Énoncé | Évidence |
|-----|--------|----------|
| **Scratchpad super-linéaire** | Le scratchpad agit en O(k^3.5) sur D (distance de généralisation) | E02→E06: passage de 0% à 85% avec scratchpad long |
| **d_model non-monotone** | Augmenter d_model au-delà d'un optimum **casse** le grokking | E01 (d=768: 0%) vs E06 (d=256: 85%) |
| **Curriculum > joint training** | L'entraînement séquentiel par phase bat le joint training | Insight 8, données SpectraLM B2c |
| **Masquer sous-ensemble > masquer tout** | La cascade (sous-ensemble) > calcul aveugle (tout) | E12 (18/18: 0%) vs E13 (format structuré: 100%) |
| **Sommeil = consolidation fréquentielle** | Le sommeil transfère macro→micro (basses→hautes fréquences) | E17 (en cours), Insight 1-2 |
| **FFT a besoin de structure syntaxique** | Le FFT exige séparateurs `\|` et padding aligné | E09 (0%) → E13 (100%) |
| **Copie impossible en AR char-level** | Sans attention, l'AR ne peut pas copier du texte | E07 (0% × 4) |

---

## Architecture matérielle

- **GPU**: AMD RX 7900 XTX 24GB
- **Backend**: ROCm
- **Contrainte mémoire**: 24GB VRAM limite d_model et batch_size
- **Optimisation**: d=256 optimal pour xLSTM; FFT permet d_model plus large

---

## Prochaines étapes

1. **Terminer E17** (sommeil multi-phase) — valider la consolidation macro→micro
2. **Tester le grokking séquentiel** (c=1.0 → c=0.02 → d=0.999) sur addition k>1
3. **Explorer l'anti-raccourci asymétrique** sur morphologie (E15/E16 ont échoué avec symétrique)
4. **DOSC syntaxique** appliqué à des tâches plus complexes (multiplication, algèbre)

---

*Dernière mise à jour: Session v3-v7*
