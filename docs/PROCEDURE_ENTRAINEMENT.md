# PROCÉDURE D'ENTRAÎNEMENT OCM-26400 — officielle

> Protocole canonique, aligné sur `CANON_OCM26400.md` (lois L1-L10, scaling, sommeil, grok).
> Construit sur du **prouvé** (rapports FFT/SCB + session 2026-07), pas de réinvention.
> **Règle primordiale** : le modèle **comprend** les règles/formules/lois/logique → **généralise depuis sa compréhension**. Différent des transformers (pas de scale, pas de joint, curriculum gated par la compréhension).

---

## 0. PRINCIPES TRANSVERSES (à respecter à chaque phase)

- **AMV-256 canonique FIGÉ** `[ent(64)|prop(64)|op(64)|meta(64)]`, `meta[0]`=gate (L_align), `meta[1]`=confiance source, `meta[2]`=cohérence cross-modal. Dictionnaire `ent` orthogonal partagé = **interface standard B'** (lobes remplaçables).
- **Split loss** (canon §) : `CE` pour **lobes/perception** (stochastique) · `1-cos` pour **cœur/raisonnement** (grok) · `flow-matching` pour **génération continue** (audio/image/vidéo) · `CE vocab` pour **texte/chat**. **Jamais** 1-cos sur perception/génération, jamais CE sur continu.
- **Tout numérique** : signaux → VQ/codebook → IDs → cœur FFT (les fréquences = les nombres).
- **Gates = stop sur compréhension, pas sur epochs** : L1≥0.99 (primitives), L2≥0.95 (composition), L5≥0.90 (scratchpad), L6≥0.85 (association). `meta[0]=L_align`.
- **Sommeil OBLIGATOIRE** : 3 phases spectrales (léger low-pass kf=0.5 / moyen entropie / profond high-pass kf=0.3) + replay. Boucle autonome : sommeil tant que gate<τ (validé 3-4 cycles, +73pt vs pur, gate calibrée).
- **DOSC séquentiel** (L7/L8) : 1 champ/phase, anti-raccourci **symétrique**, interleaved final. Joint = interférence (0.13 vs 0.999 séquentiel).
- **Rigueur** : val-split, select-on-val, test-once, ≥3 seeds, sanity check P1 (loss→0 en 100 steps), pas de sélection-sur-test.

---

## ARCHITECTURE (B', rappel)
```
[MODALITÉ] → [LOBE spectral mince SÉPARÉ, CE] → [AMV-256 canonique figé] → [GATE L_align+snap] → [CŒUR SCB unique, 1-cos] → [LOBE INVERSE spectral]
  audio STFT / image DCT / vidéo FFT-3D / 3D SH / texte tok / world Laplacien                   (raisonnement, capture L6)        (génération, Phase 4)
```
- Lobes séparés, remplaçables sans retrain du cœur (validé B' : synth 100%, réel image 100%).
- Capture simultanée : superposition 1 passe des AMV → associations L6 (validé 91%, L10-safe).

---

## PHASE 0 — APPRENDRE : moteur (crown-jewel) + perception (lobes)

### 0a. Cœur — arithmétique crown-jewel (racine, sans dépendance)
- **Tâche** : `op(a,b)=(a+b) mod P` et `(3a+5b) mod P`, P∈{3..27}. Association numérique (FFT = Fourier = les nombres).
- **Loss** : **1-cos** sur le slot `ent` vs canonique. Adam 3e-3, seed 0, ~2k-6k steps SOLO.
- **Sommeil autonome** : tant que gate(L_align) < 0.99 → sommeil 3 phases + replay.
- **Gate L1≥0.99** = primitive comprise (certifiée). Gap décomposition−oneshot ≥ +95pt.
- **Composition test** : cascade gate-certifiée profondeur 50 → 100% (valide le moteur gate→composition avant tout investissement).
- *Canon* : grok sur règle (§8), L3 profondeur, L4 récurrence. *Validé session* : `grok_gate_composition.py` (100% D=50).

### 0b. // Lobes sensoriels (CE, indépendants du cœur, en parallèle GPU)
- Chaque lobe spectral mince entraîné **CE** sur sa perception → émet `canon[classe_prédite]` (AMV dérivé du CE, **pas 1-cos direct**).
  - Audio M5+SCB (AdamW cosine 100k, speed-perturb) → 93.9%. Image DCT/patch. Texte tok. Vidéo FFT-3D. 3D SH. World Laplacien.
- **Indépendants du cœur** (CE perception ≠ 1-cos raisonnement) → aucune interférence avec 0a.
- *Canon* : frontière règle/perception (§8), CE sur lobes. *Validé session* : B' remplaçable.

---

## PHASE 1 — COMPRENDRE : primitives linguistiques noyau (ID→ID)

- **Primitives** (1-source, déterministes, **structurellement = arithmétique**) : grok SOLO en 1-cos, 2k-6k steps chacune, **DOSC séquentiel** (L7, 1 primitive/phase).
  1. word_id → plural_id
  2. word_id → tense_id (3e pers.)
  3. word_id → past_id (VBD)
  4. word_id → phoneme_id (pattern CVC)
  5. word_id → syllable_id (compte)
  6. word_id → category_id (sémantique discrète)
  7. word_id → synonym_id
  8. word_id → antonym_id
  9. word_id → syntax_role_id
- **Anti-raccourci symétrique** (L8) : masquer TOUTES les variables récupérables depuis la cible, jamais les 18/18.
- **Gate L1≥0.99/primitive, L6≥0.85** (association 1-source).
- **Sommeil** entre primitives (consolide, comble oubli 89.8%).
- *Canon* : L6 (1-source direct), L7 DOSC, L8 anti-raccourci. ⚠️ **PAS les 44 primitives** (les autres = perception CE ou composition Phase 2).

---

## PHASE 2 — RAISONNER : composition linguistique (intra-modale texte)

- **Tâches multi-source** (décomposer, L6) : morphologie (stem+affixe→mot), conjugaison composée (stem+tense→fléchi), chaînes syntaxiques courtes.
- **Décomposition** (L1) : cascade scratchpad — intermédiaire **PUIS** final. Scratchpad propre (1×, tous masqués, cascade).
- **Masquage incrémental** (L2) : sous-ensemble visible à l'entraînement → cascade se résout à l'inférence.
- **L10 (CRITIQUE)** : extraction k≥3 champs ⇒ masques phase i couvrent **tous les champs futurs j>i** (mixer FFT bidirectionnel, sinon collapse).
- **Gate L2≥0.95** (composition), L5≥0.90 (scratchpad), cascade ≥ 0.95 (L8). **Métrique = cascade** (les individuelles mentent).
- *Canon* : L1, L2, L3 (D=1/(1−p_step)), L8, L10. ⚠️ **Cascade v3 seule** (pas v1/v2 faux).

---

## PHASE 3 — ASSOCIER : capture simultanée inter-modale (L6)

- **Capture 1 passe** : superposition des AMV (texte↔audio↔image IDs) → cœur SCB → associations.
- **Complétion cross-modale** : capture partielle (2/3 modalités) → cœur retrouve le concept (association comble l'absent).
- **Gate L6≥0.85, gate≥τ_grok=0.9/étape**.
- *Canon* : L6 (multi-source=décomposer). *Validé session* : `test_capture_simultanee.py` (91%, +31pt, L10-safe).

---

## PHASE 4 — GÉNÉRER + DOMAINES

### 4a. Domaines (compositions longues sur vraies données)
- D1=Maths (GSM8K NL→CoT→arith, AIME), D2=Code (SWE-bench, Terminal), D3=Science (GPQA-Diamond, HLE).
- Compositions longues des primitives Phase 0-3. Profondeur = cascade scratchpad (L4, raisonner=étapes), pas scale.
- *Canon* : L4, comprehension-first. ⚠️ **Valeurs à re-mesurer sur audio** (lois archi transférables, valeurs texte-spécifiques).

### 4b. Générer (lobes inverses spectraux, omni-out)
- **Lobes inverses** (miroirs des lobes avant, entraînés **SÉPARÉMENT**, cœur figé) :
  - Audio → iSTFT (miroir AudioEncoder, phase apprise). Image → iDCT-2D/fold. Vidéo → iFFT-3D. 3D → SH inverse. World → Laplacien inverse. Texte/chat → tokens (SpectralCoreBlock causal, pas de transformer).
- **Loss** : flow-matching (continu : audio/image/vidéo/3D/world, Lipman 2023) · CE vocab (texte/chat).
- **Chat = InverseTextLobe** (une tête omni-out, SCB causal, CE).
- **Remplaçabilité inverse** : un lobe inverse décode l'AMV émis par le cœur nourri par un AUTRE lobe avant (test B' en sens inverse).
- *Canon* : B' séparé, loss split. *Réf* : flow-matching (generators.py), HiFi-GAN, CoCa/Show-O, SpectroStream inverse, Live Music streaming.

---

## ⚠️ ANTI-RÉGRESSION (cautions canon)

1. **Versions validées seulement** : cascade **v3** (pas v1/v2), **E6/E12** (pas E8/B2 gonflés fuite format), **k=4 cascade non validée** (bug L10).
2. **Valeurs à re-mesurer sur audio** : lois archi (L1-L10, γ≈0) transférables ; valeurs (cascade 0.97, β=3.5) mesurées sur texte/arith → re-mesurer.
3. **L10 obligatoire** pour tout DOSC/capture multi-champ L>1 (mixer bidirectionnel).
4. **Scaling** : `D=k^3.5·n_blk^(2/3)·T^1.34`, **γ≈0 (largeur PAS un levier)** → ne pas scale la largeur (élargir dégrade, L1).
5. **Liste noire** (interdit) : xLSTM/sLSTM, Hopfield comme mixer (réfuté), ContentAddress, FHRR, Fast-Weights socle, Quaternion FFT, SWM/neural-PDE, MoE, Var-JEPA, EWC/radix/ternary.

---

## VALIDATION DA+JUGES (checklist par phase)

- [ ] **Phase 0a** : crown-jewel gate L1≥0.99, composition D=50 à 100%, gap decomp−oneshot ≥+95pt
- [ ] **Phase 0b** : chaque lobe CE atteint son acc (audio 93.9%), émet canon[classe] (AMV standard)
- [ ] **Phase 1** : 9 primitives ID→ID gate L1≥0.99 SOLO (DOSC, anti-raccourci symétrique)
- [ ] **Phase 2** : cascade v3 ≥0.95, L10 masques futurs (k≥3), scratchpad propre
- [ ] **Phase 3** : capture L6 ≥0.85, complétion cross-modale, L10-safe (invariant ordre)
- [ ] **Phase 4a** : domaines (observateur correct+confiant sur non-vus)
- [ ] **Phase 4b** : lobes inverses séparés, remplaçabilité inverse, loss split (flow/CE)
- [ ] **Transverse** : sommeil autonome entre phases, gates stop-criterion, ≥3 seeds, select-on-val, pas de sélection-sur-test

---

*Références : `CANON_OCM26400.md` (lois/liste noire/cautions), `ARCHITECTURE_UNIFIEE.md` (B'), `SOLUTION_OCM26400.md` (théorie), `REPRODUCIBILITE_OCM26400.md` (hyperparams), `rapports/` (fondation FFT/SCB).*
*Document vivant. Dernière MAJ : 1 juil 2026, aligné sur le canon après audit DA+juges.*
