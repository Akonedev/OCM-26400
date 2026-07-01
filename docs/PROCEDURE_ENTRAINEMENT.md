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

## PHASE 1 — COMPRENDRE : primitives linguistiques (caractère layout-fixe + mémoire)

> ⚠️ **CORRIGÉ après test + verdict expert** (rapports/25, `phase1_morphology_char.py`) :
> - **Morphologie RÉGULIÈRE** (plural +s, past +ed, comparative +er) = **RÈGLE → grok**, MAIS uniquement en représentation **CARACTÈRE layout-fixe + diffusion-fill bidirectionnel** (ljust, offset constant → copie=phase+édition locale, Fourier-native).
> - **IDs arbitraires word_id→form_id = IMPOSSIBLE par construction** (projection QR orthogonale = rotations aléatoires indépendantes, zéro structure Fourier partagée → held-out 0%, `phase1_linguistic_primitives.py`).
> - **Phonème-append variable = mur de représentation** (27%, V1/V2/V3 figés → non-Fourier-native).
> - **Primitives arbitraires** (synonyme, antonyme, catégorie, phoneme_id, syllable_id, syntax_role_id) = **FAITS → MÉMOIRE ADR-0016** (lookup routeable, pas de règle).

### Grok morphologique (caractère layout-fixe) — VALIDÉ 98%+ (recette v4)
- **Représentation** : caractère, `[Q: word.ljust(W_q)] + [answer TOUT masquée.ljust(W_a)]`, layout fixe.
- **Tâches** (DOSC, 1 règle/phase) : PLURAL (+s nom régulier), PAST (+ed verbe), COMPARATIVE (+er adj). Lexique **lemminflect ~400 mots réguliers/règle**.
- **Modèle** : **1 SpectralCoreBlock d=48** (bidirectionnel/diffusion-fill) — γ≈0, NE PAS scale (3 blocs d=128 = underfit). Loss **1-cos** par position sur la zone-réponse.
- **Hyperparams validés (v4)** : `wd=1e-3` (PAS 1e-2 — le COPY grok est déjà compact, wd fort l'écrase = underfit), 12000 steps, bs 64, lr 3e-3, grad-clip 1.0, ≥3 seeds, split **crc32 par lemme** (disjoint).
- **Masque** : **TOUT-masqué** (PAS partiel — le partiel crée un raccourci answer→answer sur les copy tasks ; il aide le COMPUTE/arithmétique, nuit au COPY/morphologie).
- **Sommeil GATE-GUARDÉ** : seulement si `gate_train≥0.95 ET held<0.85` (overfit/mémorisation). Si `gate<0.95` → underfit → **re-entraîner** (PAS filtrer). En pratique 0 cycle (le modèle convergé+généralisé directement).
- **Gate L1≥0.99**, métrique = exact-match mot fléchi sur 30% held-out.
- *Validé session* : `phase1_morphology_v4.py` → **PLURAL 98.2%, PAST 100%, COMPARATIVE 98.0% held-out** (3 seeds, cible 0.927 dépassée). v1=81.8% (36 mots), v2=1.3% (partiel=raccourci), v3=0% (wd=1e-2 underfit), **v4=98%+** ✓.
- *Canon* : L6, §8 (grok règle≠perception), §24 (copie=phase, Fourier-native), γ≈0 (pas de scale), ADR-0016 (double voie).

### Mémoire lexicale (ADR-0016) — pas de grok
- Primitives arbitraires (synonyme, antonyme, catégorie, syntax_role, irréguliers mouse→mice, go→went) = **lookup TSV routeable** (mémoire externe), pas de grok (pas de règle).
- Double voie Pinker : régulier en poids (grok), irrégulier/arbitraire en mémoire.

---

## PHASE 2 — RAISONNER : composition (cascade)

> ⚠️ **Découverte critique (expert + test)** : **COPY cascade (morphologie) ≠ COMPUTE cascade (arithmétique)**.
> - **COMPUTE** (arithmétique) : shortcut = réponse fausse → L8 single-SCB validé (rapports/58 v3, cascade 0.97).
> - **COPY** (morphologie) : shortcut (stem+fulness) = chemin composé ((stem+ful)+ness) **algébriquement équivalent** → aucun masquage ne peut forcer une composition distincte + garder un SCB stable (filtre FFT global per-fréquence ne supporte pas le flip STEM vis/maské). **Single-SCB COPY cascade impossible** (v1=0%, v2=1.3%).
> - **=> Composition morphologique = CHAÎNER des modèles Phase-1 à l'inférence** (v3=88%).

### Composition morphologique = chaînage de primitives Phase-1 (VALIDÉ 88%)
- **Recette** : entraîner chaque étape comme un modèle Phase-1 v4 indépendant (1 SCB d=48, char layout-fixe, tout-masqué, wd=1e-3). Chaîner à l'inférence : `stem → Modèle_A(stem→stem+ful) → m1_pred → Modèle_B(m1→m1+ness) → ans_pred`.
- *Validé session* : `phase2_composition_v3.py` → **CASCADE 88%** (M1 96%, ANS 88%, 3 seeds). Composition réelle (l'erreur se propage). Bottleneck = formes longues (Model B careful→carefulness). Pour 0.96 : +steps Model B + layout W_A=17.
- **Métrique** : cascade exact-match (chaînage), 70/30 held-out par lemme.
- *Canon* : L1 (décomposition), §8 (grok règle), §24 (copie=phase). L8 single-SCB = COMPUTE uniquement.

### Composition arithmétique/COMPUTE = single-SCB L8 (validé Phase 0)
- Pour les cascades COMPUTE (arithmétique, raisonnement multi-étapes), le single-SCB L8 (rapports/58 v3) reste valide (cascade 0.97). Phase 0 l'a prouvé (crown-jewel composition D=50 = 100%).

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
