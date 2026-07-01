# RÈGLES FONDAMENTALES OCM-26400 — NE JAMAIS DÉRIVER

> **Document de référence absolu.** LIRE avant chaque décision technique.
> Si une idée viole une règle → L'ÉCARTER. Incertain → consulter les experts.

---

## 1. 100% SPECTRAL — JAMAIS DE TRANSFORMER/LLM
FFT/DCT/STFT/SH/Laplacien/wavelet. PAS d'attention O(L²), PAS de transformer, PAS de LLM externe. **TOUS les composants** spectraux (lobes, cœur, décodeur, parsing NL).

## 2. COMPRENDRE → RESTITUER DEPUIS LA COMPRÉHENSION + LES RÈGLES
Le modèle **grok les règles** (grammaire, maths, logique) via 1-cos+sommeil+gate. Il **comprend** en APPLIQUANT les règles grokkées. Il **restitue** depuis cette compréhension. **PAS de parsing mécanique** (regex). Le NL est compris via les règles grokkées (comme la morphologie +s), pas extrait mécaniquement.

## 3. APPRENDRE→COMPRENDRE→RAISONNER→ASSOCIER→GÉNÉRER
Phases séquentielles, gated par compréhension (meta[0]≥τ), pas par epochs.

## 4. LOIS L1-L10 (canon figé)
L1 décomposition>scale · L2 masquage par champ · L3 D=1/(1−p_step) · L4 récurrence⊥L⊥P · L5 L=1+4D · L6 association (1-source direct, multi-source=décomposer) · L7 DOSC séquentiel · L8 anti-raccourci symétrique+interleaved · L9 distance-scaling · L10 masque bidirectionnel (k≥3 champs).

## 5. SPLIT LOSS
CE pour lobes/perception (stochastique) · 1-cos pour cœur/raisonnement (grok déterministe) · flow-matching pour génération continue · CE vocab pour texte. **JAMAIS inverser.**

## 6. ARCHITECTURE B' (CANONIQUE)
Lobes spectraux **séparés** (minces, CE) → **AMV-256 canonique figé** (ent orthogonal partagé) → **cœur SCB unique** (1-cos). Lobes **remplaçables** sans retrain du cœur. PAS de Frankenstein. PAS de joint-training. OmniModel = variante déviante (joint).

## 7. GROK COMPUTE ≠ COPY
COMPUTE (arithmétique) : masque partiel aide, wd fort aide. COPY (morphologie) : tout-masqué, wd doux (1e-3), modèle simple (1 bloc d=48). Ne PAS confondre les leviers.

## 8. CASCADE COPY ≠ COMPUTE
COPY (morphologie) : chaînage de modèles Phase-1. COMPUTE (arithmétique) : single-SCB L8 (cascade 0.97).

## 9. GATE = SIGNAL DE COMPRÉHENSION
meta[0] = alignement au dictionnaire canonique (L_align). Pilote le sommeil. Calibrée.

## 10. SOMMEIL SPECTRAL (3 phases, GATE-GUARDÉ)
Low-pass kf=0.5 + high-pass kf=0.3 + replay. Seulement si gate≥0.95 ET held<0.85 (overfit). Si underfit → re-entraîner.

## 11. AMV GÉNÉRATIF = per-position + round-trip joint
Latent per-position (pas mean), encodeur dégelé + round-trip (invertibilité apprise).

## 12. CANON — VERSIONS VALIDÉES SEULEMENT
Cascade v3 (pas v1/v2). E6/E12 (pas E8/B2). Scaling γ≈0 (largeur PAS levier). d=64 sweet-spot. Liste noire : xLSTM, Hopfield-mixer, ContentAddress, FHRR, Fast-Weights-socle, Quaternion-FFT, SWM, MoE, Var-JEPA, EWC/radix/ternary.

## 13. TOUT NUMÉRIQUE
Signaux → VQ/codebook → IDs → SCB (fréquences = nombres).

## 14. NL = COMPRENDRE PAR RÈGLES (SPECTRAL, PAS MÉCANIQUE)
Le NL est compris en **grokkant les règles** linguistico-mathématiques ("more than"→+, "remainder"→-) comme la morphologie grok +s. **PAS de regex/extraction mécanique.** Le SCB spectral a besoin de : encoding positionnel + loss CE (pas 1-cos pour perception) + format aligné. Rapports/21 a déjà obtenu 0.105 en spectral pur.

## 15. PAS DE RÉGRESSION
Ne jamais réinventer ce qui est prouvé (canon). Ne jamais contredire un résultat validé. Vérifier la cohérence avant d'agir.

## 16. TOUJOURS CONSULTER LES EXPERTS
Pour toute décision technique (architecture, training, parsing), consulter les agents experts (DA+juges).

## 17. RÈGLE PRIMORDIALE
Le modèle **comprend** les règles/formules/lois/logique pour **généraliser depuis sa compréhension**. Différent des transformers/LLM standard. Comprehension-first, toujours.

---

*Dernière MAJ : 1 juil 2026. Source : Besoins + canon + rapports + session validée.*
