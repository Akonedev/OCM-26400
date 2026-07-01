# CANON OCM-26400 — liste canonique anti-régression

> Référence figée des lois/mécanismes/décisions **prouvés** pour OCM-26400/SCB (lignée FFT/spectralm).
> Source : audit DA+juges des `rapports/` (juil 2026). **À respecter sans réinventer.**
> Les rapports sont NOTRE fondation (FFT/SpectralBlock = ancêtre du SCB), PAS spxlm/xLSTM (projet séparé, déplacé).

---

## ✅ CANON (applicable, prouvé sur FFT/SCB)

### Architecture
1. **SpectralBlock / FFT mixer** : `rFFT → filtre complexe appris ⊙ → modReLU → irFFT` + branche locale (conv depthwise k=7) + FFN. (rapports 02, 55)
2. **Init identité** du filtre spectral (W=1, b=0) → bloc=identité au départ (stabilité). (02:70)
3. **Parseval** : `‖x‖²=‖FFT(x)‖²` — garde-fou énergétique, pas d'explosion gradient. (01, 55)
4. **O(L log L)**, pas de cache KV (diffusion = bidirectionnel). (02, 55)
5. **Dual-paradigm** : cœur AR causal spectral (fluidité) + cœur diffusion-fill (raisonnement/scratchpad), séparés par tâche. (53, 55)
6. **Récurrence fenêtrée** : itérer le même bloc r fois → profondeur découplée de L et des params. (53)

### Grokking (fondement du grok 1-cos)
7. **Grok prouvé sur règles** (mod-97 add/mul/mixed → val 0.99+) ; **weight decay fort = clé** ; minibatch+warmup+grad-clip accélèrent. (14)
8. **Frontière règle/perception** : grok sur **règle** (circuit Fourier compact), **PAS sur perception** (mémorisation = terminal). Early-stop `save-best` pour perception ; sur-entraîner sous wd pour les règles. (51)
9. **Prérequis grok** : `train_acc≈1` d'abord. Sinon = problème capacité/optim, pas un grok lent. (14)

### Lois L1-L10 (composition / raisonnement)
10. **L1 Décomposition > scale** : la composition émerge de la **structure du calcul**, pas de la largeur (élargir DÉGRADE). (53, 55)
11. **L2 Masquage incrémental par champ** ⇒ grok par pas exact. (53)
12. **L3 Profondeur (EXACT)** : `D_reliable = 1/(1−p_step)`, `A_cascade = ∏ pᵢ`, `D_total = k/(1−A^(1/k))` (±0.003). (54, 55)
13. **L4 Récurrence fenêtrée** : `D_rec = r × D_single`, indépendant de L et P. Validé r=100000 → A=1.0. (55)
14. **L5 Budget VRAM** : `batch·L ≈ 1.9e4` (3 Go, d=256, n_blk=3) ; `L=1+4·k` ; `batch ≈ 4750/k`. (54, 55)
15. **L6 Multi-source** : difficulté ∝ nb de SOURCES à combiner ; n≥3 → décomposer obligatoirement. (53)
16. **L7 DOSC** : entraînement **séquentiel par ordre topologique** (1 champ/phase) ⇒ cascade ≈ 0.99^k vs ~0.3-0.65 joint. Gradient propre par phase. (57)
17. **L8 DOSC + anti-raccourci symétrique + interleaved** : (i) phase solo = masquer TOUTES les variables récupérables depuis cᵢ ; (ii) phase finale = interleaved 1/k ; garantie cascade ≥ 0.95 si chaque phase solo grokke ≥ 0.99. (58, **version v3 seule**)
18. **L9 Distance-scaling** : difficulté grok solo croît avec `dist(op/val → cible)` ; dist=12 plateau 0.15, dist=4 → 0.60 ; **format SBS (dist≤4) standard pour k≥3**. (62)
19. **L10 Masque bidirectionnel (CRITIQUE)** : pour k≥3 champs extraits, les masques de phase i **doivent couvrir tous les champs futurs j>i** (sinon le FFT bidirectionnel lit les champs futurs → train≠test → collapse). Fix = 2 lignes. (63)
20. **Scratchpad propre** : chaque intermédiaire **1 fois, TOUS masqués, cascade** (sinon raccourci copie). (53, 55)
21. **Cascade = seule métrique honnête** ; les métriques individuelles mentent (raccourci `m1=ans−c`). (58)

### Scaling (FFT-spécifique)
22. **`D = k^3.5 · n_blk^(2/3) · T^1.34 · C₀`** ; **γ≈0 (largeur NON-monotone, PAS un levier)**. Leviers : k ≫ n_blk > T. d_model sweet-spot 128-256. (54) ⚠️ Exposants peu calés (à confirmer ≥5 points).

### Mémoire / Sommeil (fondement du sommeil spectral)
23. **Generative replay (« sommeil »)** : entre 2 stades, le cœur dé-bruite ses pseudo-exemples → réentraînement entrelacé → oubli catastrophique comblé **89.8%**. Natif au cœur. (17)
24. **Copie verbatim** : offset fixe → exact-match 1.000 (translation = phase en Fourier). (17)
25. **Hippocampe = MemoryStore externe TF-IDF** (déjà v0) ; pont = consolidation vers le cœur. (17)
26. **Capacité cœur de faits** : d=256 parfait jusqu'à **600 faits** ; au-delà steps ∝ params (croissance naïve = piège). (37)
27. **Déclencheur auto** `maybe_auto_consolidate` (file ≥25 → sommeil arrière-plan). (37)

### Capacités prouvées
28. **Filler-gap longue distance** : G8 (8 mots RC) = 0.931 ; archi CAPABLE ; BLiMP bas = données, pas archi. (60)
29. **Double voie règle/mémoire (ADR-0016)** : règles productives dans les poids (généralisent) ; faits/irréguliers en mémoire externe routeable. (53)

---

## ❌ LISTE NOIR (ne PAS réintroduire dans OCM-26400/SCB)

- **xLSTM / sLSTM** (exponential gating, Beck 2024) — cœur de spxlm_v2, pas de FFT. Hors projet.
- **Sliding Window Attention / attention locale** (spxlm_v2).
- **Hopfield moderne comme mixer** — réfuté : cœur spectral SEUL = 0.977 vs « +adressage » = 0.48. (rapports 01, 17, 21, 56)
- **ContentAddress / adressage associatif** — réfuté (21). Ne pas réintroduire.
- **FHRR / Holographic Reduced Representation** — gains non conclusifs (artefact overfit). (46)
- **Fast Weights comme socle** — accessoire test-time, pas socle SCB. (39)
- **Quaternion FFT (QSD)** — REJETER MVP (×4 VRAM, ROCm non supporté). (56)
- **Spectral World Model / neural PDE spectrale** — ATTENTE, spéculatif. (56)
- **MoE / slot-attention / object-centric** — hors MVP, non-spectral. (17)
- **Var-JEPA (μ,σ)** — gonflé, hors budget. (52)
- **EWC / radix economy / ternaire BitNet** — vieux projet spxLM, absents des rapports.

---

## ⚠️ CAUTIONS anti-régression (avant de canoniser/citer)

1. **Versions validées seulement** : cascade **v3** (0.955-0.995), PAS v1/v2 (faux). **E6/E12**, PAS E8/B2 (gonflés par fuite format). **k=4 cascade non validée** (bug L10, prédiction 0.970 à confirmer).
2. **Valeurs à re-mesurer sur audio** : lois archi (L1-L10, γ≈0) transférables, mais valeurs (cascade 0.97, β=3.5) mesurées sur **texte/arith** → re-mesurer pour la cible audio.
3. **L10 = critique pour capture multi-lobes / DOSC** : mixer FFT bidirectionnel ⇒ masquer champs futurs dès l'extraction. Vérifier chaque test multi-champ.
4. **Scaling exposants peu calés** (β 1pt, δ 2pt, α 3pt) → confirmer avant de figer.

---

## Session 2026-07-01 (converge avec le canon, pas de contradiction)
- Gate+composition D=50 100% → aligné L7/L8 (DOSC cascade).
- Sommeil spectral +73pt → prolonge rapport 17 (generative replay 89.8%).
- `complete_formula` γ≈0 → **confirme rapport 54** (re-dérivation mineure, consistent).
- B' lobes séparés + capture L6 91% → archi B' + L6.
- Anti-grok réfuté → cohérent (γ≈0, largeur pas un levier).

*Document vivant. Dernière MAJ : 1 juil 2026, après audit DA+juges des rapports.*
