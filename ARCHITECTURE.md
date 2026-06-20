# ARCHITECTURE OCM-26400 — carte implémentation vs cahier des charges

**Date:** 20 Juin 2026
**Réf spec:** `Besoins/Besoins_Tests.md` (cahier des charges du modèle omnimodal)
**Paradigmes respectés** (donnés par l'utilisateur) : nouvelle architecture, compositionnalité,
grokking, capture-une-passe, alignement amodal, vérification symbolique (pas de Frankenstein),
apprentissage+généralisation après compréhension, honnêteté scientifique (verdict panel d'experts).

Cette carte est HONNÊTE : elle sépare les **mécanismes prouvés** (validés empiriquement,
tests verts) du **système complet** visé par le spec (1M mots, multimodal, computer use…).
Le noyau cognitif est implémenté et intégré ; les capacités "production" sont l'étape suivante.

---

## Le noyau cognitif implémenté (`ocm26400/`)

Le cycle cognitif complet est câblé et testé (**85 tests verts**) :

```
   entrée → [ENCODAGE AMODAL] → concept (AMV-256)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          [RETRIEVE mémoire]              [RAISONNER composer]
          (KnowledgeBase)                 (crown-jewel, récursion)
                  │                               │
          connu ? ├─oui→ réponse O(1)              │
                  └─non──────────────────────► [VÉRIFIER gate]
                                                  │ Verifier symbolique
                                          valide ? ├─oui→ [APPRENDRE store]
                                                  │       → désormais 'retrieved'
                                                  └─non→ [ABSTENTION]
                                                          'je ne sais pas'
                                                          → déclenche apprentissage
```

### Modules et mapping au spec

| Module | Rôle | Spec visé | Statut |
|---|---|---|---|
| `amv.py` | AMV-256 (ent\|prop\|op\|meta), partition meta 3 rôles | §A1.3 structure mentalese | ✅ prouvé (7 tests) |
| `verifier.py` | SymbolicDict + Verifier, V(), compose(a,b,op_id) | §2.2 gate symbolique déterministe | ✅ prouvé (9 tests) |
| `acsp.py` | loss ACSP (align+step+sparse+consist), consist_term | §2.4 Causal Rigor Loss | ✅ prouvé (8 tests) |
| `infonce.py` | InfoNCE stable (L_consist core) | §2.4 cohérence multimodale | ✅ prouvé (7 tests) |
| `reasoner.py` | ReasonerBlock + LSRA (gate τ_grok + ANOMALIE) | §3 test-time compute | ✅ prouvé (9 tests) |
| `learned_vocab.py` | dictionnaire DENSE V>64, anti-collapse | primitives illimitées | ✅ prouvé (14 tests, survie dense +100pt) |
| `concept_amodal.py` | alignement multi-vue + ancrage AMV | §A1.3 f_T~f_A~f_V~v_C, capture 1 passe | ✅ prouvé (5 tests, retrieval 100%) |
| `morphology.py` | MorphologyVerifier dispatch op_id (conjugaison) | conjugaison + règles vérifiables | ✅ prouvé (5 tests, dispatch VALIDÉ) |
| `knowledge_base.py` | retrieval cosinus + abstention | recherche base + "je sais pas" | ✅ prouvé (6 tests, precision@1 100%) |
| `cognitive_agent.py` | cycle retrieve→raisonner→vérifier→apprendre | spec "apprentissage/évolution" | ✅ prouvé (4 tests, accuracy 100%) |

### Expériences (démonstrations empiriques)

| Expérience | Démontre | Résultat honnête |
|---|---|---|
| `experiment_composition` | crown-jewel arithmétique Z₁₁ | décomp +99.5pt vs one-shot (1131 triples neufs) |
| `experiment_linguistic` | crown-jewel morphologie anglaise | décomp +100pt (12 triples neufs) |
| `experiment_linguistic_dense` | survie one-hot→dense | +100pt (ortho+random), le grok ≠ aligné axes |
| `experiment_vocab_scale` | scaling V>64 (Z₁₂₀) | grok règle 99.7% brut ; gate stricte limitée à 120/64 |
| `experiment_refinement` (P3) | gate calibrée + abstention | OOD refusé 100%, AUROC 1.0 |
| `experiment_amodal` | alignement amodal | retrieval 1.6%→100%, ancrage 100% |
| `experiment_conjugation` | conjugaison multi-temps op_id | dispatch VALIDÉ (1 block 3 temps) ; flat-map généralise NON (crown-jewel: needing décomp) |
| `experiment_recursion` | récurrence fenêtrée profondeur k | 100% à profondeur 2-5 (raisonner longuement) |
| `experiment_knowledge` | base de connaissance | precision@1 100%, abstention OOD 100%, cycle apprentissage |
| `experiment_agent` | agent cognitif auto-apprenant | 118 faits appris (acc 100%), 682 retrieved, courbe 62%→85% |
| `spxlm_v6/measure_single_forward` | encodeur spectral single-forward | 96.4% ≥ diffuse 91.3% (réfute DA) |
| `spxlm_v6/experiment_v6_bridge` (P4) | pont v6→AMV | 100% sur dico fixe ; 0% généralisation (cibles arbitraires) |

---

## Couverture du cahier des charges (`Besoins_Tests.md`) — honnête

| Section spec | Couverture | Détail |
|---|---|---|
| **§1 Apprentissage primitives (grok)** | ⚠️ mécanisme ✅ / échelle ❌ | grok compositionnel prouvé (crown-jewel), mais ~120 symboles pas 1M mots. LearnedVocab V>64 lève le plafond one-hot. |
| capture simultanée tous champs 1 passe | ⚠️ | alignement amodal aligne les vues ; v6 encode les champs en 1 forward |
| **§2 Domaines connaissance** | ❌ | pas de connaissances encyclopédiques (math/physique/code…) — pas dans le scope du noyau |
| **§3 Capacités** comprendre/raisonner long | ✅/⚠️ | raisonnement compositionnel+récursif ✅ ; compréhension de prompt libre ❌ |
| **§4 Création artefacts** | ❌ | génération texte/image/vidéo/3D non implémentée |
| **§5 TESTS** (dialogue, grammaire, apprentissage) | ⚠️ | grammaire (morphology) ✅, apprentissage (agent) ✅ ; dialogue libre / multimodal ❌ |
| "je ne sais pas → mode apprentissage" | ✅ | abstention (P3) + cycle agent (retrieve→raisonner→apprendre) |
| chat modes audio/image/vidéo | ❌ | harness amodal simulé (concept_amodal), pas de vraies modalités |
| RAG / code-gen / MCP / computer use | ❌ | retrieval base (KB) ✅ ; reste non implémenté |

### Frontière honnête (ce qui reste à bâtir pour le système complet)

1. **Échelle du vocabulaire** : passer de ~120 à des milliers/millions de symboles. Borné par
   le packing ent=64-dim (montré en P2) ; leviers = dim>64 (casse la partition AMV) ou
   hiérarchie de dictionnaires.
2. **Vraies modalités** : remplacer les encodeurs simulés (concept_amodal) par v6 (texte) +
   encodeurs audio/image réels. L'interface (bridge P4) existe mais ne généralise pas
   (cibles arbitraires — levier : embeddings symbole appris fonction du hidden).
3. **Connaissances & domaines** : peupler la KnowledgeBase avec des faits réels (math, code…).
4. **Génération d'artefacts** : décodeur AMV→texte/image (miroir du bridge).
5. **Dialogue & intégration externe** (web/RAG/MCP/computer use) : couche applicative au-dessus du noyau.

---

## Reproduire
```bash
cd MathsBase
python3 -m pytest ocm26400/ -q                       # 85 tests verts
for e in composition linguistic linguistic_dense vocab_scale refinement amodal conjugation recursion knowledge agent; do
  python3 -m ocm26400.experiment_$e                  # démos (~15-90s chacune, GPU)
done
```
Voir `ocm26400/STATUS.md` pour le détail des résultats et `ocm26400/EXPERT_PANEL_VERDICT.md`
pour la méthodologie (collège d'experts → DA → juge) qui garantit l'honnêteté des claims.
