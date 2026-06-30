# CAHIER DES CHARGES — MODELE IA OMNIMODAL COMPLETE
# Date: 2026-06-17
# Architecture: SPXLM (FFT bidirectionnel + diffusion-fill + xLSTM + scratchpad DOSC)

## PRINCIPE DIRECTEUR
Le modele doit pouvoir GRANDIR, GROSSIR sans limites.
La consommation de VRAM est fonction des besoins du modele.

## 1. APPRENTISSAGE DES PRIMITIVES (grok 1.0, verifie severement)
- TOUT le vocabulaire anglais (1M mots) + francais (60K)
- Grammaire complete (toutes les regles)
- Conjugaison complete (tous les verbes, tous les temps)
- Phonologie, morphologie, etymologie
- Affixes, morphemes, lexemes, radicaux
- Syntaxe, semantique, synonymes, nuances
- Tout en UNE PASSE (capture simultanee)

## 2. DOMAINES DE CONNAISSANCE (integralite)
- Mathematiques (arithmetique, algebre, geometrie, analyse)
- Physique (mecanique, quantique, thermodynamique)
- Developpement: Python, TypeScript, React, Vue.js, JavaScript, Java, Kotlin, CSS
- Architectures, DevOps, Securite, Pentest expert, OSINT
- Computer use, browser use (interaction complete)
- Biologie, Medecine, Histoire, Geographie, Sociologie
- Psychologie, Litterature, Musique, Economie, Finance
- Robotique, Mecanique
- IoT (Arduino, Raspberry Pi)
- Mobile dev (Android, iOS)
- Mobile use (Android, iOS)
- Radar satellite (detection, exploitation)
- Object detection (objects, personnes, plaques immatriculation)
- Object identification, Object following
- Object detection par sondes (wifi, radio, bluetooth)
- Tous les protocoles reseaux (couche OSI complete 1-7)
- Capacites de predictions type JEPA (base sur physique, sans frankenstein, tout integre)

## 3. APPRENTISSAGE AUTONOME
- Le model doit apprendre a partir d'une source donnee
- Respecter le PROTOCOLE d'entrainement
- Comprendre, valider ce qu'il doit apprendre
- Apprendre tout ou uniquement ce qui est demande dans le prompt
- Generaliser apres COMPREHENSION
- Sources possibles:
  - Fichier PDF (texte, images, graphiques)
  - Lien de site internet
  - Image
  - Video
  - Lien YouTube (aller sur le lien, comprendre, extraire, apprendre)
- Attention aux phases de sommeil

## 4. CAPACITES (verifiees, ne rien laisser passer)
- Comprendre parfaitement le prompt
- Raisonner tres longuement (recurrence fenetree)
- Etre intelligent (la comprehension, pas la memoire)

## 5. CREATION D'ARTEFACTS
- Generation de texte (code, documents, explications)
- Generation d'images
- Generation de videos
- Generation 3D
- Generation de mondes interactifs

## 6. TESTS
- Derouler TOUS les tests du fichier Tests.md du dossier Besoins
- Enregistrer le resultat des tests pour verifications

## 7. ARCHITECTURE
- FFT bidirectionnel (raisonnement via diffusion-fill)
- FFT causal (generation fluide)
- xLSTM (recurrence, memoire long terme)
- Scratchpad SBS + DOSC
- Recurrence fenetree (profondeur sans params)
- Champ partage multi-niveaux (associations any->any)
- Sommeil multi-phase (leger/moyen/profond, macro->micro)
- d=256, n_blk=3-4 (evolutif)
- Croissance: le modele grossit selon les besoins (pas de limite fixe)

## 8. PROTOCOLE D'ENTRAINEMENT (obligatoire, voir skill spxlm-training-protocol)
- Pre-entrainer les primitives jusqu'au grok (1.0)
- DOSC: curriculum sequentiel par dependance
- Masquage incremental (single_frac=0.3)
- Anti-raccourci symetrique
- Diffusion-fill bidirectionnel
- Format SBS avec separateurs | et padding
- weight_decay=0.1, warmup 500+, betas=(0.9, 0.95)
- Sommeil/consolidation entre stages
- Per-step accuracy > 0.99 avant composition

## 9. VALIDATION (cible: 98%+ comme SpectraLM)
- Addition: 100% (ATEINT)
- Morphologie: cible 98% (IDs numeriques)
- Vocabulaire: cible 99% (champ partage)
- Grammaire: cible 95%
- Raisonnement: cible 90%
