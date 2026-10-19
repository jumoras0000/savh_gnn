# Audit scientifique et correctifs Phase 3 — 19 octobre 2026

## Ce que les résultats démontrent

Le run `phase3_corrigee` est terminé, mais le meilleur checkpoint provient de
l'époque 3, avant le dégel de l'encodeur à l'époque 6. Le score affiché pendant
l'entraînement était la moyenne des six propriétés ; le checkpoint était choisi
par un autre score, `0.6 * clinical_score + 0.4 * composite_score`. Le terme
`clinical_score` n'est qu'un proxy Tox21 à seuil 0,5, pas une validation clinique.
Dans cette politique, chaque tâche non Tox21 influence directement le choix à
hauteur d'environ 6,7 %. Les seuils/calibrations finaux sont ajustés après le
choix de l'époque. Ces faits expliquent pourquoi une amélioration de l'AUC VIH
peut coexister avec le choix d'une époque antérieure. Ils ne prouvent pas qu'une
époque ultérieure soit meilleure sur une population indépendante.

## Corrections de code

- Le nouveau choix d'époque est **pré-déclaré** : `multitask` (défaut),
  `hiv_activity` (average precision VIH) ou `legacy_toxicity` (comparaison
  historique). Le score choisi, le proxy Tox21 et l'éligibilité sont enregistrés.
  Un checkpoint final n'est éligible qu'après le dégel initial de l'encodeur.
  L'entraînement exige au moins une époque après ce dégel. Les anciens
  checkpoints d'un même dossier sont archivés avant un nouveau run.
- Un encodeur Phase 3 versionné utilise maintenant les étiquettes CIP R/S par
  atome. La version précédente ne distinguait pas une paire d'énantiomères
  testée localement. Les anciens checkpoints restent chargeables avec leur
  schéma historique ; le nouveau schéma initialise seulement l'embedding de
  chiralité et reprend les autres poids de Phase 2. Il faut **réentraîner la
  Phase 3** pour utiliser cette nouvelle représentation. Les stéréoisomères E/Z
  des liaisons et la géométrie 3D ne sont toujours pas représentés.
- Un audit en lecture seule indique, pour chaque jeu et partition, le nombre de
  labels observés, la proportion de positifs, la quantité de molécules à
  scaffold Murcko vide et la performance d'un prédicteur constant basé sur la
  moyenne du train pour les régressions. Il rejette les labels non numériques.
  Commande :

  ```bash
  python -m src.validation.phase3_data_audit --data_dir data/external/phase3
  ```

- Le sampler est décrit correctement : il intercale les tâches jusqu'à
  épuisement de leurs pools. Il ne garantit pas un batch équilibré pendant
  toute l'époque. Le démarrage affiche les volumes par tâche. Une simulation
  complète sur CPU a aussi révélé que l'ajusteur pouvait ramener un batch de 2
  à 1 et faire échouer BatchNorm ; la Phase 3 impose maintenant un minimum de
  2 graphes par batch et refuse un jeu d'entraînement plus petit.
- Le « succès » du raisonneur est explicitement appelé proxy d'indépendance
  issu des prédictions individuelles. Sans courbes dose-réponse mesurées, ce
  calcul ne démontre pas une synergie de Bliss, une dose, ni une efficacité.

## Articles consultés et conséquences méthodologiques

- [MoleculeNet, Wu et al., *Chemical Science* 2018](https://doi.org/10.1039/C7SC02664A)
  décrit les tailles, cibles et protocoles recommandés. Son ESOL utilise un
  split aléatoire recommandé ; nos métriques sous split global scaffold ne sont
  donc pas comparables sans reproduire exactement les partitions.
- [Saito et Rehmsmeier, *PLOS ONE* 2015](https://doi.org/10.1371/journal.pone.0118432)
  motive l'examen de la précision-rappel lorsque les positifs sont rares. La
  politique `hiv_activity` emploie l'average precision, avec la prévalence à
  conserver comme base de comparaison. L'AUC ROC reste également rapportée.
- [ChiENN, Gaiński et al.](https://arxiv.org/abs/2307.02198) et
  [Message Passing Networks for Molecules with Tetrahedral Chirality](https://arxiv.org/abs/2012.00094)
  documentent l'insuffisance d'un graphe standard pour la chiralité. Notre
  embedding CIP est une correction minimale testable, **pas** une reproduction
  de leurs architectures ni une garantie de gain de performance.
- [Splitting chemical structure data sets, Simm et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8650276/)
  observe le regroupement de toutes les molécules acycliques sous un scaffold
  vide. Nous l'auditons sans modifier silencieusement les partitions du run
  terminé, car cela invaliderait ses métriques et demanderait un nouvel examen
  de la séparation chimique.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
  fonde l'étalonnage par température. Une température VIH à la borne 0,1 est un
  signal à inspecter sur un jeu indépendant ; ce chiffre seul ne prouve pas le
  sens ou l'ampleur d'une erreur de calibration.
- Le [cadre FDA de crédibilité pour un usage défini](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological)
  reste au stade de projet de guide. Aucune de ces métriques n'établit une
  guérison, une innocuité humaine ou une fiabilité universelle.

## Vérifications encore nécessaires sur la machine GPU

1. Exécuter l'audit des partitions et conserver le JSON avec le manifeste et
   les empreintes du checkpoint. Comparer les prévalences train/val/test et le
   baseline ESOL au RMSE 2,9482 du run actuel.
2. Auditer l'exposition Phase 2 aux molécules Phase 3 de validation/test : le
   checkpoint Phase 2 utilisé ne porte pas la provenance suffisante pour le
   prouver rétroactivement. Les nouveaux checkpoints enregistrent cette
   provenance et le chargement Phase 3 la vérifie ; voir
   [UPSTREAM_AUDIT_2026-10-19.md](UPSTREAM_AUDIT_2026-10-19.md).
3. Si un nouveau run est souhaité, employer un **nouveau dossier**, une
   politique de sélection annoncée avant lancement, et `--warm_start 0`.
   Comparer les politiques sur validation ; garder un jeu externe neuf pour
   l'estimation finale. Un meilleur score n'est pas garanti.
4. Vérifier CUDA/BF16 et un run complet réel : les tests locaux sont CPU et
   synthétiques. Évaluer séparément chaque tâche et chaque population chimique.

Les contradictions expérimentales déjà placées en quarantaine ne sont pas
« corrigées » par une étiquette inventée. Leur source doit être réexaminée.

Validation locale : 205 tests, dont un lancement Phase 3 complet sur de
minuscules CSV synthétiques et un ancien encodeur Phase 2. Cela vérifie le
chemin de code, pas la qualité prédictive, la robustesse CUDA ou la sûreté.
