# Panacée Studio v2 — décision produit et traçabilité

## Ce que le modèle peut réellement faire

Le rapport scientifique du 6 novembre 2026 retient **A'' propre** (GNN
`panacee_phase3_hiv_chembl_clean.pth` + quatre RF augmentées) pour un
**classement relatif de molécules anti-VIH**. Le score est un rang moyen parmi
les molécules du lot ; il n'est ni une probabilité absolue, ni une valeur
d'affinité, ni une preuve d'activité. Le test rapporté compte 3 778 molécules,
84 actives, et le gate DrugBank retrouve 32/34 ARV au top 10 % du pool.

L'encodeur Phase 1 est une base d'apprentissage, pas un score VIH. Le checkpoint
Phase 2 correspond à Tox21 ; le rapport de novembre ne fournit pas ici une
validation clinique permettant de le mélanger au score VIH. La Phase 4 est un
jeu de résultats issus du même pipeline, pas un autre modèle.

## Pourquoi l'ancienne interface n'aidait pas le modèle

1. L'accueil priorisait les courbes d'époques 2026-10-21 plutôt que le modèle
   final du 2026-11-06. Les KPI d'entraînement ont été perçus comme les
   métriques du modèle livré.
2. Les sélecteurs mélangeaient checkpoints, objectifs, QED, toxicité, ancien
   Model A et recherche conversationnelle. Plusieurs fonctions demandaient de
   connaître l'architecture avant de poser une question simple.
3. Le laboratoire inversait le sens du rang A'' : `score_model_a` utilise
   `rankdata` puis le criblage scientifique trie **par score décroissant**.
   Le laboratoire présentait auparavant le score inférieur comme meilleur.
   Corrigé et couvert par un test.
4. `cheminfo.applicability_domain` compare une nouvelle molécule au petit
   catalogue local, alors que le rapport calcule le domaine contre le train.
   Ce proxy ne doit pas être présenté comme le domaine validé du pipeline.
5. Les résultats ZINC et repurposing étaient enfouis dans des menus, alors
   qu'ils sont les sorties concrètes du modèle (124 et 6 hits propres).

## Nouveau parcours

- **Cribler** : coller un lot de SMILES → un seul modèle A'' → rang décroissant,
  percentile sur une référence fixe, domaine Tanimoto contre le train augmenté,
  conformal du GNN calibré sur la validation, QED et alertes → export CSV.
- **Explorer** : lire les hits propres ZINC et repurposing, filtrer les lignes,
  afficher 20 résultats à la fois.
- **Vérifier** : voir les huit métriques reproduites, les empreintes SHA-256,
  le rôle exact des phases et l'état de la validation externe.

Les anciens outils restent sur `/legacy`, sans être la page d'accueil. Le
nouveau Studio n'appelle aucun service cloud. Les splits HIV train/val/test ont
été recopiés depuis Jupyter, et les 15 580 actifs ajoutés sont dans le bundle.
`build_studio_support.py` a reproduit **exactement les huit métriques** du
rapport, y compris DrugBank 32/34, puis figé les références et hashes dans
`registry.json`. Toute modification d'un artefact requis rend ce registre
invalide et empêche le Studio de scorer.

Le percentile fixe utilise les scores bruts des cinq membres sur 3 778
structures de test **sans utiliser les labels** ; il reste identique si un autre
candidat est ajouté au lot. Il ne porte pas les métriques du rank-average
historique : seul ce dernier a été évalué dans le rapport. Le domaine v2
compare à 49 150 structures train+augmentation ; ce n'est pas le petit
catalogue historique. Le conformal est calibré sur 3 772 structures de
validation et s'applique au **GNN seul**, pas à l'ensemble ; couverture interne
observée sur test 93,09 % pour une cible 90 %, avec 67,44 % de résultats
ambigus. Cette couverture interne n'est pas une garantie sous déplacement de
distribution externe.

## Contrats et limites

- `GET /api/studio/overview` : bundle retenu, validation reproduite, registre,
  phases, comptes des hits et état de la validation externe.
- `POST /api/studio/screen` : 1–100 SMILES ; rang moyen **plus élevé = meilleur**,
  uniquement comparable dans le lot soumis ; percentile fixe, domaine v2 et
  conformal GNN séparés.
- `GET /api/studio/hits?source=zinc|repurposing` : seuls les CSV `*_clean.csv`
  de la Phase 4, non recalculés par la web app.
- `GET /api/studio/package` : liste aveugle de 26 structures préparée pour
  évaluation indépendante ; la clé de décodage reste locale et non servie.

`evaluate_external_hiv.py` refuse de présenter les structures déjà vues
(identité canonique ou scaffold) comme validation externe. Aucun jeu externe
indépendant étiqueté n'était disponible dans le projet : statut explicite
`not_performed`, pas de performance inventée. Le paquet de 26 structures
(12 ZINC diversifiées, 6 repurposing, 4 ARV et 4 références inactives) est
préparé localement, sans aucun résultat expérimental.

Le rapport `augmented_report.json` du 6 novembre correspond au modèle retenu,
mais les rapports intermédiaires contiennent des baselines de versions
différentes. Studio n'agrège donc pas leurs chiffres dans une unique métrique.
La validation prospective expérimentale reste hors du logiciel.
