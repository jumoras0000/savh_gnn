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
  QED, masse, LogP, Lipinski, alertes structurelles → export CSV.
- **Explorer** : lire les hits propres ZINC et repurposing, filtrer les lignes,
  afficher 20 résultats à la fois.
- **Vérifier** : voir les métriques du rapport et le rôle exact des phases.

Les anciens outils restent sur `/legacy`, sans être la page d'accueil. Le
nouveau Studio n'appelle aucun service cloud. Il ne calcule pas de domaine
d'applicabilité ni de conformal pour les nouveaux lots, car les références de
train et les calibrations ne sont pas présentes dans le bundle local.

## Contrats et limites

- `GET /api/studio/overview` : bundle retenu, validation rapportée, phases,
  comptes des hits et limites de lecture.
- `POST /api/studio/screen` : 1–100 SMILES ; rang moyen **plus élevé = meilleur**,
  uniquement comparable dans le lot soumis.
- `GET /api/studio/hits?source=zinc|repurposing` : seuls les CSV `*_clean.csv`
  de la Phase 4, non recalculés par la web app.

Le rapport `augmented_report.json` du 6 novembre correspond au modèle retenu,
mais les rapports intermédiaires contiennent des baselines de versions
différentes. Studio n'agrège donc pas leurs chiffres dans une unique métrique.
La validation prospective expérimentale reste hors du logiciel.
