# Handoff pour Claude — Panacée, 9 novembre 2026

## Mise à jour opérationnelle du 10 novembre — vNext

Lire `docs/VNEXT_MODEL_PROTOCOL.md` avant tout réentraînement. Le code
`scripts/audit_vnext_dataset.py` audite localement un CSV EC50/CC50 apparié,
les réplicats, campagnes et chevauchements exacts/de scaffold d'une proposition
de split chronologique ; ses rapports sont sous `data/vnext_audit/` (hors Git).
Il n'entraîne aucun modèle, car aucun jeu cellulaire indépendant utilisable
n'est encore présent. `webapp.translational.evaluate_assay` produit maintenant
un bilan prospectif par provenance et budgets 5/10/18 après réception complète
du lot aveugle ; une mesure censurée non concluante est `indeterminate`, pas
un négatif. Le bilan ne mesure **pas** une performance externe générale.

Point de jugement indépendant : les scores des membres A'' sont mal calibrés,
le GNN seul est moins bon que l'ensemble en PRC-AUC sur le test historique,
et ce même test a déjà été exploité pour choisir A''. Ne pas augmenter la
complexité du réseau ou recalibrer les poids sur ce test. Priorité réelle :
EC50/CC50 cellulaires appariées avec inactifs mesurés, provenance et QC ;
nouvelle validation chronologique/chimique ; comparaison RF/GNN à budget fixe.
Ne promouvoir vNext qu'après gain prospectif reproductible. Le paquet gelé de
26 structures ne doit pas être modifié ni utilisé pour former rétroactivement
A''.

La revue scientifique consolidée, générée après ce handoff initial, se trouve
dans `reports/revue_scientifique_panacee_2026-11.pdf`. Elle est téléchargeable
depuis la section « Preuves » de Studio ou l'API
`GET /api/studio/scientific-report`. Le générateur
`scripts/build_scientific_review.py` crée huit figures fondées sur les données
archivées et documente explicitement les versions incompatibles.

**Mise à jour du 10 novembre :** la source ChEMBL Jupyter brute a été copiée
localement (`data/external/chembl_hiv/chembl_hiv_merged.csv`) et auditée dans
`reports/chembl_source_audit_2026-11.json`. Les 15 842 lignes sont toutes
positives et mélangent IC50, Ki, Kd et seulement 564 EC50, sans CC50 ni
provenance d'essai cellulaire exploitable. Ne pas entraîner un modèle de
puissance/sélectivité cellulaire en traitant ces valeurs comme un endpoint
homogène. Le dossier de demande au laboratoire est généré par
`scripts/build_lab_submission.py` et téléchargeable dans Studio ; la clé et les
prédictions en sont exclues.

## Finalité et état réel

L'utilisateur veut un outil local utile pour identifier un **candidat
antirétroviral VIH testable**, puis faire progresser le projet vers une preuve
expérimentale. Il ne demande pas seulement une meilleure interface ou un
chatbot. Les quatre phases Jupyter doivent être prises en compte, mais seul le
dernier ensemble A'' du 6 novembre 2026 est autorisé comme modèle de
classement VIH dans Studio. Les anciens poids Kaggle et les versions
intermédiaires ne doivent pas remplacer silencieusement A''.

À cette date, **aucune nouvelle molécule n'a une efficacité antivirale
expérimentalement confirmée dans ce projet**. Le logiciel prépare et audite
des données ; il ne réalise pas les expériences. Ne pas employer « médicament
validé », « candidat clinique » ou « 32/34 médicaments découverts » pour les
résultats actuels.

## État Git et environnement

- Dépôt : `https://github.com/jumoras0000/savh_gnn.git`, branche `main`.
- Dernier commit au handoff : `e8e6693` (`Add blinded assay intake and
  translational evidence gates`). Le précédent `0ecb0b1` fige les preuves
  A'' et la nouvelle interface. Le push GitHub a réussi ; local et `origin/main`
  étaient synchrones (`0 0`).
- Dossier de travail : `D:\02_PROJETS\AI_GNN_Panacee\Projet_Panacee`.
- ASGI local : `http://127.0.0.1:8000/`, processus Uvicorn vivant au handoff.
  `/legacy` existe pour compatibilité ; le flux principal est `/`.
- `note` dans le projet et `../.playwright-mcp/` sont des fichiers non suivis
  de l'utilisateur : **ne pas les modifier, supprimer ou committer**. Les
  données sous `data/experiment_package/`, `data/assay_results/`,
  `data/development_evidence/`, `data/model_audit/` et les splits HIV locaux
  sont ignorées par Git.
- Les gros poids sont dans Git LFS. Sur une autre machine : `git lfs pull`.
  Les splits Jupyter `data/external/phase3/hiv/hiv_{train,val,test}.csv` ne sont
  **pas** publiés dans Git ; ils sont présents localement à ce handoff.
- Connexion au JupyterHub existant : `webapp/jupyter_sync.py`, via
  `connection()` puis `Hub(*connection())`. L'URL et le jeton viennent de
  variables d'environnement ou de la configuration VS Code. Ne jamais
  afficher/committer les jetons. La source scientifique est
  `panacee/Projet_Panacee` sur JupyterHub ; vérifier les dates et empreintes
  avant de copier une nouvelle version, plutôt que supposer que GitHub est la
  dernière source.

## Quatre phases et référence scientifique

1. **Phase 1** : encodeur moléculaire pré-entraîné ; composant, pas score VIH.
2. **Phase 2** : toxicité Tox21 historique ; ne pas l'afficher comme preuve de
   sécurité du candidat VIH.
3. **Phase 3** : GNN VIH propre + 4 RF ; l'ensemble A'' augmenté utilise le GNN
   inchangé et les RF augmentés par 15 580 actifs ChEMBL.
4. **Phase 4** : criblages exploratoires ZINC et repurposing, 124 + 6 hits
   propres disponibles ; les scores de bibliothèque ne sont pas des mesures
   biologiques nouvelles.

Artefacts de référence :

- `checkpoints/model_A_prime_augmented/manifest.json` : modèle créé le
  2026-11-06, GNN propre, 4 fingerprints RF.
- `checkpoints/model_A_prime_augmented/registry.json` : SHA-256 des 12
  artefacts, SHA-256 des trois splits Jupyter, taille AD 49 150.
- `checkpoints/model_A_prime_augmented/validation_reproduction.json` : huit
  métriques exactes reproduites ; ROC-AUC 0,620132, PRC-AUC 0,154147,
  EF@1 % 16,5702, DrugBank 32/34 sur le montage décrit dans le rapport.
- Test MoleculeNet HIV : 3 778 structures, 84 actives (2,223 %). La sélection
  de A'' a déjà considéré ce test ; **ne pas** le traiter comme un jeu vierge
  pour choisir les hyperparamètres de la prochaine version.
- `checkpoints/model_A_prime_augmented/{reference_scores,
  calibration_scores,ad_train_ecfp4}.npz` : percentile fixe (sans labels),
  calibration GNN et domaine ECFP4.
- Couverture conformal GNN interne 93,09 % pour cible 90 %, mais 67,44 % des
  ensembles sont ambigus. Elle ne vaut pas garantie sur nouvelle chimie.
- Le manifeste compte 189 actifs ajoutés partageant un scaffold avec des
  antirétroviraux ; le rapport expose ce fait. Auditer toute revendication de
  généralisation ou d'absence de fuite, notamment DrugBank.

`scripts/audit_a_prime_members.py` a été exécuté. Audit local ignoré par Git :
`data/model_audit/member_audit.json`. Sur le test figé :

| Modèle | ROC-AUC | PRC-AUC | Précision top 100 |
| --- | ---: | ---: | ---: |
| GNN | 0,5765 | 0,0776 | 0,17 |
| RF ECFP4 | 0,5971 | 0,1073 | 0,18 |
| RF ECFP6 | 0,5864 | 0,0949 | 0,13 |
| RF MACCS | 0,6719 | 0,0878 | 0,12 |
| RF RDKit | 0,6247 | 0,0563 | 0,08 |
| A'' rank-average | 0,6201 | **0,1541** | **0,20** |

Les scores bruts des membres sont mal calibrés : Brier 0,039 à 0,140 contre
0,0217 pour la constante empirique à la prévalence. Le rang A'' n'est **pas une
probabilité**. A'' est intéressant pour l'enrichissement des premiers rangs,
moins pour la discrimination globale. Cet audit n'autorise pas de retuner
l'ensemble sur le même test.

## Ce qui a été mis en place

### Studio (`webapp/`)

- `webapp/studio.py`, `studio_support.py` : classement A'' de 1–100 SMILES,
  percentile de référence stable entre lots, domaine contre le vrai train +
  augmentation, conformal **GNN seul**, métriques/provenance vérifiées.
- `webapp/static/studio.{html,js,css}` : parcours court Cribler → Hits →
  Preuves → Validation ; testé sur desktop et largeur mobile 390 px, sans erreur
  JavaScript ni débordement horizontal.
- `webapp/server.py` : `GET /api/studio/overview`, `POST /api/studio/screen`,
  `GET /api/studio/hits`, `GET /api/studio/package`,
  `GET /api/studio/assay-template`, `POST /api/studio/assays`,
  `GET /api/studio/development`.

### Paquet et résultats aveugles

- `scripts/build_candidate_package.py` a créé et gelé localement 26 structures :
  12 ZINC, 6 repurposing, 4 ARV de référence, 4 **comparateurs négatifs du
  criblage** MoleculeNet (non contrôles négatifs certifiés d'un essai).
- `data/experiment_package/submission_blinded.csv` est le seul fichier à
  transmettre au laboratoire. `key_local.json` et `predictions_local.json`
  restent locaux ; `package_manifest.json` contient leurs empreintes. Le
  constructeur vérifie/réutilise un paquet déjà gelé et ne le régénère pas.
- SHA-256 de la soumission gelée :
  `70d5517340dce2cf565983487d2beb7e3bd98ad9d80c806cfeda88fd43c035d3`.
- `assay_results_template.csv` : colonnes `blind_id,assay_id,replicate,
  ec50_um,cc50_um`, en µM ; `>nombre` est une borne censurée. Deux réplicats,
  même assay et contrôle de dispersion sont exigés pour un lot complet.
- `webapp/translational.py` et `scripts/ingest_blind_assays.py` : vérification
  de l'identité/complétude, gel du CSV reçu, résultats décodés seulement pour
  un lot complet/cohérent, médianes et SI seulement si valeurs exactes,
  association descriptive aux percentiles pré-gelés. Seuils exploratoires
  pré-enregistrés : EC50 ≤ 1 µM et SI ≥ 10. Ils ne sont ni une règle clinique
  ni une validation de l'essai.
- Le statut `data_complete` signifie seulement **cohérence informatique**.
  La qualité de l'essai doit être attestée séparément par le laboratoire.
- `scripts/register_development_evidence.py` archive localement les rapports
  de QC, confirmation, résistance, ADME/sécurité et dossier réglementaire ;
  `evidence_recorded` ne signifie jamais « validé ».
- `scripts/evaluate_external_hiv.py` évalue un jeu indépendant `smiles,label`
  en refusant identités et scaffolds déjà présents dans train/val/test/actifs
  ajoutés. Aucune évaluation externe admissible n'est enregistrée à ce jour.

## Vérifications de reprise

À exécuter dans `Projet_Panacee` :

```powershell
git status --short
git fetch origin main
git rev-list --left-right --count HEAD...origin/main
python -c "from webapp.studio_support import verify_registry; v=verify_registry('checkpoints',datasets=True); print(v['status'],v['checked'],v['problems'])"
python -m pytest tests/test_translational.py tests/test_studio.py tests/test_studio_support.py tests/test_lab.py tests/test_phase4.py tests/test_phases.py tests/test_load.py -q
python -m ruff check webapp/studio.py webapp/studio_support.py webapp/translational.py scripts/audit_a_prime_members.py scripts/build_candidate_package.py scripts/ingest_blind_assays.py scripts/register_development_evidence.py
Invoke-RestMethod http://127.0.0.1:8000/api/studio/development
```

Au handoff : registre `verified`, 12 artefacts concordants ; 55 tests ciblés
passent ; interface et API ont été vérifiées. La suite complète conserve **six
échecs préexistants** dans `tests/test_audit_contracts.py`,
`test_auxiliary_audit.py`, `test_metrics_edges.py`, `test_model_pipeline_audit.py`
et `test_upstream_phase_audit.py` : écarts de contrats historiques dans le
code d'entraînement/audit, sans lien avec les nouvelles routes Studio. Ne pas
les masquer ; les examiner séparément avant toute nouvelle version de modèle.

Si le serveur doit être relancé après modification du backend : arrêter son
processus Uvicorn courant, puis
`python -m webapp.run --host 127.0.0.1 --port 8000`.

## Priorités pour demain (ordre recommandé)

1. **Valider la traçabilité scientifique** : relire les rapports Jupyter Phase
   3/4, les filtres anti-fuite, la sélection des 26 structures et les
   définitions des comparateurs. Ne pas changer le paquet après une première
   mesure. Corriger tout écart documenté, avec tests de non-régression.
2. **Préparer le transfert au laboratoire partenaire** : fiche de demande
   centrée sur identité des structures, objectifs, type de résultats à rendre,
   unités, limites de mesure, réplicats, contrôles validés par le laboratoire,
   rapport QC, provenance, confidentialité. Le logiciel a un CSV de retour,
   mais pas de protocole ni de résultats biologiques.
3. **Contrôler l'analyse prospective** : une cohorte enrichie de 18 candidats
   n'est pas représentative d'une bibliothèque générale. Ajouter des
   intervalles d'incertitude et une évaluation par source chimique avant toute
   revendication de performance externe ; garder les contrôles et les
   candidats distincts.
4. **Concevoir vNext sans fuite** : données quantitatives EC50/CC50 avec
   provenance d'essai et censure, séparation par campagne/date/scaffold,
   baselines RF/GNN, objectif précision/enrichissement à budget fixe, puis
   calibration sur validation distincte. Les 26 résultats prospectifs sont
   d'abord un test gelé ; ils ne servent pas à entraîner la version qui les
   a prédits. Aucune promotion de vNext sans gain indépendant reproductible.
5. **Étapes biologiques/réglementaires** : n'afficher comme faites que des
   pièces réelles fournies par le partenaire ; mécanisme, résistance,
   confirmation, ADME et sécurité restent à produire. Les lignes directrices
   [FDA pré-IND antiviraux](https://www.fda.gov/about-fda/cder-offices-and-divisions/division-antivirals-dav-information-pre-ind-submissions)
   et [NIH sur les objectifs d'un ARV](https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv/principles-antiretroviral-therapy)
   cadrent les catégories de preuves.

## Décision scientifique actuelle

La prochaine preuve décisive est **l'activité cellulaire et la sélectivité
mesurées indépendamment** pour le paquet gelé. Réentraîner sur les mêmes labels
binaires avant ces mesures ne démontre pas qu'un médicament VIH prometteur a
été trouvé. Le résultat attendu de demain est donc un dossier prêt à être
transmis au laboratoire et un plan de modèle vNext testable, pas l'annonce
d'un composé validé.
