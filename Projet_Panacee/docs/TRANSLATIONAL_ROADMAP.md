# Panacée — du classement VIH au candidat testable

## État constaté (9 novembre 2026)

- Modèle A'' du 6 novembre : 5 membres (GNN + 4 RF), poids figés et métriques
  reproduites sur le test HIV MoleculeNet. ROC-AUC 0,6201 ; PRC-AUC 0,1541 ;
  84 positifs sur 3 778 (prévalence 2,22 %) ; DrugBank 32/34. Ce résultat
  démontre un **enrichissement de classement**, pas une activité mesurée des
  nouvelles molécules.
- Le conformal GNN est ambigu pour 67,44 % du test interne. Le domaine est
  calculé contre 49 150 structures train + augmentation, sans preuve de
  généralisation prospective.
- Paquet gelé local : 12 hits ZINC, 6 hits repurposing, 4 ARV de référence et
  4 comparateurs négatifs du criblage MoleculeNet. Ces derniers ne sont pas
  des contrôles négatifs certifiés d'un essai de laboratoire.
- Aucun résultat cellulaire indépendant, dossier ADME, mécanisme confirmé ou
  candidat clinique n'est présent dans le projet.

## Profil-cible de produit à fixer avec le partenaire

L'indication visée et la valeur ajoutée par rapport aux traitements actuels
doivent être explicites : activité sur variants résistants, nouvelle cible,
tolérance, interactions, modalité d'administration ou usage en association.
La puissance antivirale seule ne suffit pas : la barrière à la résistance,
la tolérance et les interactions font partie de la valeur clinique.
Source : [NIH, Principles of Antiretroviral Therapy](https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv/principles-antiretroviral-therapy).

## Gates et données à produire

| Gate | Livrable vérifiable | État |
| --- | --- | --- |
| 0. Gel informatique | Liste aveugle, clé locale, prédictions, SHA-256, seuils exploratoires pré-enregistrés | Prêt |
| 1. Activité cellulaire | Résultats individuels avec identifiants aveugles, réplicats, EC50, CC50 et unités ; rapport de qualité de l'essai du laboratoire | En attente |
| 2. Confirmation | Répétition indépendante des hits et contrôles adaptés, relation concentration–effet, sélectivité confirmée | En attente |
| 3. Virologie | Mécanisme d'action, variants pertinents, potentiel de résistance et résistance croisée, compatibilité avec traitements existants | En attente |
| 4. Développement | Identité/pureté, solubilité, stabilité, exposition, toxicologie, interactions, fabrication | En attente |
| 5. Clinique | Dossier réglementaire et essais autorisés | En attente |

Ces catégories suivent les domaines de données décrits par la
[FDA pour les antiviraux VIH](https://www.fda.gov/about-fda/cder-offices-and-divisions/division-antivirals-dav-information-pre-ind-submissions),
les [études virologiques et de résistance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/antiviral-product-development-conducting-and-submitting-virology-studies-agency)
et le [dossier IND](https://www.fda.gov/drugs/types-applications/investigational-new-drug-application-ind).
Les expériences biologiques, leur qualité et leur interprétation relèvent du
laboratoire partenaire ; le logiciel n'en crée aucune donnée.

## Flux local déjà opérationnel

1. `python scripts/build_candidate_package.py` crée le paquet **une seule fois**.
   Un paquet déjà gelé est vérifié et réutilisé, non régénéré.
2. `python scripts/build_lab_submission.py` assemble le dossier aveugle
   `data/experiment_package/dossier_laboratoire_aveugle.zip` : liste des 26
   structures, modèle de résultats, demande de mesures et manifeste public.
   Le ZIP est téléchargeable dans Studio. Il **ne contient pas** la clé locale
   ni les prédictions.
3. Le modèle `data/experiment_package/assay_results_template.csv` comporte
   `blind_id,assay_id,replicate,ec50_um,cc50_um`. Les concentrations sont en µM ;
   une borne droite s'écrit `>10`. Deux réplicats par structure sont requis.
4. Importer dans Studio ou exécuter
   `python scripts/ingest_blind_assays.py chemin/mesures.csv`.
   Les identifiants, les doublons, les valeurs, la complétude et une tendance
   des comparateurs sont contrôlés. Un lot incomplet **n'est pas décodé**.
5. Les rapports, les prédictions et les données brutes restent dans `data/`
   ignoré par Git. Le SHA-256 lie chaque rapport au CSV source. Le statut
   `data_complete` n'est **pas** une attestation de qualité de l'essai.
6. Les pièces des gates suivants peuvent être archivées localement par
   `python scripts/register_development_evidence.py lab_qc chemin/rapport.pdf --title "Rapport QC laboratoire"`.
   Les autres gates acceptés sont `replication`, `mechanism_resistance`,
   `adme_safety` et `regulatory`. Leur statut devient `evidence_recorded`,
   jamais « validé » automatiquement. Les pièces et leurs SHA-256 restent sous
   `data/development_evidence/`, hors Git et hors téléchargement web.

Les seuils exploratoires pré-enregistrés (`EC50 ≤ 1 µM`, indice de sélectivité
`CC50/EC50 ≥ 10`) servent à trier des signaux, pas à qualifier un médicament.
Le classement des 18 candidats enrichis ne permet pas d'estimer une performance
externe générale ; l'analyse de corrélation avec la puissance est descriptive.

## Amélioration du modèle : ordre de travail

Le contrat exécutable du jeu vNext, l'audit sans entraînement et les règles
de promotion figurent dans [`VNEXT_MODEL_PROTOCOL.md`](VNEXT_MODEL_PROTOCOL.md).
Le rapport prospectif des résultats aveugles distingue désormais les signaux
déterminés des mesures censurées indéterminées, par source et par budget de
sélection gelé (5/10/18). Aucun résultat biologique n'étant encore présent,
aucune métrique prospective n'est revendiquée à ce stade.

1. **Changer la cible d'apprentissage.** Le binaire `HIV_active` mélange des
   contextes d'essai. Pour sélectionner un médicament, constituer des données
   de puissance quantitative, cytotoxicité, limites de mesure, mécanisme et
   provenance de l'essai. Garder les labels censurés comme bornes, pas comme
   valeurs exactes. Ne pas fusionner des protocoles hétérogènes sans contrôle.
2. **Verrouiller les jeux d'évaluation.** Dédupliquer par structure canonique,
   sels/tautomères et scaffold ; séparer par campagne/laboratoire et par date.
   Les 26 candidats aveugles servent d'évaluation prospective et ne doivent
   pas alimenter le modèle qui les a classés.
3. **Comparer des baselines simples avant complexification.** RF ECFP et GNN
   seuls, ensemble actuel, modèle quantitatif et modèle multitâche activité–
   cytotoxicité. Comparer à budget de sélection fixe : précision parmi les 10
   ou 20 premiers, enrichissement, PRC-AUC, calibration, couverture du domaine
   et intervalles d'incertitude. Le test du 6 novembre est déjà utilisé pour
   choisir A'' ; ne pas l'utiliser pour régler un nouvel ensemble.

   L'audit reproductible `python scripts/audit_a_prime_members.py` a été
   exécuté localement sur ce test figé :

   | Modèle | ROC-AUC | PRC-AUC | Précision top 100 |
   | --- | ---: | ---: | ---: |
   | GNN seul | 0,5765 | 0,0776 | 0,17 |
   | RF ECFP4 | 0,5971 | 0,1073 | 0,18 |
   | RF ECFP6 | 0,5864 | 0,0949 | 0,13 |
   | RF MACCS | 0,6719 | 0,0878 | 0,12 |
   | RF RDKit | 0,6247 | 0,0563 | 0,08 |
   | A'' ensemble | 0,6201 | **0,1541** | **0,20** |

   L'ensemble améliore la récupération en tête malgré un ROC-AUC inférieur
   à RF MACCS. Cela favorise un objectif d'**enrichissement à budget de test**,
   non une optimisation du ROC-AUC seul. Ce tableau est diagnostique : il ne
   doit pas servir à réajuster les poids sur le même test.
   Les Brier scores bruts des cinq membres (0,039 à 0,140) sont tous moins
   bons que la prédiction constante à la prévalence (0,0217) : leurs sorties
   ne sont donc pas des probabilités fiables. La calibration doit être
   évaluée sur un jeu séparé, sans changer rétroactivement ce test.
4. **Rendre l'incertitude utile.** Recalibrer sur des données indépendantes du
   développement ; vérifier couverture par scaffold et source d'essai. La
   couverture interne de 93,1 % n'établit pas la validité sur une nouvelle
   chimie. Comparer le taux d'abstention et la précision des prédictions
   conservées.
5. **Boucle prospective.** Après confirmation expérimentale, choisir le lot
   suivant par activité estimée **et** diversité/incertitude, puis geler les
   prédictions avant mesure. Documenter chaque version de poids, données,
   seuils et résultats. Une nouvelle version ne remplace A'' qu'après un
   gain prospectif reproductible et une meilleure sélectivité.

La prochaine décision scientifique dépend des résultats indépendants du gate
1 ; réentraîner immédiatement sur les mêmes données binaires sans nouveaux
résultats ne résoudrait pas l'incertitude biologique.

### Audit quantitatif ChEMBL du 10 novembre

La source Jupyter `chembl_hiv_merged.csv` a été copiée localement et auditée par
`python scripts/audit_chembl_source.py` (rapport :
`reports/chembl_source_audit_2026-11.json`). Elle contient **15 842 lignes
toutes positives** et sélectionnées à `pChEMBL ≥ 5` : 12 624 IC50, 2 582 Ki,
564 EC50 et 72 Kd, sur cinq cibles. Elle ne contient ni CC50 ni métadonnées
d'essai cellulaire permettant de déclarer ces 564 EC50 comme une mesure
homogène d'inhibition de la réplication VIH. Les structures brutes recouvrent
exactement 77 molécules du train, cinq de validation et quatre du test ; les
filtres du modèle A'' ont écarté les recouvrements pertinents avant entraînement.
Cette source est utile pour une analyse *par cible*, mais **ne suffit pas**
à entraîner honnêtement un modèle quantitatif d'efficacité cellulaire et de
sélectivité. Elle ne fournit pas non plus de classe inactive mesurée dans le
même contexte. La prochaine donnée décisive reste donc le résultat du
laboratoire sur le lot aveugle et, ensuite, une campagne plus large avec
activité et cytotoxicité appariées.
