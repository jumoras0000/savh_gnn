# Panacée vNext — protocole de données et de comparaison

## Décision au 10 novembre 2026

A'' (6 novembre) demeure l'unique modèle de classement VIH en production.
Le test interne MoleculeNet a déjà servi au choix de l'ensemble et ne peut
plus arbitrer une nouvelle version. Le lot aveugle de 26 structures est gelé ;
les 18 candidats non témoins constituent une cohorte enrichie, pas un
échantillon représentatif d'une bibliothèque. Aucun résultat cellulaire
indépendant n'a encore été fourni. Il n'existe donc pas de poids vNext à
entraîner ou à promouvoir honnêtement aujourd'hui.

## 1. Données à acquérir et à protéger

Le modèle suivant vise en priorité **l'activité VIH cellulaire et la
sélectivité** mesurées dans un contexte d'essai défini, plutôt qu'un label
`HIV_active` hétérogène. Les données EC50/CC50 appariées doivent préserver
les réplicats, unités, censures (`>10`), laboratoire, campagne, identifiant
d'essai et date. Les molécules inactives **mesurées dans le même essai** sont
indispensables pour estimer les faux positifs. La seule source ChEMBL locale
contient 15 842 entrées toutes positives, plusieurs endpoints et aucune CC50 ;
elle ne remplit pas ce contrat.

Format d'entrée vNext :

```csv
molecule_id,smiles,lab_id,campaign_id,assay_id,replicate_id,measured_on,ec50_um,cc50_um
```

Une ligne représente une paire de mesures en µM ; `measured_on` est une date
ISO `AAAA-MM-JJ`. Une valeur censurée est saisie comme `>nombre`, jamais
remplacée par la borne comme si elle était exacte. Un rapport de qualité du
laboratoire doit accompagner le CSV. Les résultats du lot de 26 ne deviennent
pas des exemples d'entraînement pour la version A'' qui les a sélectionnés.

Audit local, sans entraînement :

```powershell
python scripts/audit_vnext_dataset.py chemin/mesures_appariees.csv
```

Le rapport SHA-256 est enregistré sous `data/vnext_audit/` (hors Git). L'outil
contrôle schéma, valeurs, identité chimique, réplicats, campagnes et chevauchement
exact/de scaffold d'une **proposition** de découpe chronologique par campagnes.
`split_ready=true` signifie seulement que cette découpe préliminaire ne révèle
pas de chevauchement ; cela ne certifie ni la qualité biologique ni la taille
adéquate du jeu. Les sels, tautomères, formes stéréochimiques, séries proches
et changements de protocole requièrent encore une revue chimique et
expérimentale avant gel d'un split définitif.

## 2. Évaluation prospective déjà codée

Après un import complet et cohérent du lot aveugle, `evaluate_assay` ajoute
`prospective_evaluation` au rapport local : bilan séparé par provenance
(ZINC/repurposing), puis nombre de signaux aux budgets 5, 10 et 18 selon le
rang A'' **gelé**. Les valeurs censurées qui empêchent une décision sont
`indeterminate`, jamais assimilées à des échecs. Des bornes descriptives
inférieure/supérieure de la fraction de signaux sont affichées. Ce n'est pas
une précision externe générale, car le lot a été présélectionné par le modèle.
Le CSV source, le manifeste et les prédictions sont liés par SHA-256 ; une
réception partielle ou incohérente ne décode pas la clé.

## 3. Comparaison à mener dès que des données indépendantes existent

1. Geler une question de prédiction, l'endpoint, le budget de molécules
   testables et les critères d'exclusion **avant** d'observer les labels de
   validation. Ne pas mélanger IC50 enzymatique, Ki et EC50 cellulaire dans
   une unique cible numérique.
2. Dédupliquer les structures et geler train/validation/test par campagne et
   temps, avec audit des identités et scaffolds. Garder un test prospectif
   totalement inédit ; le test MoleculeNet du 6 novembre reste descriptif.
3. Comparer à même données et même budget : RF ECFP4 simple, A'' gelé,
   régression quantitative EC50, puis multitâche EC50/CC50 si la couverture
   le permet. Ne conserver un GNN plus complexe que s'il améliore une
   validation indépendante répétée. L'audit des membres actuels donne
   PRC-AUC 0,0776 au GNN seul contre 0,1541 à l'ensemble sur le test déjà
   utilisé ; ce constat ne permet pas de régler de nouveaux poids.
4. Rapporter en priorité la récupération et la fraction de signaux parmi
   les `k` premières molécules, PRC-AUC avec prévalence, calibration sur
   **validation distincte**, couverture et performance dans/hors domaine.
   Ajouter les intervalles d'incertitude appropriés aux campagnes, plutôt
   qu'une seule valeur ponctuelle. Les scores A'' actuels ne doivent pas être
   interprétés comme des probabilités.
5. Verrouiller version du code, poids, données, splits, règles de censure et
   seuils. Une promotion exige un gain prospectif reproductible et un profil
   de sélectivité acceptable, pas une optimisation répétée du test historique.

## 4. Après un signal cellulaire

Un signal confirmé impose encore des données indépendantes sur mécanisme,
résistance et résistance croisée, activité en présence de sérum, exposition,
toxicité, interactions, identité/pureté et stabilité. La [FDA, informations
pré-IND antiviraux](https://www.fda.gov/about-fda/cder-offices-and-divisions/division-antivirals-dav-information-pre-ind-submissions)
distingue ces domaines de preuve. Le domaine d'applicabilité est également un
élément formel de l'[évaluation (Q)SAR de l'OCDE](https://www.oecd.org/en/publications/q-sar-assessment-framework-guidance-for-the-regulatory-assessment-of-quantitative-structure-activity-relationship-models-and-predictions_d96118f6-en.html).
