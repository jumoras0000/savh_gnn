# Phase 4 — recherche anti-VIH exploratoire

La Phase 4 produit un dossier de **priorisation de recherche**, pas un remède, un diagnostic, une prescription ou un protocole de fabrication. Aucun score du GNN, avis de Claude ou titre PubMed ne démontre une efficacité, une innocuité ou une fiabilité proche de 100 %.

Le pourcentage de fiabilité du modèle est **indéterminé** tant qu'un checkpoint identifié et des résultats sur des données externes indépendantes ne sont pas fournis. Une absence de faux négatifs observés n'est pas une garantie de risque nul.

## Fonctionnement vérifiable

1. L'utilisateur ou Claude propose 1 à 5 molécules identifiées par SMILES, éventuellement issues du catalogue de l'application.
2. Le moteur existant calcule les descripteurs et, si un checkpoint Phase 3 est effectivement chargé, le score d'activité sur le jeu VIH. Sans ce checkpoint, le score est absent ; il n'est pas inventé.
   Le dossier inclut l'empreinte SHA-256 du checkpoint lorsqu'un fichier correspondant est accessible ; « non disponible » n'est pas une preuve de provenance.
3. Pour chaque molécule nommée, NCBI ESearch/ESummary retourne au plus trois **notices** PubMed avec PMID, titre, date et URL. Un résultat de recherche n'est ni une lecture critique de l'article ni une preuve de lien causal avec la molécule.
4. Claude peut rédiger une interprétation séparée, marquée `generated_unverified`. Les PMID explicitement cités mais absents des notices sont signalés. Le contenu de son interprétation reste à vérifier indépendamment.
   Si les deux fournisseurs sont configurés, la chaîne bornée est : modèle → analyses initiales indépendantes Claude et ChatGPT sur les mêmes données → critique croisée Claude → critique croisée ChatGPT (quatre appels IA maximum). Les premières requêtes ne contiennent aucun texte de l'autre IA ; les critiques croisées reçoivent les analyses antérieures, toujours marquées non vérifiées. Chaque texte reste non vérifié et ne peut modifier les scores du modèle. Le contrôle « modèle » signale automatiquement l'absence de checkpoint ou de validation ; le GNN ne comprend pas ni ne critique directement les affirmations en prose. Les hypothèses fournies par l'utilisateur reçoivent des identifiants H1…H10 ; l'absence de mention d'un identifiant est signalée, mais sa présence ne prouve pas une analyse complète.
5. Le rapport affiche les données manquantes et le parcours de validation requis. Les fonctions de synergie et de dose de la Phase 3 reposent sur des cibles dérivées du modèle, pas sur des mesures expérimentales de combinaison ; elles ne servent pas à prescrire.

L'API web de combinaison masque désormais les sorties de dose et remplace l'interprétation par un avertissement. Les scores de synergie restent des proxys non validés. Des scripts internes historiques peuvent encore contenir les anciens libellés : ils ne doivent pas être utilisés comme recommandations de traitement.

## Intégrité des partitions Phase 3

`python run_phase3.py --download` repartitionne les six jeux avec une attribution **globale** par scaffold et identité SMILES canonique, puis refuse toute fuite détectée. Un manifeste `split_manifest.json` consigne les comptes et empreintes SHA-256 des partitions produites. Les données déjà présentes sont également auditées au démarrage ; si leurs partitions anciennes se chevauchent, il faut régénérer via `--download` avant d'entraîner. La canonicalisation ne règle pas toutes les ambiguïtés de mesure ou de standardisation chimique ; une provenance indépendante reste nécessaire.

## Accès

Lancer l'application comme indiqué dans `webapp/README.md`, puis ouvrir l'onglet « Phase 4 ». L'API `POST /api/phase4` accepte par exemple :

```json
{"candidates":[{"name":"Molécule étudiée","smiles":"CCO"}],"with_claude":false,"literature":true}
```

`with_claude=true` requiert l'intégration Claude déjà configurée. Les requêtes NCBI requièrent une connexion réseau. Une erreur réseau est signalée ; aucune source n'est fabriquée.

`with_chatgpt=true` nécessite le SDK `openai`, `OPENAI_API_KEY` et `OPENAI_MODEL` (un identifiant de modèle auquel **ton projet API** a accès). La connexion à ChatGPT dans le navigateur ne fournit pas automatiquement une clé API au serveur. L'API `GET /api/phase4/status` montre la configuration des composants sans afficher les secrets ; elle ne teste pas les crédits. Une requête peut inclure `"hypotheses":["question 1","question 2"]` (maximum 10). Le rapport conserve séparément les quatre revues et leurs statuts. En l'absence de clé, de crédit ou en cas d'erreur, la revue correspondante est explicitement indisponible ; aucun texte n'est simulé.

### Mise en service

1. Synchroniser le dépôt et installer les dépendances du projet. Pour la revue Claude : `python -m pip install anthropic`. Pour la revue ChatGPT facultative : `python -m pip install openai`.
2. Configurer la clé Anthropic dans l'onglet Assistant ou via `ANTHROPIC_API_KEY` ; un **crédit API valide** est indispensable. Une clé présente mais sans crédit ne permet pas l'appel.
3. Pour le contre-examen ChatGPT, configurer séparément `OPENAI_API_KEY` et `OPENAI_MODEL` dans l'environnement du processus web, puis redémarrer le serveur. Le modèle choisi doit être accessible au compte API concerné ; aucun modèle n'est deviné par l'application.
4. Copier le checkpoint Phase 3 entraîné dans le chemin attendu par la configuration du projet. Sans lui, le dossier fonctionne avec les descripteurs, **sans score VIH**.
5. Démarrer `python -m webapp.run`, ouvrir `http://127.0.0.1:8000`, vérifier le bandeau de l'onglet Phase 4, puis lancer un dossier. Les erreurs de fournisseur et les étapes manquantes apparaissent dans le dossier ; le bandeau seul ne prouve pas les crédits ni la validité d'un checkpoint.

Ne transmettre aucune clé API dans un chat public, un rapport ou un commit Git. Un abonnement à l'interface ChatGPT n'active pas automatiquement l'API du projet.

## Audit des faux positifs et faux négatifs

Le script `run_phase4_audit.py` prend un CSV de prédictions **déjà figées** contenant `smiles,y_true,predicted_probability,split`, ainsi qu'un endpoint et un seuil choisi **avant** l'examen du test :

```bash
python run_phase4_audit.py --csv test_externe.csv --threshold 0.5 --endpoint HIV_active
```

Il calcule TP/FN/TN/FP, sensibilité, spécificité, taux de faux négatifs/positifs et intervalles de confiance de Wilson à 95 %. Il refuse les probabilités NaN et les doublons de texte SMILES. Avec RDKit disponible, il vérifie aussi l'identité canonique et les chevauchements de scaffold entre les lignes **présentes dans le CSV** ; les champs du rapport indiquent si ces contrôles ont réellement été exécutés. Il **ne vérifie pas** la provenance externe du CSV ni que toutes les données d'entraînement y figurent. Même zéro faux négatif observé n'établit pas un risque nul.

## Vérifications encore nécessaires

- Enregistrer version, empreinte et provenance de chaque dataset et checkpoint, puis vérifier les splits par structure canonique et scaffold.
- Exécuter une évaluation aveugle sur un jeu externe indépendant, mesurer calibration, PR-AUC et performances par sous-groupes, avec intervalles de confiance.
- Lire les articles complets pertinents et vérifier endpoint, contexte expérimental et qualité des preuves ; ne pas déduire une conclusion scientifique des métadonnées.
- Faire confirmer activité antivirale, cytotoxicité, sélectivité, résistance et interactions par des laboratoires indépendants ; puis suivre les étapes précliniques, cliniques et réglementaires applicables.

Références : [NIH, recommandations VIH](https://clinicalinfo.hiv.gov/en/guidelines/hiv-clinical-guidelines-adult-and-adolescent-arv) ; [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) ; [FDA, cadre de crédibilité IA (projet)](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological).
