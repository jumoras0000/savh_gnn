# Curation des données Phase 3

Les CSV MoleculeNet bruts sont conservés sous `data/external/phase3/<jeu>/`. Le
fichier `data/external/phase3/split_manifest.json` contient leurs empreintes
SHA-256, celles des 18 partitions et les observations contradictoires. Chaque
observation en quarantaine comporte le SMILES original, la ligne du CSV brut,
l'identifiant source quand il existe et toutes les étiquettes disponibles.

Une contradiction n'est **pas** une étiquette corrigée. Le composé concerné
reste hors des partitions de ce jeu jusqu'à une curation documentée :

1. Retrouver la ligne et l'identifiant dans le CSV brut, puis la source primaire
   et la définition exacte de la mesure ou de la classe.
2. Vérifier l'identité chimique, notamment stéréochimie, sels et tautomères.
   Un SMILES sans stéréochimie ne permet pas de distinguer deux stéréoisomères.
3. Si une valeur est démontrée erronée, documenter la preuve et produire une
   **nouvelle version** du jeu de données ; ne pas modifier silencieusement la
   copie brute ni choisir par majorité ou moyenne.
4. Régénérer les partitions avec `--download`, vérifier le nouveau manifeste,
   réentraîner la Phase 3 depuis la Phase 2 dans un nouveau dossier de
   checkpoints, puis réaliser une évaluation externe indépendante.

La vérification au démarrage compare les partitions au manifeste et chaque
observation en quarantaine à sa ligne brute. Cette traçabilité prouve la
cohérence des fichiers, **pas** l'exactitude expérimentale ou clinique des
étiquettes. Les scores du modèle ne constituent pas des preuves de sécurité,
d'efficacité ou de guérison.
