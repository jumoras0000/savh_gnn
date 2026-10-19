# Audit des Phases 1 et 2 et correction du diagnostic Phase 3

## Constat, preuve et degré de certitude

| Constat | Preuve dans le code | Certitude |
|---|---|---|
| Phase 2 et Phase 3 produisaient des partitions Tox21 indépendantes ; le seul audit interne Phase 3 ne pouvait exclure une exposition supervisée en Phase 2. | `toxicity_loader._scaffold_split`, `multi_property_loader.repartition_global_scaffolds`, ancien checkpoint Phase 2 sans liste de composés | Élevée pour le risque structurel ; **chevauchement réel du vieux run non vérifié** faute de ses CSV/checkpoint sur cette machine |
| Phase 2 pouvait choisir une époque avant le dégel de son encodeur (10 époques par défaut), puis transmettre celui-ci en Phase 3. | `finetune_toxicity.train_one_run`, ancien `is_best` sans condition de dégel | Élevée pour le défaut ; époque du checkpoint Phase 2 de l'utilisateur inconnue |
| Le warm-start Phase 2 pouvait réutiliser des poids d'un autre jeu, et un checkpoint Phase 1 incomplet pouvait être chargé partiellement. | Ancien `load_state_dict(..., strict=False)` et absence de provenance | Élevée |
| Le masque de validation Phase 1 était tiré à nouveau à chaque évaluation ; sa perte de sélection était donc bruitée. | Ancien `PretrainDataset.__getitem__` appelait `mask_atoms` sans générateur fixe | Élevée ; ampleur de l'effet sur le modèle inconnu |
| `max_molecules` sélectionnait la tête du fichier avant mélange. | Anciennes troncatures Phase 1/2 | Élevée pour le biais potentiel ; ordre réel des données inconnu |

## Correctifs

- La Phase 1 fixe le masque **uniquement en validation**, mélange avant tout sous-échantillonnage et refuse les paramètres/partitions vides. Son entraînement garde des masques renouvelés.
- La Phase 2 vérifie les poids de Phase 1, valide les étiquettes binaires, refuse un objectif de validation sans classe positive et négative, archive les anciens checkpoints et ne retient que les époques post-dégel. Le warm-start exige des tâches et une exposition identiques.
- Un nouveau checkpoint Phase 2 conserve les composés canoniques et scaffolds utilisés pour entraînement **et** sélection. La Phase 3 rejette tout chevauchement connu avec ses jeux val/test ; un vieux checkpoint sans provenance reste marqué `unverified` et **ne doit pas être présenté comme indépendant**.
- `run_phase2.py --phase3_data_dir ...` crée un train/val Phase 2 uniquement à partir du train Tox21 Phase 3 vérifié, avec séparation interne par scaffold. Cette voie ne modifie pas les partitions Phase 3.

## Verdict sur les checkpoints existants

Les métriques du checkpoint Phase 3 terminé sont descriptives pour ce run, mais **leur indépendance vis-à-vis de la Phase 2 n'est pas démontrée**. Le code ne peut pas reconstruire rétroactivement quels composés ont servi à entraîner ou sélectionner l'ancien checkpoint Phase 2. Ce constat ne prouve pas une fuite effective ; il interdit de prétendre qu'elle est exclue. Le pré-entraînement auto-supervisé de Phase 1 peut aussi avoir vu des structures évaluées plus tard ; ce n'est pas une exposition aux étiquettes, mais cela doit être déclaré dans un protocole de généralisation.

Pour une nouvelle évaluation avec provenance contrôlée, la Phase 1 déjà entraînée peut être réutilisée ; **repartir de la Phase 2**, puis réentraîner la Phase 3 avec un nouveau dossier et `--warm_start 0`. Exemple sur la machine GPU, après synchronisation du code :

```bash
cd ~/panacee/Projet_Panacee
python run_phase2.py --phase3_data_dir data/external/phase3 --save_dir checkpoints/phase2_aligned --warm_start 0 --device cuda:0
python run_phase3.py --pretrained_model checkpoints/phase2_aligned/best_toxicity_model.pth --run_name phase3_upstream_audite --warm_start 0 --selection_policy multitask --device cuda:0
```

Ces commandes n'ont pas été exécutées sur l'A10 ici. Le test local emploie de minuscules données synthétiques. Un jeu **externe indépendant** reste nécessaire : le jeu test actuel a déjà servi à orienter l'analyse. Aucune amélioration de score, innocuité ni guérison n'est garantie.

Validation locale : mini-entraînements complets Phase 1 et Phase 2 sur CPU,
tests d'absence de chevauchement connu, de rejet de warm-start incohérent et de
masques de validation répétables. Ces essais ne mesurent pas les performances
sur les vrais jeux ni la stabilité CUDA/AMP.

Références : [MoleculeNet, Wu et al.](https://doi.org/10.1039/C7SC02664A) pour les tâches et découpages ; [projet de cadre FDA](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological) pour l'exigence d'évaluer un modèle selon un contexte d'usage précis.
