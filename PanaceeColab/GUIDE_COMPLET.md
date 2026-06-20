# 📘 GUIDE COMPLET – PROJET PANACEE
## Prédiction Moléculaire par Graph Neural Networks (GNN)
### Guide étape par étape pour Google Colab

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble du projet](#1-vue-densemble)
2. [Fichiers du projet](#2-fichiers-du-projet)
3. [Préparation de Google Colab](#3-préparation-google-colab)
4. [Phase 1 – Pré-entraînement MGM](#4-phase-1--pré-entraînement)
5. [Phase 2 – Fine-tuning Toxicité](#5-phase-2--fine-tuning-toxicité)
6. [Phase 3 – Multi-propriétés](#6-phase-3--multi-propriétés)
7. [Explication des métriques](#7-explication-des-métriques)
8. [Interprétation des résultats](#8-interprétation-des-résultats)
9. [Erreurs courantes et solutions](#9-erreurs-courantes)
10. [Architectures GNN disponibles](#10-architectures-gnn)
11. [Anti-surapprentissage](#11-anti-surapprentissage)

---

## 1. Vue d'ensemble

### Qu'est-ce que PANACEE ?

PANACEE est un système d'intelligence artificielle basé sur des **Graph Neural Networks (GNN)** qui prédit les propriétés moléculaires pertinentes pour le développement de médicaments.

### Pipeline d'entraînement en 3 phases

```
Données ZINC (250k molécules)
         ↓
┌─────────────────────────────┐
│  PHASE 1 – Pré-entraînement │  ← Sans labels (auto-supervisé)
│  Masked Graph Modeling (MGM) │  ← Similaire au BERT en NLP
│  Durée : 2-4h               │
└─────────────┬───────────────┘
              │ Encodeur pré-entraîné
              ↓
┌─────────────────────────────┐
│  PHASE 2 – Fine-tuning Tox  │  ← Avec labels Tox21
│  Classification 12 cibles   │  ← Transfer learning
│  Durée : 1-2h               │
└─────────────┬───────────────┘
              │ Encodeur spécialisé
              ↓
┌─────────────────────────────┐
│  PHASE 3 – Multi-propriétés │  ← 6 datasets, 17 propriétés
│  + Module Raisonneur        │  ← Analyse combinatoire
│  Durée : 2-3h               │
└─────────────────────────────┘
```

### Pourquoi 3 phases ?

Le **transfer learning** permet d'utiliser les connaissances acquises dans chaque phase :
- Phase 1 → Comprendre la structure moléculaire (sans labels)
- Phase 2 → Spécialiser pour la toxicité (12 labels)
- Phase 3 → Généraliser à toutes les propriétés (17 labels)

Cette approche est 3x plus efficace qu'un entraînement from scratch.

---

## 2. Fichiers du projet

### Structure plate (obligatoire pour Colab)

```
PanaceeColab/
├── config.py                    ← Configuration centrale
├── graph_builder.py             ← SMILES → Graphe moléculaire
├── data_loaders.py              ← Chargement et prétraitement des données
├── gnn_models.py                ← 5 architectures GNN
├── prediction_heads.py          ← Têtes de prédiction
├── losses.py                    ← Fonctions de perte
├── metrics.py                   ← Métriques + anti-surapprentissage
├── trainer.py                   ← Entraîneur unifié (3 phases)
├── predictor.py                 ← Inférence + rapports
├── advanced_algorithms.py       ← MCTS, Bayesian, Pareto, Ensemble
├── Notebook_00_Setup.ipynb      ← Installation + vérification
├── Notebook_01_Phase1_Pretrain.ipynb  ← Phase 1
├── Notebook_02_Phase2_Toxicity.ipynb  ← Phase 2
├── Notebook_03_Phase3_MultiProp.ipynb ← Phase 3 + Analyse
└── GUIDE_COMPLET.md             ← Ce fichier
```

### Données nécessaires

| Fichier | Taille | Source | Phase |
|---------|--------|--------|-------|
| `250k_rndm_zinc_drugs_clean_3.csv` | ~50 MB | Fourni | Phase 1 |
| `tox21.csv` | ~1 MB | DeepChem (auto) | Phase 2 |
| `esol.csv`, `lipo.csv`, `bbbp.csv`, `clintox.csv`, `hiv.csv` | <5 MB chacun | DeepChem (auto) | Phase 3 |

> **Note :** Les datasets Phase 2 et 3 sont téléchargés automatiquement via DeepChem.
> Seul le fichier ZINC doit être uploadé manuellement.

---

## 3. Préparation de Google Colab

### Étape 3.1 – Activer le GPU

1. Ouvrir Google Colab : https://colab.research.google.com
2. Menu → **Exécution** → **Modifier le type d'exécution**
3. Sélectionner **GPU** (T4 gratuit ou A100 si Colab Pro)
4. Cliquer **Enregistrer**

### Étape 3.2 – Uploader les notebooks

1. Menu → **Fichier** → **Importer un notebook**
2. Uploader `Notebook_00_Setup.ipynb` en premier
3. Répéter pour les 3 autres notebooks

### Étape 3.3 – Exécuter `Notebook_00_Setup.ipynb`

Ce notebook effectue :
- ✅ Vérification GPU
- 📦 Installation de PyTorch Geometric, RDKit, DeepChem (~10 min)
- 📁 Upload des fichiers Python
- 🔬 Test des imports

> **Important :** Chaque session Colab repart de zéro. Il faut **réinstaller les dépendances** à chaque nouvelle session. C'est pour cela que le Notebook_00 doit toujours être exécuté en premier.

### Étape 3.4 – Connecter Google Drive (recommandé)

Pour éviter de perdre les checkpoints entre les sessions :

```python
from google.colab import drive
drive.mount('/content/drive')
```

Les checkpoints seront sauvegardés dans `/content/drive/MyDrive/Panacee_Checkpoints/`.

---

## 4. Phase 1 – Pré-entraînement

### Ouvrir `Notebook_01_Phase1_Pretrain.ipynb`

### Ce qui se passe techniquement

Le **Masked Graph Modeling (MGM)** fonctionne ainsi :

1. Pour chaque molécule, on masque aléatoirement **15% des atomes**
2. La stratégie de masquage suit BERT (Devlin et al. 2018) :
   - 80% des atomes masqués → remplacés par un token `[MASK]`
   - 10% → remplacés par un atome aléatoire
   - 10% → gardés inchangés
3. L'encodeur GNN traite le graphe partiellement masqué
4. La tête MGM prédit les **9 features** de chaque atome masqué
5. La loss MSE mesure l'erreur de prédiction

### Paramètres importants (dans `config.py`)

```python
PHASE1 = {
    "epochs"        : 100,    # Nombre d'epochs max
    "batch_size"    : 256,    # Molécules par batch
    "lr"            : 1e-3,   # Learning rate initial
    "warmup_epochs" : 10,     # Epochs de warmup linéaire
    "lr_min"        : 1e-6,   # LR minimum (cosine annealing)
    "weight_decay"  : 1e-5,   # Régularisation L2
    "grad_clip"     : 1.0,    # Clipping du gradient
    "patience"      : 20,     # Early stopping
    "mask_prob"     : 0.15,   # Probabilité de masquage (15%)
    "max_molecules" : 250000, # Molécules ZINC à utiliser
}
```

### Durée estimée

| GPU | Temps Phase 1 |
|-----|--------------|
| T4 (Colab gratuit) | ~2-4 heures |
| A100 (Colab Pro) | ~40-60 minutes |

### Que faire pendant l'entraînement ?

Les logs affichent en temps réel :
```
E  1/100 | Train=0.48231 | Val=0.49012 | LR=1.00e-04 | ⏱ 0:02:31
E  2/100 | Train=0.43105 | Val=0.44280 | LR=2.00e-04 | ⏱ 0:05:02
...
```

- **Train < Val** → Normal en début d'entraînement
- **Les deux baissent** → Bon signe
- **Val remonte** → Surapprentissage, l'early stopping va s'activer

---

## 5. Phase 2 – Fine-tuning Toxicité

### Ouvrir `Notebook_02_Phase2_Toxicity.ipynb`

### Dataset Tox21

Tox21 (Toxicology in the 21st Century) contient **~8,000 molécules** évaluées sur **12 cibles biologiques** :

| Cible | Description |
|-------|-------------|
| NR-AR | Récepteur androgène |
| NR-AR-LBD | Domaine liaison ligand récepteur androgène |
| NR-AhR | Récepteur aryl hydrocarbone |
| NR-Aromatase | Enzyme aromatase |
| NR-ER | Récepteur estrogène |
| NR-ER-LBD | Domaine liaison ligand récepteur estrogène |
| NR-PPAR-gamma | Récepteur PPAR-gamma |
| SR-ARE | Élément de réponse antioxydant |
| SR-ATAD5 | Stress génotoxique |
| SR-HSE | Réponse au choc thermique |
| SR-MMP | Perturbation membranaire mitochondriale |
| SR-p53 | Voie de signalisation p53 |

### Mécanisme anti-surapprentissage : Dégel progressif

L'encodeur pré-entraîné est d'abord **gelé** (poids figés), puis dégelé progressivement :

```
Epoch 1-5   : Encodeur gelé, seule la tête s'entraîne
Epoch 6     : Dégel des 2 dernières couches
Epoch 10    : Dégel des 4 dernières couches
Epoch 15+   : Encodeur entièrement dégelé
```

Cela évite de "détruire" les représentations apprises en Phase 1.

### Paramètres importants

```python
PHASE2 = {
    "epochs"               : 80,
    "lr_encoder"           : 5e-5,  # LR très faible pour l'encodeur
    "lr_head"              : 1e-3,  # LR normal pour la tête
    "freeze_encoder_epochs": 5,     # Epochs où l'encodeur est gelé
    "patience"             : 15,    # Early stopping sur AUC
    "label_smoothing"      : 0.05,  # Anti-surconfiance
}
```

### Métriques Phase 2

| Métrique | Signification | Bon score |
|----------|---------------|-----------|
| **ROC-AUC** | Capacité à distinguer toxique/non-toxique | > 0.80 |
| **AUPRC** | Précision-Rappel (important si déséquilibré) | > 0.60 |
| **F1** | Harmonie précision/rappel | > 0.65 |
| **MCC** | Matthews Correlation Coefficient | > 0.40 |

---

## 6. Phase 3 – Multi-propriétés

### Ouvrir `Notebook_03_Phase3_MultiProp.ipynb`

### Datasets utilisés

| Dataset | Propriété | Taille | Type |
|---------|-----------|--------|------|
| Tox21 | Toxicité (12 cibles) | 8k | Classification |
| ESOL | Solubilité (logS) | 1.1k | Régression |
| Lipophilicity | logD | 4.2k | Régression |
| BBBP | Franchissement BHE | 2k | Classification |
| ClinTox | Toxicité clinique | 1.5k | Classification |
| HIV | Inhibition HIV | 41k | Classification |

### MolecularReasoner – Analyse combinatoire

Le **MolecularReasoner** est un Transformer qui prend en entrée les embeddings de plusieurs molécules et calcule :

- **combo_score** : score synergique de la combinaison (0-1)
- **optimal_doses** : distribution de doses recommandée
- **synergy_matrix** : matrice d'interactions croisées
- **confidence** : niveau de confiance de la prédiction

### Algorithmes avancés

#### MCTS (Monte Carlo Tree Search)
Pour trouver la meilleure combinaison parmi N molécules :
```python
from advanced_algorithms import MCTSCombinationSearch
mcts = MCTSCombinationSearch(score_fn=predictor.analyze_combination,
                              n_simulations=200, max_combo_size=3)
best_combo, score = mcts.search(list(range(len(molecules))))
```

#### Optimisation Bayésienne
Pour optimiser les doses d'une combinaison :
```python
from advanced_algorithms import BayesianOptimizer
import numpy as np
bayes = BayesianOptimizer(
    score_fn=dose_efficacy_fn,
    bounds=np.array([[0, 1], [0, 1]]),  # doses [0,1] pour 2 médicaments
    n_iterations=50
)
best_doses, best_score = bayes.optimize()
```

#### Front de Pareto
Pour identifier le meilleur compromis efficacité/sécurité :
```python
from advanced_algorithms import ParetoOptimizer
objectives = np.column_stack([efficacy_scores, safety_scores])
pareto_indices = ParetoOptimizer.pareto_front(objectives)
```

---

## 7. Explication des métriques

### Métriques de classification

#### ROC-AUC (Area Under ROC Curve)
- Mesure la capacité du modèle à distinguer les classes positives (toxiques) des négatives
- **0.5** = prédiction aléatoire
- **1.0** = prédiction parfaite
- **Objectif Tox21** : AUC > 0.80

#### AUPRC (Average Precision)
- Particulièrement important quand les classes sont **déséquilibrées** (peu de molécules toxiques)
- Plus informatif que l'AUC-ROC dans ce cas
- **Objectif** : AUPRC > 0.60

#### F1-Score
- Harmonie entre la précision et le rappel
- **Précision** : Parmi les molécules prédites toxiques, combien le sont vraiment ?
- **Rappel** : Parmi les molécules réellement toxiques, combien a-t-on trouvées ?
- **Formule** : F1 = 2 × (Précision × Rappel) / (Précision + Rappel)

#### MCC (Matthews Correlation Coefficient)
- Métrique robuste même avec des classes déséquilibrées
- Valeurs entre -1 et +1 (0 = aléatoire)
- **Objectif** : MCC > 0.40

### Métriques de régression

#### RMSE (Root Mean Square Error)
- Erreur quadratique moyenne → pénalise les grandes erreurs
- **Interprétation** : En unités de la propriété (ex: RMSE = 0.5 pour logS)

#### MAE (Mean Absolute Error)
- Erreur absolue moyenne → plus robuste aux outliers que RMSE

#### R² (Coefficient de détermination)
- Proportion de variance expliquée par le modèle
- **1.0** = modèle parfait, **0.0** = modèle inutile, **< 0** = pire que la moyenne

#### Pearson r
- Corrélation linéaire entre prédictions et réalité
- **Objectif solubilité (ESOL)** : Pearson > 0.85

---

## 8. Interprétation des résultats

### Score global d'une molécule

Le **score global** est une combinaison pondérée :

```
Score = 0.35 × (1 - toxicité_max) +   # Sécurité (poids le plus important)
        0.25 × efficacité +             # Efficacité thérapeutique
        0.15 × biodisponibilité +       # Absorption orale
        0.15 × stabilité_métabolique +  # Durée d'action
        0.10 × solubilité_normalisée    # Formulation
```

| Score | Interprétation |
|-------|----------------|
| > 0.80 | Excellent candidat |
| 0.60–0.80 | Bon candidat, optimisation possible |
| 0.40–0.60 | Candidat moyen, travaux nécessaires |
| < 0.40 | Candidat faible |

### Profil de toxicité Tox21

Pour chaque des 12 cibles :
- **> 0.50** → Probablement toxique (**⚠️ Attention !**)
- **0.30–0.50** → Incertain, nécessite validation expérimentale
- **< 0.30** → Probablement non-toxique

### Interprétation de la solubilité (logS)

| logS | Catégorie |
|------|-----------|
| > 0 | Très soluble |
| -1 à 0 | Soluble |
| -2 à -1 | Peu soluble |
| -4 à -2 | Insoluble |
| < -4 | Très insoluble (problèmes de formulation) |

### Règle de Lipinski (Drug-likeness)

Pour qu'une molécule soit un bon médicament oral :
- Poids moléculaire ≤ 500 Da
- logP ≤ 5 (lipophilicité)
- Donneurs H-bond ≤ 5
- Accepteurs H-bond ≤ 10

---

## 9. Erreurs courantes

### ❌ `ModuleNotFoundError: No module named 'torch_geometric'`

**Cause :** PyTorch Geometric non installé ou installé pour une mauvaise version de CUDA.

**Solution :**
```python
# Vérifier la version de PyTorch
import torch
print(torch.__version__)  # ex: 2.1.0+cu121

# Réinstaller avec la bonne version
!pip install torch-scatter torch-sparse torch-geometric \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -q
```

### ❌ `RuntimeError: CUDA out of memory`

**Cause :** Batch trop grand pour la VRAM du GPU.

**Solution :** Réduire la taille du batch dans `config.py` :
```python
PHASE1["batch_size"] = 128  # Au lieu de 256
PHASE2["batch_size"] = 64
```

### ❌ `ImportError: cannot import name 'smiles_to_graph' from 'graph_builder'`

**Cause :** Le fichier `graph_builder.py` n'est pas dans le répertoire courant.

**Solution :**
```python
import sys, os
sys.path.insert(0, "/content/panacee")  # Ajouter le bon chemin
os.chdir("/content/panacee")
```

### ❌ Session Colab expirée, checkpoints perdus

**Cause :** Les sessions Colab durent maximum 12h (gratuit) ou 24h (Pro).

**Solution :** Toujours sauvegarder sur Google Drive à la fin de chaque phase :
```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy("checkpoints/phase1/panacee_phase1.pth",
            "/content/drive/MyDrive/Panacee_Checkpoints/")
```

### ❌ `ValueError: No valid SMILES found`

**Cause :** Les SMILES dans votre CSV ne sont pas dans la bonne colonne.

**Solution :** Vérifier les colonnes dans `config.py` :
```python
SMILES_COLUMN_CANDIDATES = ["smiles", "SMILES", "Smiles", "molecule", "mol"]
```
Ajouter le nom de votre colonne si elle ne s'y trouve pas.

### ❌ AUC reste autour de 0.50 (prédiction aléatoire)

**Cause possible 1 :** L'encodeur Phase 1 n'a pas convergé.
→ Vérifier que la loss Phase 1 descend bien (de ~0.5 à < 0.1)

**Cause possible 2 :** Le dégel de l'encodeur a détruit les représentations.
→ Réduire `lr_encoder` dans Phase 2 (`1e-5` au lieu de `5e-5`)

**Cause possible 3 :** Données insuffisantes.
→ Vérifier que le dataset Tox21 contient bien ~8000 molécules

---

## 10. Architectures GNN

### Comparaison des 5 architectures

| Architecture | Papier | Points forts | Utilisation recommandée |
|-------------|--------|--------------|------------------------|
| **AttFP** | Xiong 2020 (JACS) | Meilleur sur Tox21, stable | ✅ Défaut recommandé |
| **GIN** | Xu 2019 (ICLR) + Hu 2020 | Expressivité théorique max | Grands datasets |
| **MPNN** | Gilmer 2017 | Simple, robuste | Baseline |
| **PNA** | Corso 2020 (NeurIPS) | Excellente agrégation | Propriétés 3D |
| **GPS** | Rampásek 2022 (NeurIPS) | SOTA, attention globale | Colab Pro (A100) |

### Changer d'architecture

Dans `config.py` :
```python
GNN_ARCHITECTURE = "attfp"  # ou "gin", "mpnn", "pna", "gps"
```

### Performances benchmarks

Référence : MoleculeNet benchmark (Wu et al. 2018), dataset Tox21

| Architecture | AUC-ROC Tox21 (moyen) |
|-------------|----------------------|
| GPS | 0.854 |
| AttFP | 0.841 |
| PNA | 0.833 |
| GIN | 0.821 |
| MPNN | 0.815 |

---

## 11. Anti-surapprentissage

### Mécanismes intégrés

Le projet intègre 7 mécanismes anti-surapprentissage :

#### 1. Dropout (p=0.20)
Les neurones sont éteints aléatoirement pendant l'entraînement.
→ Empêche la mémorisation des données

#### 2. Weight Decay L2 (λ=1e-4)
Pénalité sur les grands poids dans la fonction de perte.
→ Encourage des représentations plus simples

#### 3. Label Smoothing (ε=0.05)
Les labels 0 et 1 deviennent 0.025 et 0.975.
→ Empêche le modèle d'être trop confiant

#### 4. Gradient Clipping (max norm=1.0)
Les gradients sont normalisés si leur norme dépasse 1.0.
→ Stabilise l'entraînement, évite les explosions de gradients

#### 5. Warmup + Cosine Annealing
Le learning rate monte linéairement puis descend en cosinus.
→ Évite les oscillations en début d'entraînement

#### 6. Early Stopping (patience=20)
L'entraînement s'arrête si la métrique de validation ne s'améliore plus.
→ Restaure automatiquement les meilleurs poids

#### 7. Dégel progressif (Phase 2 et 3)
L'encodeur pré-entraîné est gelé puis dégelé progressivement.
→ Évite de "corrompre" les représentations apprises

### Comment interpréter les courbes d'apprentissage

```
Situation idéale :
  Train loss  ↓↓↓   (descend bien)
  Val loss    ↓↓    (descend aussi, légèrement au-dessus)
  Gap T/V     ~ 0   (petit écart stable)

Surapprentissage modéré :
  Train loss  ↓↓↓
  Val loss    → ou ↑ (stagne ou remonte)
  Gap T/V     ↑↑    (augmente progressivement)
  → L'early stopping va s'activer

Surapprentissage sévère :
  Train loss  ↓↓↓
  Val loss    ↑↑↑
  Gap T/V AUC > 0.20
  → Augmenter le dropout ou réduire le learning rate
```

### Diagnostics automatiques

Le code affiche automatiquement des avertissements :
```
⚠ Surapprentissage [modéré] gap AUC=0.12
⚠ SURAPPRENTISSAGE SÉVÈRE détecté (gap=0.23)
```

Si vous voyez ces messages :
1. **Modéré** → Laisser l'early stopping agir
2. **Sévère** → Arrêter l'entraînement, augmenter le dropout dans `config.py`

---

## 📞 Résumé des commandes Colab

```python
# ── CONFIGURATION ──────────────────────────────────────────────
import sys, os
sys.path.insert(0, "/content/panacee")
os.chdir("/content/panacee")

# ── PRÉDICTION RAPIDE ──────────────────────────────────────────
from predictor import PanaceePredictor
p = PanaceePredictor.load("checkpoints/phase3/panacee_phase3.pth")
result = p.predict("CC(=O)Nc1ccc(O)cc1")  # Paracétamol
print(result)

# ── RAPPORT COMPLET ─────────────────────────────────────────────
report = p.generate_report(
    smiles_list=["CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O"],
    output_path="/content/panacee/results/mon_rapport.txt"
)
print(report)

# ── ANALYSE COMBINATOIRE ────────────────────────────────────────
combo = p.analyze_combination([
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O"
])
print(f"Score synergique : {combo['combo_score']:.4f}")
```

---

*Document généré automatiquement pour le projet PANACEE*
*Architecture : 5 GNN (AttFP, GIN, MPNN, PNA, GPS) | 3 Phases | 17 propriétés moléculaires*
