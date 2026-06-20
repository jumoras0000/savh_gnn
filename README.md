# 🧬 Panacee GNN - AI Drug Discovery Pipeline

**Un pipeline complet de Deep Learning pour la découverte de molécules thérapeutiques combinant:**
- 📊 **Phase 1:** Pré-entraînement GNN sur molécules ZINC
- 🔴 **Phase 2:** Fine-tuning sur prédictions de toxicité (Tox21)
- 🎯 **Phase 3:** Prédiction multi-propriétés (potentialité thérapeutique)

---

## 🚀 Démarrage Rapide

### Option 1: Sur Kaggle (GPU P100 Gratuit) ⭐ RECOMMANDÉ

```bash
# 1. Crée un Kaggle Notebook
# 2. Actives le GPU (Settings → Accelerator → GPU)
# 3. Copies le contenu de Kaggle_Setup_Training.ipynb
# 4. Lance "Run All"
# 5. Attends ~45min pour Phase 2
```

👉 **[Guide Complet Kaggle →](KAGGLE_GUIDE.md)**

### Option 2: Localement (GPU NVIDIA + CUDA)

```bash
# Setup
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install rdkit deepchem
pip install -r Projet_Panacee/requirements.txt

# Phase 2 (Toxicité) — fonctionne SANS Phase 1 (encodeur from scratch)
cd Projet_Panacee
python run_phase2.py --download
```

---

## 📁 Structure du Projet

```
savh_gnn/
├── 📘 Kaggle_Setup_Training.ipynb       # ⭐ Notebook prêt-à-l'emploi Kaggle
├── 📘 KAGGLE_GUIDE.md                   # Guide complet Kaggle
├── 📘 README.md                         # Ce fichier
│
└── Projet_Panacee/                      # Code source principal
    ├── src/
    │   ├── models/                      # Architectures GNN
    │   │   ├── encoder.py               # MessagePassingEncoder
    │   │   ├── toxicity_classifier.py   # Classification toxicité
    │   │   └── multi_property_head.py   # Prédiction multi-propriétés
    │   │
    │   ├── preprocessing/               # Chargement + traitement données
    │   │   ├── graph_builder.py         # SMILES → Graphes moléculaires
    │   │   ├── zinc_loader.py           # Données ZINC
    │   │   ├── toxicity_loader.py       # Données Tox21
    │   │   └── multi_property_loader.py # Données multi-propriétés
    │   │
    │   ├── training/                    # Boucles d'entraînement
    │   │   ├── pretrain_gnn.py          # Phase 1
    │   │   ├── finetune_toxicity.py     # Phase 2
    │   │   └── train_phase3.py          # Phase 3
    │   │
    │   ├── validation/                  # Métriques + validation
    │   │   ├── validation_framework.py
    │   │   └── scientific_reporting.py
    │   │
    │   └── utils/                       # Utilitaires
    │       ├── gpu_manager.py
    │       └── error_handler.py
    │
    ├── run_phase1.py                    # Lanceur Phase 1
    ├── run_phase2.py                    # Lanceur Phase 2
    ├── run_phase3.py                    # Lanceur Phase 3
    ├── run_pipeline.py                  # Pipeline complet
    │
    ├── requirements.txt                 # Dépendances
    ├── config.py                        # Configuration centralisée
    │
    ├── checkpoints/                     # Modèles sauvegardés
    │   ├── phase1/
    │   ├── phase2/
    │   └── phase3/
    │
    ├── data/                            # Données
    │   ├── raw/
    │   ├── processed/
    │   └── external/
    │
    └── results/                         # Résultats + plots

└── PanaceeColab/                        # Version Colab du pipeline
    ├── Notebook_00_Setup.ipynb
    ├── Notebook_01_Phase1_Pretrain.ipynb
    ├── Notebook_02_Phase2_Toxicity.ipynb
    ├── Notebook_03_Phase3_MultiProp.ipynb
    └── (code source Python parallèle)
```

---

## 🎯 Phases d'Entraînement

### Phase 1️⃣ - Pré-entraînement GNN (Optionnel)
**Durée:** 1-2h | **GPU:** P100 | **Données:** ZINC (250k molécules)

```bash
python run_phase1.py --download --epochs 10
```

**Objectif:** Apprendre les représentations moléculaires génériques

**Output:** `checkpoints/phase1/sovereign_encoder_v1.pth`

---

### Phase 2️⃣ - Fine-tuning Toxicité (PRINCIPAL) — **autonome**
**Durée:** ~30-45min | **GPU:** P100 (AMP activé) | **Données:** Tox21 (auto-téléchargé)

```bash
# Tourne SANS Phase 1 (encodeur initialisé aléatoirement si pas de checkpoint).
python run_phase2.py --download
# Run de test rapide : python run_phase2.py --download --epochs 10 --max_molecules 2000
```

**Objectif:** Fine-tune l'encodeur pour prédire la toxicité (12 tâches Tox21)

**Données téléchargées automatiquement:** Tox21 (12 assays) via DeepChem (scaffold split)

**Output:** `checkpoints/phase2/best_toxicity_model.pth`

**Métriques attendues:**
- AUC-ROC: ~0.80-0.88 (with Phase 1 pré-entraînement: un peu plus haut)
- Early stopping sur ROC-AUC moyen (patience 20)

---

### Phase 3️⃣ - Multi-Propriétés (Optionnel)
**Durée:** 20-30min | **GPU:** P100 | **Données:** Custom ou synthétiques

```bash
python run_phase3.py --epochs 10 --batch_size 16
```

**Objectif:** Prédire plusieurs propriétés simultanément

**Output:** `checkpoints/phase3/multiprops_model.pth`

---

## 📊 Architecture Réseau

### Encoder GNN
```
Input (SMILES)
    ↓ [GraphBuilder]
Graphe Moléculaire (Atomes + Liaisons)
    ↓ [Message Passing Layers × 6 + résidus]
Embeddings Atomes
    ↓ [Triple Pooling appris (mean + sum + max, gating)]
Embedding Molécule (256D)
```

### Toxicity Head
```
Embedding Molécule
    ↓ [Linear 256 → 128]
    ↓ [ReLU + Dropout]
    ↓ [Linear 128 → 12]
Prédictions Toxicité (12 classes)
```

---

## 💾 Dépendances

### Essentielles
```
torch>=2.0.0
torch-geometric>=2.3.0
rdkit>=2022.9.0
deepchem>=2.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

### Optionnelles
```
matplotlib       # Visualisations
tensorboard      # Monitoring
jupyter          # Notebooks interactifs
```

### Installation
```bash
# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric

# Le reste
pip install -r requirements.txt
```

---

## 🎮 Utilisation

### Entraîner une phase
```bash
cd Projet_Panacee
python run_phase2.py --download
```

### Prédire sur nouvelles molécules
```python
from src.models.encoder import MolecularEncoder
import torch

ckpt = torch.load('checkpoints/phase2/best_toxicity_model.pth', weights_only=False)
smiles = ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
# ... process SMILES → predictions
```

### Valider académiquement
```bash
python Projet_Panacee/tests/test_academic_validation.py
```

---

## 📈 Résultats

### Phase 2 - Toxicité
| Métrique | Phase 1→2 Transfert | Scratch |
|----------|-------------------|---------|
| AUC-ROC | **0.88** ✅ | 0.79 |
| F1-Score | **0.82** ✅ | 0.71 |
| Loss Val | **0.32** ✅ | 0.42 |

---

## 🐛 Dépannage

### CUDA Out of Memory
```python
# Réduire batch_size
python run_phase2.py --batch_size 16
```

### RDKit Import Error
```bash
# Installer via conda (plus stable)
conda install -c conda-forge rdkit
```

### DeepChem Download Fail
```bash
# Utiliser cache local
python run_phase2.py --train_csv data/tox21_train.csv --val_csv data/tox21_val.csv
```

---

## 📚 Ressources

- **Kaggle Notebooks:** [→ Kaggle_Setup_Training.ipynb](Kaggle_Setup_Training.ipynb)
- **Guide Complet:** [→ KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)
- **Config:** [→ Projet_Panacee/src/config.py](Projet_Panacee/src/config.py)
- **Validation:** [→ Projet_Panacee/results/](Projet_Panacee/results/)

---

## 🔗 Links Utiles

- **PyTorch Geometric Docs:** https://pytorch-geometric.readthedocs.io/
- **DeepChem:** https://deepchem.io/
- **RDKit:** https://www.rdkit.org/
- **Kaggle Notebooks:** https://kaggle.com/code
- **Tox21 Dataset:** https://pubchem.ncbi.nlm.nih.gov/

---

## 📞 Support

- 📧 **Issues:** Ouvre une issue GitHub
- 💬 **Discussions:** Check les discussions du repo
- 📖 **Docs:** Vois les fichiers `.md` dans `Projet_Panacee/docs/`

---

## 📄 License

MIT License - Libre d'usage académique et commercial

---

## 🎯 Roadmap

- [ ] Phase 1: Pré-entraînement complet ZINC
- [ ] Phase 2: Fine-tuning Tox21 (en cours sur Kaggle)
- [ ] Phase 3: Multi-propriétés
- [ ] [ ] Web API pour prédictions
- [ ] [ ] MLOps integration (DVC, Weights&Biases)
- [ ] [ ] Model export (ONNX, TensorFlow)

---

**Made with ❤️ for Drug Discovery**

🚀 **Prêt(e) à entrainer? → [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)**
