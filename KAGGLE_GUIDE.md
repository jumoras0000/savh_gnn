# 🚀 Guide Complet : Entraîner Panacee GNN sur Kaggle

## 📋 Vue d'ensemble

Ce guide t'explique comment utiliser le **GPU P100 gratuit de Kaggle** pour entraîner ton modèle GNN Panacee sans configuration locale complexe.

---

## ✅ Prérequis

- ✅ Compte Kaggle (gratuit)
- ✅ GPU activé dans les settings
- ✅ C'est tout !

---

## 🎯 Étapes (5 minutes)

### 1️⃣ Créer un Kaggle Notebook

1. Va sur **[kaggle.com](https://kaggle.com)**
2. Login avec ton compte
3. **"Create"** → **"Notebook"**
4. Choisis **"Python"**

### 2️⃣ Activer le GPU

1. **⚙️ Settings** (coin haut-droit)
2. **"Accelerator"** → Sélectionne **"GPU"** (P100)
3. **"Save"**

### 3️⃣ Copier le Notebook

**Option A : Depuis GitHub (recommandé)**
```python
# Cell 1
!git clone https://github.com/jumoras0000/savh_gnn.git
```

Puis copie le code du fichier `Kaggle_Setup_Training.ipynb` dans ton notebook Kaggle.

**Option B : Direct sur Kaggle**
1. Va sur **[Kaggle Notebooks](https://kaggle.com/code)**
2. **"New Notebook"**
3. Colle le code du `Kaggle_Setup_Training.ipynb`

### 4️⃣ Lancer l'entraînement

1. **Run all** (Ctrl+Alt+Enter) ou exécute cell par cell
2. **Attends** ~30-45min pour Phase 2 sur P100 (AMP activé) ✅
3. Vérifie la console pour les erreurs

### 5️⃣ Télécharger les résultats

Une fois terminé :
1. **Data** (onglet left) → **Output**
2. Télécharge :
   - `checkpoints/` (modèles pré-entraînés)
   - `results/` (métriques, graphiques)
   - `logs/` (logs d'entraînement)

---

## 📊 Ce que le notebook fait

| Étape | Description | Durée | Status |
|-------|-------------|-------|--------|
| 0 | Clone du repo GitHub | 1min | ✅ Automatique |
| 1 | Install dépendances (PyTorch, RDKit, etc.) | 5min | ✅ Automatique |
| 2 | Setup GPU et vérifications | 1min | ✅ Automatique |
| 3 | Phase 1 (Pré-entraînement GNN) | 1-2h | ⏭️ Optionnel (SKIP_PHASE1=True) |
| 4 | Phase 2 (Fine-tuning Toxicité) | 30-45min | ✅ **À lancer** |
| 5 | Phase 3 (Multi-propriétés) | 20-30min | ⏭️ Optionnel (SKIP_PHASE3=True) |

---

## 🎮 Configuration Recommandée

### Pour débuter rapidement:
```python
SKIP_PHASE1 = True    # Phase 1 prend trop longtemps
SKIP_PHASE3 = True    # Phase 3 optionnelle
# Cela lance juste Phase 2 (toxicité) = 45min
```

### Pour entraînement complet:
```python
SKIP_PHASE1 = False   # Pré-entraîne sur ZINC
SKIP_PHASE3 = False   # Fine-tune multi-propriétés
# Durée totale: ~2-3h (limite de session Kaggle = 12h)
```

---

## 📥 Dépendances Installées

Le notebook installe automatiquement:

```
✅ PyTorch 2.0+ (avec CUDA)
✅ PyTorch Geometric (GNN)
✅ RDKit (chimie moléculaire)
✅ DeepChem (données Tox21)
✅ Scikit-learn, Pandas, NumPy
✅ Matplotlib, BeautifulSoup4
✅ Et tout ce qu'il faut...
```

**Aucun setup manuel nécessaire** ✅

---

## 🎯 Données Utilisées

| Phase | Source | Statut |
|-------|--------|--------|
| Phase 1 | ZINC (molécules organiques) | Auto-téléchargé ✅ |
| Phase 2 | Tox21 (toxicité) | Auto-téléchargé depuis DeepChem ✅ |
| Phase 3 | Custom (multi-propriétés) | Données locales ou synthétiques |

**Rien à uploader manuellement** ✅

---

## 🚨 Dépannage Courant

### ❌ "Module not found: rdkit"
→ Réexécute la cell "Installation de RDKit"

### ❌ "CUDA out of memory"
→ Baisse `batch_size` dans Phase 2:
```python
--batch_size 16  # au lieu de 32
```

### ❌ "DeepChem download failed"
→ Utilise Tox21 pré-téléchargé:
```python
--train_csv /kaggle/input/tox21/train.csv
--val_csv /kaggle/input/tox21/val.csv
```

### ❌ "Kernel died unexpectedly"
→ Kaggle a tué la session (>12h)
→ Réexécute le notebook (tes checkpoints sont sauvegardés)

---

## 💡 Tips & Tricks

### 1. Monitorer la GPU
```python
!nvidia-smi -l 1  # Affiche GPU toutes les 1sec
```

### 2. Sauvegarder pendant l'entraînement
```python
# Dans Kaggle, les checkpoints sont auto-sauvegardés dans /output
# Tu peux les télécharger à tout moment
```

### 3. Continuer sur GPU Plus puissant?
Si tu atteins la limite de Kaggle:
- **Colab GPU** (gratuit, plus lent)
- **AWS SageMaker** (payant, plus rapide)
- **Paperspace Gradient** (gratuit tier limité)

### 4. Exporter le modèle
```python
import torch
model = torch.load('checkpoints/phase2/best_toxicity_model.pth')
# Utilise-le localement sur VSCode
```

---

## 📈 Résultats Attendus

### Phase 2 (Toxicité)
```
Epoch 1/5:
  Train Loss: 0.45, Val Loss: 0.38, AUC: 0.82
  ...
Epoch 5/5:
  Train Loss: 0.28, Val Loss: 0.32, AUC: 0.88
  ✅ Best model saved → checkpoints/phase2/best_toxicity_model.pth
```

### Checkpoints créés
```
checkpoints/
├── phase1/
│   ├── sovereign_encoder_v1.pth (si Phase 1 lancée)
│   └── ...
├── phase2/
│   ├── best_toxicity_model.pth ✅ (toxicité fine-tuned)
│   ├── last_model.pth
│   └── ...
└── phase3/
    └── (si Phase 3 lancée)
```

---

## 🔄 Workflow Complet

```
Local (VSCode)              GitHub                  Kaggle
    ↓                          ↓                       ↓
Modifie code       →       git push         →   git clone
                                            →   Notebook Training
                                            →   Auto-download data
                                            →   Train avec GPU P100
                                            ↓
                            Download       ← Checkpoints
                            checkpoints
                                ↓
Local (VSCode)
    ↓
Utilise modèle entraîné
```

---

## 📞 Support

Si tu as des questions:
1. Regarde les **logs du notebook** (console)
2. Vérifie les **requirements.txt**
3. Check le **config.py** pour les paramètres
4. Post une issue sur GitHub 📧

---

## 🎉 Résumé

✅ **Clone repo** → **Notebook Kaggle** → **Run all** → **Attend** → **Télécharge checkpoints**

**Durée:** 45min (Phase 2 seule) → 2-3h (Phase 1+2+3)

**Coût:** 0$ (GPU Kaggle gratuit) 🎁

**Prochaine étape:** Utilise les checkpoints sur VSCode local ! 🚀

---

Bonne chance ! 🍀
