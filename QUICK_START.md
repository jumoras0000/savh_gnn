# 🚀 QUICK START - 5 Min Setup

## En Deux Mots

**Tu veux entraîner ton modèle GNN?** Utilise Kaggle (GPU P100 gratuit)!

---

## 🎯 Pour Commencer Maintenant

### 1️⃣ Crée un Notebook Kaggle
```
kaggle.com → Create → Notebook → Python
```

### 2️⃣ Active le GPU
```
Settings ⚙️ → Accelerator → GPU (P100)
```

### 3️⃣ Copie ce code
```python
!git clone https://github.com/jumoras0000/savh_gnn.git
```

### 4️⃣ Lance l'entraînement
- Copie le contenu de `Kaggle_Setup_Training.ipynb`
- Run All (Ctrl+Alt+Enter)
- Attends ~45min

### 5️⃣ Télécharge les résultats
```
Data → Output → Download checkpoints/
```

---

## 📚 Documentation Complète

| Document | Contenu |
|----------|---------|
| **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** | ✅ Checklist 5 min (À LIRE D'ABORD!) |
| **[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)** | 📖 Guide complet Kaggle + dépannage |
| **[Kaggle_Setup_Training.ipynb](Kaggle_Setup_Training.ipynb)** | 🔥 Notebook prêt-à-l'emploi |
| **[README.md](README.md)** | 📘 Documentation complète du projet |
| **[Projet_Panacee/requirements.txt](Projet_Panacee/requirements.txt)** | 📦 Dépendances |

---

## ⚡ Fastest Path (Copy-Paste)

```python
# Cell 1 - Clone repo
!git clone https://github.com/jumoras0000/savh_gnn.git
%cd savh_gnn/Projet_Panacee

# Cell 2 - Install deps (5 min)
!pip install -q torch-geometric rdkit-pypi deepchem scikit-learn pandas matplotlib

# Cell 3 - Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")

# Cell 4 - Train Phase 2 (40 min)
!python run_phase2.py --download --epochs 5 --batch_size 32 --save_dir checkpoints/phase2
```

**Done!** 🎉

---

## 🆘 Si ça échoue

1. Check la console Kaggle pour les erreurs
2. Lis [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) → Dépannage
3. Réexécute les cells d'installation

---

## 📊 Timeline

- ⏳ Install: 5 min
- ⏳ GPU Setup: 1 min
- ⏳ **Phase 2 Training: 40 min**
- ⏳ Download: 5 min
- **TOTAL: ~50 min** (GPU P100 gratuit!)

---

## ✅ Success Indicators

Après 50 min, tu dois voir:

```
✅ Données téléchargées
✅ Phase 2 entraînement complété
✅ AUC-ROC: 0.85-0.90
✅ Checkpoints sauvegardés
✅ Ready to download
```

---

## 🎯 Prochaines Étapes

1. **Utilise localement:**
   ```python
   import torch
   model = torch.load('checkpoints/phase2/best_model.pth')
   # Prédis sur nouvelles molécules
   ```

2. **Améliore:**
   - Tune hyperparamètres (batch_size, learning_rate)
   - Ajoute Phase 1 pré-entraînement
   - Launch Phase 3 multi-propriétés

3. **Déploie:**
   - Crée une API (Flask/FastAPI)
   - Push sur Hugging Face Hub
   - Crée une web UI

---

## 🔗 Links

- GitHub: https://github.com/jumoras0000/savh_gnn
- Kaggle: https://kaggle.com/code
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

---

## ❓ FAQ Rapide

**Q: GPU gratuit sur Kaggle?**
A: ✅ Oui! P100 (16GB VRAM) gratuit

**Q: Combien de temps Phase 2?**
A: ~40 min avec P100

**Q: J'ai besoin de uploader des données?**
A: ❌ Non! Tout est auto-téléchargé (Tox21 via DeepChem)

**Q: Je peux garder les checkpoints après?**
A: ✅ Oui! Download depuis Kaggle Output

**Q: Ça marche sur ma machine locale?**
A: ✅ Oui! Mais tu besoin de GPU NVIDIA + CUDA

---

## 🎬 Let's Go!

**👉 Commencez par [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** ✅

Tu as tout ce qu'il faut! 🚀
