# ✅ Kaggle Setup Checklist - 5 Minutes

## 🎯 Objectif
Entraîner ton modèle GNN **Panacee** sur le **GPU P100 gratuit de Kaggle** en 45min max.

---

## ✔️ PRÉ-SETUP (avant Kaggle)

- [ ] Compte Kaggle créé (gratuit sur kaggle.com)
- [ ] Repo GitHub poussé ✅ (déjà fait!)
- [ ] Ce fichier lu ✅

---

## ✔️ ÉTAPE 1 - Créer le Notebook Kaggle (2 min)

- [ ] Va sur **kaggle.com** → **"Create"** → **"Notebook"**
- [ ] Choisis **"Python"**
- [ ] Donne-lui un nom (ex: "Panacee-GNN-Training")
- [ ] Clique **"Create Notebook"**

---

## ✔️ ÉTAPE 2 - Activer GPU (1 min)

- [ ] Clique **⚙️ Settings** (coin haut-droit du notebook)
- [ ] Sélectionne **"Accelerator"** → **"GPU"** (P100 ou meilleur)
- [ ] Clique **"Save"**
- [ ] Reviens au notebook

---

## ✔️ ÉTAPE 3 - Copier le Code (1 min)

**Deux options:**

### Option A: Via GitHub (simple)
1. [ ] Première cell, copie:
```python
!git clone https://github.com/jumoras0000/savh_gnn.git
```
2. [ ] Exécute (Shift+Enter)

### Option B: Copier-coller du notebook
1. [ ] Ouvre `Kaggle_Setup_Training.ipynb` (depuis GitHub)
2. [ ] Copy toutes les cells
3. [ ] Paste dans ton Kaggle Notebook

---

## ✔️ ÉTAPE 4 - Configuration (30 secondes)

Dans la cell "ÉTAPE 3️⃣ - Lancer PHASE 2", configure:

```python
SKIP_PHASE1 = True      # ✅ Skip Phase 1 (prend trop longtemps)
SKIP_PHASE3 = True      # ✅ Skip Phase 3 (optionnel)
```

**Résultat:** Juste Phase 2 (toxicité) = ~45min

---

## ✔️ ÉTAPE 5 - Lancer l'entraînement (1 min)

- [ ] **Ctrl+Alt+Enter** (Run All) OU
- [ ] Exécute cell par cell (Shift+Enter)
- [ ] Attends les install des dépendances (~5min)
- [ ] Attends l'entraînement Phase 2 (~40min)

---

## ⏱️ Timeline Attendue

| Étape | Temps | Action |
|-------|-------|--------|
| 0. Clone repo | 1min | ⏳ Attends |
| 1. Install deps | 5min | ⏳ Attends (RDKit peut être long) |
| 2. GPU setup | 1min | ✅ Auto |
| 3. Sanity check | 1min | ✅ Auto |
| 4. Phase 2 train | 40min | ⏳ Attends (bon moment pour café ☕) |
| 5. Download résultats | 5min | ✅ Télécharge les checkpoints |
| **TOTAL** | **~53min** | **GPU P100 gratuit** 🎉 |

---

## 📊 Ce qui se passe (en arrière-plan)

```
1. Télécharge les données Tox21 (toxicité) → DeepChem
2. Construit les graphes moléculaires (SMILES → Graphes)
3. Charge le modèle pré-entraîné Phase 1
4. Fine-tune sur données Tox21
5. Évalue et sauvegarde le meilleur modèle
6. Génère métriques et plots
```

---

## 💾 Télécharger les Résultats

Une fois terminé:

1. [ ] Clique **"Data"** (onglet left)
2. [ ] Clique **"Output"**
3. [ ] Télécharge:
   - `savh_gnn/Projet_Panacee/checkpoints/phase2/` (modèles)
   - `savh_gnn/Projet_Panacee/results/` (métriques, plots)
   - `savh_gnn/Projet_Panacee/logs/` (logs détaillés)

4. [ ] Sauvegarde-les localement (VSCode)

---

## 🎯 Résultats Attendus

### Si tout est ✅ OK:
```
✅ Données Tox21 téléchargées (50 MB)
✅ Modèle Phase 1 chargé (70 MB)
✅ GPU P100 détecté (16 GB VRAM)
✅ Phase 2 entraînement:
   Epoch 1/5: Loss=0.45, AUC=0.81
   Epoch 2/5: Loss=0.38, AUC=0.84
   Epoch 3/5: Loss=0.35, AUC=0.86
   Epoch 4/5: Loss=0.33, AUC=0.87
   Epoch 5/5: Loss=0.32, AUC=0.88 ✅
✅ Best model saved: checkpoints/phase2/best_model.pth (75 MB)
✅ Validation complete ✅
```

### Si problème ❌:
Regarde les messages d'erreur dans la console et check:
- [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) → "Dépannage"
- Console Kaggle (logs détaillés)

---

## 🚨 Erreurs Courantes & Fixes

### ❌ "AttributeError: module 'rdkit' has no attribute..."
**Fix:** Réexécute la cell "Installation de RDKit"

### ❌ "CUDA out of memory"
**Fix:** Change en Phase 2:
```python
--batch_size 16  # au lieu de 32
```

### ❌ "DeepChem download failed"
**Fix:** Manque rien, ça télécharge en background. Attends 5-10min.

### ❌ "Kernel died / Session ended"
**Fix:** Kaggle tue les sessions après 12h d'inactivité. Réexécute - tes checkpoints sont sauvegardés!

---

## 📌 Important à Savoir

✅ **Kaggle Notebooks = Gratuit** (GPU, CPU, Storage)
✅ **Aucun upload de données nécessaire** (tout est auto-téléchargé)
✅ **Max 12h par session** (mais Phase 2 prend 45min)
✅ **Tous les checkpoints sont sauvegardés** (tu peux les télécharger n'importe quand)
✅ **GPU P100 = 16GB VRAM** (suffisant pour ce projet)

---

## 🔗 Liens Rapides

| Ressource | Lien |
|-----------|------|
| Kaggle | https://kaggle.com |
| Ton Repo | https://github.com/jumoras0000/savh_gnn |
| Notebook Kaggle | `Kaggle_Setup_Training.ipynb` |
| Guide Complet | [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) |
| Config | `Projet_Panacee/src/config.py` |

---

## ✅ Après l'Entraînement

1. [ ] Télécharge les checkpoints depuis Kaggle Output
2. [ ] Utilise-les localement dans VSCode
3. [ ] Push les résultats sur GitHub
4. [ ] Célèbre! 🎉

---

## 🚀 Cas d'Usage Suivants

**Prochaine fois que tu veux entrainer:**
1. Créé un nouveau Kaggle Notebook (clone ce checklist)
2. Lance "Run All"
3. Attends 45min
4. Télécharge les checkpoints

**C'est tout!** Pas besoin de réinstaller, tout est préconfiguré ✅

---

## 📝 Notes Personnelles

Utilise cette zone pour noter tes observations:

```
- Date d'entraînement: _______________
- Résultats AUC-ROC: _______________
- Erreurs rencontrées: _______________
- Améliorations à tester: _______________
```

---

**T'as tout ce qu'il faut!** 🚀

Bonne chance avec Panacee! 🧬
