# FIX_DEEPCHEM_WINDOWS.md

## Problème : DLL load failed while importing _rgi_cython

### Root Cause
- SciPy 1.17.0 avait des DLLs compilées incompatibles avec Windows
- DeepChem dépend de SciPy via pandas et d'autres modules
- Conflit de versions entre rdkit, rdkit-pypi et deepchem

### Étapes de Correction Appliquées

#### 1. **Nettoyage complet des dépendances**
```bash
pip uninstall -y deepchem rdkit rdkit-pypi scipy
```

#### 2. **Réinstallation dans le bon ordre**
```bash
# Versions stables et compatibles
pip install rdkit-pypi==2022.9.5
pip install scipy==1.11.4
pip install deepchem==2.7.1
```

#### 3. **Correction encodage Windows**
Ajout de headers UTF-8 à `run_phase2.py` et `run_phase3.py`:
```python
# -*- coding: utf-8 -*-
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### Versioning Final
- Python: 3.11.9
- PyTorch: 2.5.1+cu121
- RDKit: 2022.9.5 (via rdkit-pypi)
- DeepChem: 2.7.1
- SciPy: 1.11.4

### Validation
✓ Tous les imports testés avec succès
✓ Tox21 dataset téléchargé correctement (6258 train, 782 val, 783 test)
✓ 13 colonnes confirmées (smiles + 12 tasks)
✓ Données chimiquement valides

### Notes
- Les avertissements DeepChem (TensorFlow, DGL, Lightning) sont normaux
- Ces dépendances sont optionnelles pour la toxicité
- Phase 2 peut maintenant être lancé sans erreurs
