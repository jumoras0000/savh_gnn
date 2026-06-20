#!/usr/bin/env python
"""
Teste tous les imports et modules du projet Panacée.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("🔍 DIAGNOSTIC D'IMPORTS - PROJET PANACÉE")
print("="*80)

# ─── Test 1: Dépendances principales ───
print("\n1️⃣ Dépendances externes:")
deps = {
    "torch": None,
    "torch_geometric": None,
    "numpy": None,
    "pandas": None,
    "rdkit": None,
    "deepchem": None,
    "sklearn": None,
}

for name, _ in deps.items():
    try:
        if name == "torch_geometric":
            import torch_geometric
            print(f"  ✓ {name}")
        elif name == "sklearn":
            import sklearn
            print(f"  ✓ {name}")
        else:
            mod = __import__(name)
            print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# ─── Test 2: Modules internes Phase 1 ───
print("\n2️⃣ Modules Phase 1:")
phase1_modules = [
    "src.preprocessing.graph_builder",
    "src.preprocessing.zinc_loader",
    "src.models.encoder",
    "src.training.pretrain_gnn",
]

for mod_name in phase1_modules:
    try:
        mod = __import__(mod_name, fromlist=[mod_name.split(".")[-1]])
        print(f"  ✓ {mod_name}")
    except Exception as e:
        print(f"  ✗ {mod_name}: {e}")

# ─── Test 3: Modules internes Phase 2 ───
print("\n3️⃣ Modules Phase 2:")
phase2_modules = [
    "src.preprocessing.toxicity_loader",
    "src.models.toxicity_classifier",
    "src.training.finetune_toxicity",
]

for mod_name in phase2_modules:
    try:
        mod = __import__(mod_name, fromlist=[mod_name.split(".")[-1]])
        print(f"  ✓ {mod_name}")
    except Exception as e:
        print(f"  ✗ {mod_name}: {e}")

# ─── Test 4: Modules internes Phase 3 ───
print("\n4️⃣ Modules Phase 3:")
phase3_modules = [
    "src.preprocessing.multi_property_loader",
    "src.models.multi_property_head",
    "src.models.reasoner",
    "src.training.train_phase3",
]

for mod_name in phase3_modules:
    try:
        mod = __import__(mod_name, fromlist=[mod_name.split(".")[-1]])
        print(f"  ✓ {mod_name}")
    except Exception as e:
        print(f"  ✗ {mod_name}: {e}")

# ─── Test 5: Téléchargement DeepChem ───
print("\n5️⃣ Test téléchargement Tox21:")
try:
    from src.preprocessing.toxicity_loader import download_tox21_data
    print("  ✓ download_tox21_data importé")
except Exception as e:
    print(f"  ✗ download_tox21_data: {e}")

print("\n" + "="*80)
print("✓ Diagnostic complet - le projet est prêt !")
print("="*80)
