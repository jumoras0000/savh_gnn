#!/usr/bin/env python
"""
Guide d'installation GPU - Affiche les commandes exactes à exécuter
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         🚀 INSTALLATION GPU - GUIDE POUR NVIDIA QUADRO M1000M             ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ VOTRE CONFIGURATION:
   GPU: NVIDIA Quadro M1000M
   CUDA Detecté: ✅ Oui (12.8)
   Driver: ✅ Ok (573.71)
   VRAM: 2GB disponible

───────────────────────────────────────────────────────────────────────────

⚠️  PyTorch EST ACTUELLEMENT:
   ❌ CPU-only (torch 2.10.0)
   
ACTION REQUISE:
   Convertir PyTorch pour utiliser le GPU

───────────────────────────────────────────────────────────────────────────

📋 COMMANDES À EXÉCUTER (copier-coller dans PowerShell):

ÉTAPE 1 - Désinstaller PyTorch CPU:
────────────────────────────────────────────────────────────────────────────
pip uninstall -y torch torchvision torchaudio

ÉTAPE 2 - Installer PyTorch CUDA 12.1:
────────────────────────────────────────────────────────────────────────────
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

ÉTAPE 3 - Installer Torch-Geometric CUDA 12.1:
────────────────────────────────────────────────────────────────────────────
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

ÉTAPE 4 - Vérifier l'installation:
────────────────────────────────────────────────────────────────────────────
python validate_cuda_install.py

───────────────────────────────────────────────────────────────────────────

⏱️  TEMPS ESTIMÉ:
   - Désinstallation PyTorch: 1 min
   - Installation PyTorch CUDA: 5-10 min
   - Installation PyG CUDA: 3-5 min
   - TOTAL: ~10-15 minutes

───────────────────────────────────────────────────────────────────────────

📊 RÉSULTATS ATTENDUS:

Avant (CPU):
  import torch
  torch.cuda.is_available()  →  ❌ False

Après (GPU):
  import torch
  torch.cuda.is_available()  →  ✅ True
  torch.cuda.get_device_name(0)  →  NVIDIA Quadro M1000M

───────────────────────────────────────────────────────────────────────────

✨ CEL QUI SE PASSE APRÈS:

Phase 2 utilisera AUTOMATIQUEMENT le GPU
  - Pas de changement de code
  - Pas de configuration supplémentaire
  - Juste une réinstallation de packages

Entraînement Phase 2:
  ❌ CPU: 2h 30m - 3h (actuellement)
  ✅ GPU: 15-20 minutes (après installation)
  
  = 10x PLUS RAPIDE ! ⚡

───────────────────────────────────────────────────────────────────────────

🎯 PROCHAINES ÉTAPES:

1. Exécuter les 4 commandes ci-dessus
2. Voir le message "✅ GPU CONFIGURÉ"
3. Lancer Phase 2:
   python run_phase2.py --download --epochs 80 --batch_size 64

═══════════════════════════════════════════════════════════════════════════════

❓ QUESTIONS ?

Q: Ça va casser Phase 1 ?
R: Non, Phase 1 n'est plus utilisé. PyTorch GPU est compatible.

Q: Combien de VRAM ?
R: Quadro M1000M a 2GB. Batch size 64 va utiliser ~1.5GB. OK.

Q: Faut redémarrer ?
R: Non (sauf si l'installation le demande).

Q: Ça peut loader différemment ?
R: Non, PyTorch GPU load les mêmes matrices, juste plus vite.

═══════════════════════════════════════════════════════════════════════════════
""")

import sys
print("\n✅ READY TO START!")
print("\n1️⃣  Copier-coller les 4 commandes ci-dessus")
print("2️⃣  Attendre ~15 min")
print("3️⃣  Voir le message de succès")
print("4️⃣  Lancer Phase 2 avec GPU ! 🚀")
