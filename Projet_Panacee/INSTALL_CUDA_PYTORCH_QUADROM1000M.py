"""
Installation CUDA & PyTorch pour NVIDIA Quadro M1000M
Quadro M1000M: Compute Capability 5.2, supporte CUDA 5.2-12.x
"""
import subprocess
import sys

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║        🚀 INSTALLATION CUDA + PYTORCH POUR NVIDIA QUADRO M1000M           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("""
📋 VOTRE GPU:
  - GPU: NVIDIA Quadro M1000M
  - Architecture: Maxwell
  - Compute Capability: 5.2
  - CUDA Support: 5.2 - 12.1
  - Recommandation: CUDA 11.8 ou 12.1 (stable)

✅ PLAN D'INSTALLATION:
  1. Installer CUDA 12.1 (recommandé, stable pour Maxwell)
  2. Installer cuDNN (acceleration library)
  3. Reinstaller PyTorch avec CUDA 12.1
  4. Valider l'installation

═══════════════════════════════════════════════════════════════════════════════
""")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 1: Afficher les instructions d'installation
# ──────────────────────────────────────────────────────────────────────

print("""
🔧 ÉTAPE 1: INSTALLER CUDA TOOLKIT 12.1
───────────────────────────────────────────────────────────────────────────

Télécharger et installer CUDA Toolkit 12.1:
  URL: https://developer.nvidia.com/cuda-12-1-0-download-archive

Options d'installation:
  ✓ Sélectionner "Windows"
  ✓ Sélectionner votre architecture (x86_64)
  ✓ Sélectionner version (10 ou 11)
  ✓ Installer avec les options par défaut

Vérifier l'installation (après redémarrage):
  Ouvrir PowerShell et exécuter:
    nvcc --version
    nvidia-smi

Vous devriez voir:
  - nvcc release 12.1
  - cuda_runtime_api.h version 12.1

═══════════════════════════════════════════════════════════════════════════════
""")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 2: Vérifier CUDA actuel
# ──────────────────────────────────────────────────────────────────────

print("\n✓ Vérification CUDA actuel...")
print("─" * 80)

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ NVIDIA GPU détecté:")
        lines = result.stdout.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(f"  {line}")
    else:
        print("❌ nvidia-smi not found - CUDA toolkit not installed")
        print("   Voir ÉTAPE 1 pour installer CUDA Toolkit 12.1")
except Exception as e:
    print(f"⚠️  nvidia-smi pas trouvé: {e}")
    print("   Assurez-vous que CUDA Toolkit 12.1 est installé")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 3: Uninstaller l'ancienne version PyTorch
# ──────────────────────────────────────────────────────────────────────

print("\n\n🔄 ÉTAPE 2: RÉINSTALLER PYTORCH AVEC CUDA 12.1")
print("─" * 80)

print("""
AVANT d'installer la nouvelle version, désinstaller l'ancienne:
""")

commands_uninstall = [
    "pip uninstall -y torch torchvision torchaudio",
]

print("\nCommandes à exécuter:")
for cmd in commands_uninstall:
    print(f"  {cmd}")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 4: Installation PyTorch avec CUDA 12.1
# ──────────────────────────────────────────────────────────────────────

print("\n\n📦 ÉTAPE 3: INSTALLER PYTORCH AVEC CUDA 12.1")
print("─" * 80)

pytorch_install_cmd = """pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"""

print("""
Après désinstallation, exécuter:

  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Cela installera:
  ✓ PyTorch 2.x avec CUDA 12.1
  ✓ Torchvision avec CUDA 12.1
  ✓ Torchaudio avec CUDA 12.1

Temps d'installation: ~5-10 minutes (selon connexion)
""")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 5: Réinstaller PyG
# ──────────────────────────────────────────────────────────────────────

print("\n📦 ÉTAPE 4: RÉINSTALLER TORCH-GEOMETRIC")
print("─" * 80)

print("""
Après PyTorch, réinstaller PyG pour CUDA 12.1:

  pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

OU pour la dernière version:

  pip install torch-geometric

Temps d'installation: ~2-3 minutes
""")

# ──────────────────────────────────────────────────────────────────────
# ÉTAPE 6: Valider l'installation
# ──────────────────────────────────────────────────────────────────────

print("\n✅ ÉTAPE 5: VALIDER L'INSTALLATION")
print("─" * 80)

print("""
Après tout, exécuter ce script de validation:

  python validate_cuda_install.py

Ou manuellement en Python:

  import torch
  print(torch.__version__)                    # 2.x.x
  print(torch.cuda.is_available())            # True
  print(torch.cuda.get_device_name(0))        # NVIDIA Quadro M1000M
  print(torch.cuda.get_device_capability(0))  # (5, 2)
""")

# ──────────────────────────────────────────────────────────────────────
# RÉSUMÉ
# ──────────────────────────────────────────────────────────────────────

print("\n\n" + "="*80)
print("📋 RÉSUMÉ DE L'INSTALLATION")
print("="*80)

print("""
ORDRE D'INSTALLATION:
  1. Télécharger CUDA Toolkit 12.1 (si pas déjà installé)
     URL: https://developer.nvidia.com/cuda-12-1-0-download-archive
     
  2. Installer CUDA Toolkit 12.1
     - Redémarrer après installation
     - Vérifier avec: nvcc --version
  
  3. Désinstaller ancienne PyTorch:
     pip uninstall -y torch torchvision torchaudio
  
  4. Installer PyTorch avec CUDA 12.1:
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  
  5. Réinstaller PyG:
     pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
  
  6. Valider:
     python validate_cuda_install.py

APRÈS L'INSTALLATION:
  - Phase 2 utilisera automatiquement GPU
  - Aucun changement de code nécessaire
  - Entraînement ~10x plus rapide

TIMING ESTIMÉ:
  - Installation CUDA: 10-20 min
  - Installation PyTorch: 10 min
  - Installation PyG: 5 min
  - Total: ~30 min

═══════════════════════════════════════════════════════════════════════════════
""")

print("\n🚀 Prêt à commencer ?")
print("\nÉtape 1: Télécharger CUDA 12.1 depuis:")
print("  https://developer.nvidia.com/cuda-12-1-0-download-archive")
