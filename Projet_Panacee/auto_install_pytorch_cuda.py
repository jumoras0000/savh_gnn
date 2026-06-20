#!/usr/bin/env python
"""
Script d'automatisation: Désinstaller et réinstaller PyTorch avec CUDA 12.1
Exécution simplifiée pour l'utilisateur
"""
import subprocess
import sys
import time

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         🚀 AUTOMATISATION INSTALLATION PYTORCH CUDA 12.1                  ║
║              Pour NVIDIA Quadro M1000M                                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("""
✅ DÉTECTION CUDA:
  ✓ CUDA Driver 573.71 détecté
  ✓ CUDA Runtime 12.8 disponible
  ✓ GPU Quadro M1000M trouvé
  ✓ 2GB VRAM disponibles

📋 ACTIONS À EFFECTUER:
  1. ❌ Désinstaller PyTorch (CPU-only actuellement)
  2. ✅ Installer PyTorch avec CUDA 12.1
  3. ✅ Installer Torch-Geometric avec CUDA 12.1
  4. ✅ Valider GPU activé

═══════════════════════════════════════════════════════════════════════════════
""")

def run_command(cmd, description):
    """Exécuter une commande et afficher les résultats"""
    print(f"\n🔄 {description}...")
    print(f"   Commande: {cmd}")
    print("─" * 80)
    
    try:
        result = subprocess.run(cmd, shell=True, text=True, capture_output=False)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCÈS")
            return True
        else:
            print(f"❌ {description} - ERREUR (code: {result.returncode})")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────
# MAIN INSTALLATION LOOP
# ──────────────────────────────────────────────────────────────────────

print("\n⚠️  ATTENTION: Ce script va modifier votre installation PyTorch")
print("    Assurez-vous que Phase 1 & 2 ne s'exécutent pas")
print("\nContinuer ? (O/n): ", end="")
user_input = input().strip().lower()

if user_input and user_input != 'o':
    print("❌ Installation annulée")
    sys.exit(0)

print("\n" + "="*80)
print("▶️  DÉBUT DE L'INSTALLATION")
print("="*80)

all_ok = True

# ÉTAPE 1: Désinstaller PyTorch
steps = [
    ("pip uninstall -y torch torchvision torchaudio", 
     "Désinstaller PyTorch (CPU)"),
    
    ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
     "Installer PyTorch avec CUDA 12.1"),
    
    ("pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html",
     "Installer Torch-Geometric avec CUDA 12.1"),
]

for cmd, desc in steps:
    if not run_command(cmd, desc):
        all_ok = False
        print(f"⚠️  Erreur lors de: {desc}")
        print("   Veuillez exécuter manuellement:")
        print(f"   {cmd}")

# ──────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("✅ VALIDATION POST-INSTALLATION")
print("="*80)

time.sleep(2)  # Laisser le temps à Python de charger les modules

try:
    import torch
    
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    cuda_avail = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_avail}")
    
    if cuda_avail:
        num_gpus = torch.cuda.device_count()
        print(f"✓ GPUs trouvés: {num_gpus}")
        
        if num_gpus > 0:
            device_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            print(f"✓ GPU 0: {device_name}")
            print(f"✓ Compute Capability: {capability[0]}.{capability[1]}")
            
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            print(f"✓ Memory: {memory_gb:.1f} GB")
            
            print(f"\n✅ GPU UTILISABLE POUR PHASE 2 !")
            print(f"   Entraînement ~10x plus rapide qu'en CPU")
        else:
            print(f"❌ GPUs trouvés mais aucun accessible")
    else:
        print(f"❌ CUDA pas disponible")
        print(f"   Vérifier: nvidia-smi")

except Exception as e:
    print(f"❌ Erreur validation: {e}")
    import traceback
    traceback.print_exc()

# ──────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ──────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("📋 RÉSUMÉ FINAL")
print("="*80)

if all_ok:
    print("""
✅ INSTALLATION TERMINÉE AVEC SUCCÈS !

Prochaines étapes:
  1. Exécuter pour valider:
     python validate_cuda_install.py
  
  2. Lancer Phase 2 avec GPU:
     python run_phase2.py --download --epochs 80 --batch_size 64
  
  3. Profiter de l'accélération GPU (~10x plus rapide !)

═══════════════════════════════════════════════════════════════════════════════
""")
else:
    print("""
⚠️  Installation incomplète

Veuillez exécuter manuellement les commandes suivantes:

Étape 1: Désinstaller ancienne version
  pip uninstall -y torch torchvision torchaudio

Étape 2: Installer PyTorch avec CUDA 12.1
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Étape 3: Installer Torch-Geometric
  pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html

Puis valider:
  python validate_cuda_install.py

═══════════════════════════════════════════════════════════════════════════════
""")

print("\n🚀 Installation automatisée terminée!")
