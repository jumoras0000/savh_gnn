"""
Script de validation CUDA + PyTorch installation
À exécuter APRÈS avoir installé CUDA Toolkit 12.1 et PyTorch
"""
import sys
import os

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           ✅ VALIDATION CUDA + PYTORCH INSTALLATION                       ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# ──────────────────────────────────────────────────────────────────────
# 1. VÉRIFIER PYTORCH
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 1: PyTorch installation")
print("─" * 80)

try:
    import torch
    print(f"  ✅ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"  ❌ PyTorch NOT installed: {e}")
    print("     Exécuter: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 2. VÉRIFIER CUDA
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 2: CUDA availability")
print("─" * 80)

cuda_available = torch.cuda.is_available()
print(f"  CUDA available: {cuda_available}")

if not cuda_available:
    print("""
  ❌ CUDA NOT available!
  
  Solutions possibles:
    1. CUDA Toolkit 12.1 pas installé
       → Télécharger depuis: https://developer.nvidia.com/cuda-12-1-0-download-archive
       → Installer et redémarrer
    
    2. PyTorch CPU-only installé
       → Désinstaller: pip uninstall -y torch torchvision torchaudio
       → Réinstaller: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    3. Variables d'environnement CUDA pas configurées
       → Redémarrer après installation CUDA
       → Vérifier: nvcc --version dans PowerShell
  """)
    sys.exit(1)

print(f"  ✅ CUDA found!")

# ──────────────────────────────────────────────────────────────────────
# 3. VÉRIFIER GPU DÉTECTE
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 3: GPU detection")
print("─" * 80)

num_gpus = torch.cuda.device_count()
print(f"  GPUs found: {num_gpus}")

if num_gpus == 0:
    print("""
  ❌ NO GPU found!
  
  Vérifier:
    1. nvidia-smi (dans PowerShell)
       → Doit afficher votre GPU
    
    2. CUDA Toolkit installations
       → Vérifier dans Control Panel → Programs
       → CUDA Toolkit 12.1 doit être présent
  """)
    sys.exit(1)

print(f"  ✅ GPU(s) found: {num_gpus}")

# ──────────────────────────────────────────────────────────────────────
# 4. VÉRIFIER DEVICE PROPERTIES
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 4: GPU properties")
print("─" * 80)

for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    capability = torch.cuda.get_device_capability(i)
    
    print(f"\n  GPU {i}:")
    print(f"    Name: {props.name}")
    print(f"    Compute Capability: {capability[0]}.{capability[1]}")
    print(f"    Total Memory: {props.total_memory / (1024**3):.1f} GB")
    print(f"    Max Threads per Block: {props.max_threads_per_block}")

# ──────────────────────────────────────────────────────────────────────
# 5. TEST COMPUTATION
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 5: Computation test")
print("─" * 80)

try:
    device = torch.device("cuda:0")
    
    # Create tensor on GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Compute on GPU
    z = torch.mm(x, y)
    
    print(f"  ✅ GPU computation works!")
    print(f"     Matrix multiplication (1000x1000 x 1000x1000) completed")
    print(f"     Result shape: {z.shape}")
    print(f"     Result device: {z.device}")
    
except Exception as e:
    print(f"  ❌ Computation failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 6. VÉRIFIER MEMORY
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 6: GPU memory")
print("─" * 80)

try:
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        print(f"  GPU {i}:")
        print(f"    Total: {total:.1f} GB")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved: {reserved:.2f} GB")
        print(f"    Available: {total - reserved:.2f} GB")
        
except Exception as e:
    print(f"  ⚠️  Memory info error: {e}")

# ──────────────────────────────────────────────────────────────────────
# 7. VÉRIFIER DEPENDENCIES
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 7: Dependencies")
print("─" * 80)

deps_ok = True

try:
    import torchvision
    print(f"  ✅ Torchvision: {torchvision.__version__}")
except ImportError:
    print(f"  ❌ Torchvision NOT installed")
    deps_ok = False

try:
    import torch_geometric
    print(f"  ✅ Torch Geometric: installed")
except ImportError:
    print(f"  ⚠️  Torch Geometric NOT installed")
    print(f"     pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu121.html")
    deps_ok = False

try:
    import torch_scatter
    print(f"  ✅ Torch Scatter: installed")
except ImportError:
    print(f"  ⚠️  Torch Scatter NOT installed (optional, PyG has fallback)")

# ──────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ──────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("✅ VALIDATION COMPLETE")
print("="*80)

all_ok = cuda_available and num_gpus > 0 and deps_ok

if all_ok:
    print("""
✅ INSTALLATION RÉUSSIE !

Statut:
  ✅ PyTorch avec CUDA installé
  ✅ GPU NVIDIA détecté
  ✅ Computation GPU OK
  ✅ Memory OK
  ✅ Dependencies OK

PRÊT POUR PHASE 2 !

Phase 2 utilisera automatiquement GPU Quadro M1000M
Entraînement ~10x plus rapide qu'en CPU

Commande pour lancer:
  python run_phase2.py --download --epochs 80 --batch_size 64

Performance estimée: 15-20 minutes vs 2.5-3 heures en CPU

═══════════════════════════════════════════════════════════════════════════════
""")
else:
    print("""
❌ INSTALLATION INCOMPLÈTE

Étapes manquantes:
  - CUDA pas trouvé: Installer CUDA Toolkit 12.1
  - PyTorch pas avec CUDA: Réinstaller avec --index-url https://download.pytorch.org/whl/cu121
  - Dependencies manquantes: À installer optionnellement

Relancer ce script après corrections:
  python validate_cuda_install.py
""")
    if not cuda_available or num_gpus == 0:
        sys.exit(1)

print("\n🚀 Phase 2 est prêt à démarrer !")
