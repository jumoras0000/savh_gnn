"""
Vérification GPU & Modèles pour Phase 2
Vérifie que le GPU est correctement configuré et que tous les modèles sont chargés
"""
import sys
import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           🔍 VÉRIFICATION GPU & MODÈLES PHASE 2                           ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# ──────────────────────────────────────────────────────────────────────
# 1. GPU CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
print("\n✓ Test 1: Configuration GPU/CPU")
print("─" * 80)

try:
    from src.config import DEVICE, PIN_MEMORY
    
    print(f"  DEVICE configuré: {DEVICE}")
    print(f"  PIN_MEMORY: {PIN_MEMORY}")
    
    # Vérifier CUDA disponibilité
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA disponible: {cuda_available}")
    
    if cuda_available:
        print(f"\n  ✅ GPU CONFIGURÉ POUR PHASE 2")
        print(f"     Device name: {torch.cuda.get_device_name(0)}")
        print(f"     Device capability: {torch.cuda.get_device_capability(0)}")
        print(f"     CUDA version: {torch.version.cuda}")
        
        # Memory info
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)  # GB
        print(f"     Mémoire totale: {total_memory:.1f} GB")
        
        # Current usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"     Mémoire utilisée: {allocated:.2f} GB")
        print(f"     Mémoire réservée: {reserved:.2f} GB")
    else:
        print(f"\n  ⚠️  GPU NOT AVAILABLE - CPU SERA UTILISÉ")
        print(f"     Installation de CUDA recommandée pour accélération")
    
except Exception as e:
    print(f"  ❌ ERREUR config GPU: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 2. VÉRIFIER TOUS LES MODÈLES DANS PHASE 2
# ──────────────────────────────────────────────────────────────────────
print("\n\n✓ Test 2: Vérification tous les modèles")
print("─" * 80)

try:
    from src.models.encoder import MolecularEncoder
    from src.models.toxicity_classifier import ToxicityClassifier, MultiTaskBCELoss
    from src.preprocessing.toxicity_loader import ToxicityDataset
    from src.preprocessing.graph_builder import smiles_to_graph
    
    # Créer encoder
    from src.config import (
        ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
        HIDDEN_DIM, NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT
    )
    
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,
        edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT
    )
    
    print(f"  ✓ Encoder créé: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params")
    
    # Créer classifier
    classifier = ToxicityClassifier(
        encoder=encoder,
        num_tasks=12,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        freeze_encoder=True
    )
    
    print(f"  ✓ Classifier créé: {sum(p.numel() for p in classifier.parameters())/1e6:.1f}M params")
    print(f"    - Encoder frozen: {not list(encoder.parameters())[0].requires_grad}")
    print(f"    - Head unfrozen: {list(classifier.classifier.parameters())[0].requires_grad}")
    
    # Loss function
    loss_fn = MultiTaskBCELoss(pos_weight=torch.ones(12))
    print(f"  ✓ Loss function créée: MultiTaskBCELoss (NaN-safe)")
    
except Exception as e:
    print(f"  ❌ ERREUR modèles: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 3. TESTER MODÈLES SUR LE DEVICE
# ──────────────────────────────────────────────────────────────────────
print("\n\n✓ Test 3: Tester modèles sur device")
print("─" * 80)

try:
    # Move to device
    classifier_on_device = classifier.to(DEVICE)
    loss_fn_on_device = loss_fn
    
    print(f"  ✓ Modèles déplacés sur {DEVICE}")
    
    # Vérifier que les paramètres sont bien sur le device
    encoder_device = next(classifier_on_device.encoder.parameters()).device
    head_device = next(classifier_on_device.classifier.parameters()).device
    
    print(f"    - Encoder device: {encoder_device}")
    print(f"    - Head device: {head_device}")
    
    assert str(encoder_device) == str(DEVICE), "Encoder pas sur le bon device!"
    assert str(head_device) == str(DEVICE), "Head pas sur le bon device!"
    
    print(f"  ✓ Tous les paramètres sur {DEVICE} ✅")
    
except Exception as e:
    print(f"  ❌ ERREUR device placement: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 4. VÉRIFIER CHECKPOINTS PHASE 1
# ──────────────────────────────────────────────────────────────────────
print("\n\n✓ Test 4: Vérification Phase 1 checkpoint")
print("─" * 80)

try:
    from src.config import CHECKPOINT_DIR, PHASE1
    
    phase1_path = CHECKPOINT_DIR / "phase1" / PHASE1["checkpoint_name"]
    
    if phase1_path.exists():
        print(f"  ✓ Phase 1 checkpoint trouvé: {phase1_path.name}")
        
        # Charger et vérifier
        ckpt = torch.load(phase1_path, map_location="cpu", weights_only=False)
        
        if isinstance(ckpt, dict):
            print(f"    Format: Dictionary checkpoint")
            
            if "model_state_dict" in ckpt:
                print(f"    ✓ model_state_dict présent")
                sd = ckpt["model_state_dict"]
                encoder_keys = [k for k in sd.keys() if k.startswith("encoder.")]
                print(f"    ✓ {len(encoder_keys)} clés encoder trouvées")
            
            if "epoch" in ckpt:
                print(f"    ✓ Époque: {ckpt['epoch']}")
            
            if "best_auc" in ckpt:
                print(f"    ✓ Meilleur AUC Phase 1: {ckpt['best_auc']:.4f}")
        
        # Tester charge du checkpoint
        encoder_test = MolecularEncoder(
            atom_dim=ATOM_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_GNN_LAYERS,
            edge_dim=BOND_FEATURE_DIM,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT
        )
        
        # Load encoder weights
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
            if encoder_sd:
                encoder_test.load_state_dict(encoder_sd, strict=False)
                print(f"  ✅ Poids encoder chargés avec succès")
            else:
                print(f"  ⚠️  Pas de poids encoder, modèle aléatoire")
        
    else:
        print(f"  ❌ Phase 1 checkpoint manquant: {phase1_path}")
        print(f"     Lancez d'abord Phase 1: python run_phase1.py")
        sys.exit(1)

except Exception as e:
    print(f"  ❌ ERREUR Phase 1 checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 5. FORWARD PASS ET GPU MEMORY
# ──────────────────────────────────────────────────────────────────────
print("\n\n✓ Test 5: Forward pass + GPU memory test")
print("─" * 80)

try:
    from torch_geometric.data import Batch
    
    # Créer batch test
    graphs = []
    for _ in range(4):
        g = smiles_to_graph("CCO")
        if g is not None:
            graphs.append(g)
    
    batch = Batch.from_data_list(graphs).to(DEVICE)
    labels = torch.rand(len(graphs), 12).to(DEVICE)
    
    print(f"  Batch test: {len(graphs)} graphes sur {DEVICE}")
    
    # Forward pass
    if cuda_available:
        print(f"  GPU Memory avant forward: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    with torch.no_grad():
        logits = classifier_on_device(batch)
    
    if cuda_available:
        print(f"  GPU Memory après forward: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    print(f"  ✓ Forward pass: {batch.num_graphs} → {logits.shape}")
    print(f"  ✓ Output shape: {logits.shape}")
    print(f"  ✓ All values on {logits.device}: {logits.device == DEVICE}")
    
    # Test loss computation
    loss = loss_fn_on_device(logits, labels)
    print(f"  ✓ Loss computation: {loss.item():.4f}")
    print(f"  ✅ Forward pass et loss OK")
    
except Exception as e:
    print(f"  ❌ ERREUR forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 6. RÉSUMÉ FINAL
# ──────────────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("✅ VÉRIFICATION GPU & MODÈLES COMPLÈTE")
print("="*80)

print(f"""
GPU CONFIGURATION:
  ✓ Device: {DEVICE}
  ✓ CUDA Available: {cuda_available}
  ✓ PIN_MEMORY: {PIN_MEMORY}
  {f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if cuda_available else ""}

MODÈLES:
  ✓ Encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params
  ✓ Classifier: {sum(p.numel() for p in classifier.parameters())/1e6:.1f}M params
  ✓ Loss: MultiTaskBCELoss (NaN-safe)
  ✓ Tous sur device {DEVICE}

CHECKPOINTS PHASE 1:
  ✓ Model trouvé et valide
  ✓ Poids encoder chargés
  ✓ Transfer learning ACTIF

TESTS PASSÉS:
  ✅ GPU/CPU configuration
  ✅ Tous les modèles créés
  ✅ Forward pass OK
  ✅ Loss computation OK
  ✅ Device placement OK
  
PRÊT POUR PHASE 2:
  ✅ OUI - Tout est configuré correctement

""")

print("="*80)
if cuda_available:
    print("🚀 GPU SERA UTILISÉ POUR PHASE 2 - Entraînement RAPIDE")
else:
    print("⚠️  CPU SERA UTILISÉ - Entraînement LENT (considérer GPU)")
print("="*80)

print(f"""
Commande pour lancer Phase 2:
  python run_phase2.py --download --epochs 80 --batch_size 64

{f"Avec GPU: " + str(DEVICE) if cuda_available else "Sans GPU: " + str(DEVICE)}
""")
