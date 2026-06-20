#!/usr/bin/env python3
"""
🚀 LANCEUR PHASE 2 - GUIDE RAPIDE
Exécute: python run_phase2_guided.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    ✨ PHASE 2 - FINE-TUNING TOXICITÉ ✨                   ║
║                                                                            ║
║  Tous les bugs critiques sont corrigés ✅                                 ║
║  Toutes les optimisations sont appliquées ✅                              ║
║  Vous êtes prêt à entraîner! 🚀                                           ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_corrections():
    corrections = """
📋 CORRECTIONS APPLIQUÉES:

[CRITIQUE] ✅ Message Passing GNN
  → Utilise x_i ET x_j (fix: torch.cat([x_i, x_j, edge_attr]))
  
[CRITIQUE] ✅ MultiTaskBCELoss
  → Gère NaN + pos_weight pour imbalance de classes
  
[CRITIQUE] ✅ Class Balancing
  → get_pos_weight() calcule poids par tâche (neg/pos)

[IMPORTANT] ✅ Learning Rates
  → Différencies: encoder=1e-4, head=1e-3
  → Warmup: 5 epochs linéaire
  → Decay: Cosine annealing
  
[IMPORTANT] ✅ Early Stopping
  → Based on ROC-AUC moyen
  → Patience: 20 epochs
  
[IMPORTANT] ✅ Threshold Tuning
  → Recherche seuil optimal par tâche (max F1)
  
[BON À AVOIR] ✅ Gradual Unfreezing
  → 10 epochs encoder gelé, puis dégel progressif
  
[BON À AVOIR] ✅ Architecture Robuste
  → SiLU activation, Dropout 0.2, Skip connections
  → BatchNorm + Triple pooling dans encoder

"""
    print(corrections)


def print_commands():
    commands = """
💻 COMMANDES POUR LANCER:

[1] Simple et rapide (RECOMANDÉ):
    python run_phase2.py --download --epochs 80 --batch_size 64

[2] Personnalisé:
    python run_phase2.py --download \\
        --epochs 100 \\
        --batch_size 32 \\
        --patienece 25

[3] Avec données locales:
    python run_phase2.py \\
        --train_csv data/external/tox21/tox21_train.csv \\
        --val_csv data/external/tox21/tox21_val.csv \\
        --epochs 80

[4] Tester avec petit dataset:
    python run_phase2.py --download --epochs 5 --batch_size 32

"""
    print(commands)


def print_hyperparams():
    hyperparams = """
⚙️ HYPERPARAMÈTRES PHASE 2:

Entraînement:
  • Epochs: 80 (default)
  • Batch size: 64
  • LR encoder: 1e-4 (pré-entraîné = lent)
  • LR head: 1e-3 (nouveau = rapide)
  • Warmup: 5 epochs
  • Weight decay: 1e-4
  • Grad clip: 1.0
  • Patience: 20 epochs
  • Freeze encoder: 10 epochs

Architecture:
  • Hidden dim: 256
  • GNN layers: 6
  • Dropout: 0.2
  • Output: 256-dim par molécule
  
Tasks:
  • Tox21: 12 tâches
  • Molécules: ~8000 train, ~1000 val

"""
    print(hyperparams)


def print_expected_results():
    results = """
📈 RÉSULTATS ATTENDUS:

Après 80 epochs (30-40 min sur GPU):
  • Loss validation: 0.15 - 0.25
  • ROC-AUC moyen: 0.78 - 0.85
  • F1-Score moyen: 0.65 - 0.75
  • Temps par epoch: 2-5 min (GPU), 10-15 min (CPU)

Checkpoints sauvegardés:
  • best_toxicity_model.pth → Meilleur ROC-AUC
  • checkpoint_latest.pth → Dernier checkpoint
  • Seuils optimaux → Trouvés automatiquement

Logs:
  • logs/phase2/finetune_YYYYMMDD_HHMMSS.json

"""
    print(results)


def print_troubleshooting():
    troubleshooting = """
🔧 TROUBLESHOOTING:

❌ CUDA out of memory (OOM)?
  → Réduisez batch_size: --batch_size 32
  
❌ Entraînement très lent?
  → Vérifiez GPU: python -c "import torch; print(torch.cuda.is_available())"
  
❌ Loss = NaN?
  → La MultiTaskBCELoss doit gérer ça automatiquement
  → Sinon: baissez LR à 1e-5
  
❌ Pas de progrès après 20 epochs?
  → Normal! Early stopping va arrêter
  → Patience=20 epochs est paramétrable
  
❌ Module not found?
  → pip install -r requirements.txt
  
❌ SMILES parsing error?
  → Vérifiez colonne 'smiles' dans CSV
  → Utilisez --smiles_column si différent

"""
    print(troubleshooting)


def print_next_steps():
    next_steps = """
📅 PROCHAINES ÉTAPES (après Phase 2):

Phase 3: Drug-Drug Interactions
  • Dataset: TWOSIDES
  • Prédire interactions entre paires de drugs
  • Utiliser embeddings Phase 2

Phase 4: Validation Physique
  • DiffDock: molecular docking
  • OpenFold: structure de protéine
  • Validation expérimentale

"""
    print(next_steps)


def main():
    print_banner()
    print_corrections()
    print_commands()
    print_hyperparams()
    print_expected_results()
    print_troubleshooting()
    print_next_steps()
    
    print("\n" + "="*80)
    print("✅ READY TO LAUNCH PHASE 2!")
    print("="*80)
    print("\nCopiez la commande ci-dessus et exécutez-la!\n")


if __name__ == "__main__":
    main()
