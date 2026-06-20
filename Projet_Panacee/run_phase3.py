# -*- coding: utf-8 -*-
"""
Lanceur Phase 3 - Entraînement multi-propriétés + IA raisonnement.

Usage :
    python run_phase3.py --download                    # Télécharge tous les datasets + lance
    python run_phase3.py --download --epochs 50        # Avec epochs personnalisés
    python run_phase3.py --epochs 100 --batch_size 16  # Sans re-télécharger
    python run_phase3.py --pretrained_model checkpoints/phase2/best_toxicity_model.pth

Pré-requis :
    - Phase 1 terminée (sovereign_encoder_v1.pth)
    - Phase 2 terminée (best_toxicity_model.pth)
    - DeepChem installé (pip install deepchem)
"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINT_DIR, PHASE3


def check_prerequisites():
    """Vérifie que la Phase 2 est terminée (Phase 3 part de l'encodeur Phase 2).

    NB : la Phase 1 n'est PAS requise — la Phase 2 fournit l'encodeur fine-tuné,
    qu'elle ait été pré-entraînée (Phase 1) ou non.
    """
    errors = []

    phase2_ckpt = CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth"

    if not phase2_ckpt.exists():
        errors.append(f"  ✗ Phase 2 manquante : {phase2_ckpt}")
        errors.append("    → Lancez : python run_phase2.py --download")

    return errors


def check_dependencies():
    """Vérifie que les dépendances sont installées (via find_spec, sans importer)."""
    import importlib.util as _u
    errors = []

    if _u.find_spec("torch") is None:
        errors.append("  ✗ PyTorch non installé")
    else:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠ CUDA non disponible, entraînement sur CPU (plus lent)")

    if _u.find_spec("torch_geometric") is None:
        errors.append("  ✗ torch-geometric non installé")
    if _u.find_spec("deepchem") is None:
        errors.append("  ✗ DeepChem non installé (pip install deepchem)")
    if _u.find_spec("sklearn") is None:
        errors.append("  ✗ scikit-learn non installé")

    return errors


def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 3 - Multi-propriétés + IA Raisonnement")
    p.add_argument("--download", action="store_true",
                   help="Télécharge tous les datasets (Tox21, ESOL, Lipo, BBBP, ClinTox, HIV)")
    p.add_argument("--pretrained_model", type=str, default=None,
                   help="Chemin du checkpoint Phase 2")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Dossier des données Phase 3")
    p.add_argument("--epochs", type=int, default=PHASE3["epochs"],
                   help=f"Nombre d'epochs (défaut: {PHASE3['epochs']})")
    p.add_argument("--batch_size", type=int, default=PHASE3["batch_size"],
                   help=f"Taille de batch (défaut: {PHASE3['batch_size']})")
    p.add_argument("--patience", type=int, default=PHASE3["patience"],
                   help=f"Patience early stopping (défaut: {PHASE3['patience']})")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase3"),
                   help="Dossier de sauvegarde")
    p.add_argument("--skip_checks", action="store_true",
                   help="Ignorer les vérifications de pré-requis")
    args = p.parse_args()

    # ── Bannière ──
    print("=" * 80)
    print("🧠 PANACÉE – PHASE 3 : MULTI-PROPRIÉTÉS + IA RAISONNEMENT")
    print("=" * 80)
    print()
    print("  Cette phase entraîne un modèle qui :")
    print("  1. Prédit la toxicité (12 tâches Tox21)")
    print("  2. Prédit l'efficacité anti-VIH")
    print("  3. Prédit la solubilité (LogS)")
    print("  4. Prédit la lipophilicité (LogP)")
    print("  5. Prédit la biodisponibilité")
    print("  6. Prédit la stabilité métabolique")
    print("  7. Analyse les combinaisons de molécules (IA Raisonnement)")
    print("  8. Calcule les doses optimales et scores de synergie")
    print()

    # ── Vérifications ──
    if not args.skip_checks:
        print("🔍 Vérification des pré-requis...")

        dep_errors = check_dependencies()
        if dep_errors:
            print("❌ Dépendances manquantes :")
            for err in dep_errors:
                print(err)
            sys.exit(1)
        print("  ✓ Dépendances OK")

        prereq_errors = check_prerequisites()
        if prereq_errors:
            print("❌ Pré-requis manquants :")
            for err in prereq_errors:
                print(err)
            sys.exit(1)
        print("  ✓ Pré-requis OK")

    # ── Lancer l'entraînement ──
    print()
    print("▶️ Démarrage de l'entraînement Phase 3...")
    print()

    from src.training.train_phase3 import main as phase3_main

    # Construire les arguments pour train_phase3
    sys_argv_backup = sys.argv.copy()
    try:
        sys_args = [
            "train_phase3.py",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--patience", str(args.patience),
            "--save_dir", args.save_dir,
        ]

        if args.download:
            sys_args.append("--download")

        if args.pretrained_model:
            sys_args.extend(["--pretrained_model", args.pretrained_model])

        if args.data_dir:
            sys_args.extend(["--data_dir", args.data_dir])

        sys.argv = sys_args
        phase3_main()

    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = sys_argv_backup

    # ── Message de fin ──
    print()
    print("=" * 80)
    print("✅ PHASE 3 TERMINÉE")
    print("=" * 80)
    print()
    print("  Modèle sauvegardé : checkpoints/phase3/panacee_phase3_complete.pth")
    print()
    print("  Pour utiliser le modèle :")
    print("    python predict_molecules.py --smiles \"CCO,CC(=O)O\"")
    print("    python predict_molecules.py --smiles_file molecules.csv")
    print("    python predict_molecules.py --smiles_file molecules.csv --report rapport.json")


if __name__ == "__main__":
    main()
