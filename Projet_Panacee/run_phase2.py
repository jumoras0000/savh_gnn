# -*- coding: utf-8 -*-
"""
Lanceur Phase 2 - Fine-tuning toxicite.

Usage :
    python run_phase2.py --download                          # telecharge Tox21 + lance
    python run_phase2.py --train_csv data/train.csv --val_csv data/val.csv
    python run_phase2.py --pretrained_model checkpoints/phase1/sovereign_encoder_v1.pth
"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR, PHASE1, PHASE2


def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 2 - Fine-tuning toxicite")
    p.add_argument("--train_csv", type=str, default=None, help="Path to train CSV")
    p.add_argument("--val_csv", type=str, default=None, help="Path to validation CSV")
    p.add_argument("--download", action="store_true", help="Telecharge Tox21 depuis DeepChem")
    p.add_argument("--pretrained_model", type=str, help="Path to Phase 1 checkpoint")
    p.add_argument("--epochs", type=int, default=PHASE2["epochs"], help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=PHASE2["batch_size"], help="Batch size")
    p.add_argument("--smiles_column", type=str, default="smiles", help="SMILES column name")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase2"), help="Save directory")
    p.add_argument("--patience", type=int, default=PHASE2["patience"], help="Early stopping patience")
    args = p.parse_args()

    # --- Etape 0 : Verifications preliminaires ---
    print("="*80)
    print("🚀 PHASE 2 - FINE-TUNING TOXICITE")
    print("="*80)
    
    # Vérifier le modèle pré-entraîné
    pretrained_model = args.pretrained_model or str(CHECKPOINT_DIR / "phase1" / PHASE1["checkpoint_name"])
    if not os.path.exists(pretrained_model):
        print(f"❌ ERREUR: Modèle pré-entraîné introuvable: {pretrained_model}")
        print("   Lancez d'abord Phase 1: python train_phase1.py")
        sys.exit(1)
    print(f"✓ Modèle Phase 1 trouvé: {pretrained_model}")

    # --- Etape 1 : obtenir les CSV ---
    train_csv = args.train_csv
    val_csv = args.val_csv
    
    # Si --download ou pas de CSV fourni
    if args.download or (train_csv is None and val_csv is None):
        print("\n🔄 Téléchargement des données Tox21...")
        from src.preprocessing.toxicity_loader import download_tox21_data
        
        try:
            # Créer le dossier de destination
            tox21_dir = EXTERNAL_DIR / "tox21"
            tox21_dir.mkdir(parents=True, exist_ok=True)
            
            paths = download_tox21_data(str(tox21_dir))
            train_csv = train_csv or paths["train"]
            val_csv = val_csv or paths["val"]
            print(f"✓ Données Tox21 téléchargées")
        except Exception as e:
            print(f"❌ ERREUR lors du téléchargement: {e}")
            print("   Vérifiez que DeepChem est installé: pip install deepchem")
            sys.exit(1)
    
    # Vérifier que les fichiers existent
    if train_csv is None:
        print("❌ ERREUR: Pas de train CSV spécifié ou téléchargé")
        sys.exit(1)
    if val_csv is None:
        print("❌ ERREUR: Pas de val CSV spécifié ou téléchargé")
        sys.exit(1)
    
    train_csv = str(train_csv)
    val_csv = str(val_csv)
    
    if not os.path.exists(train_csv):
        print(f"❌ ERREUR: Train CSV introuvable: {train_csv}")
        sys.exit(1)
    if not os.path.exists(val_csv):
        print(f"❌ ERREUR: Val CSV introuvable: {val_csv}")
        sys.exit(1)
    
    print(f"✓ Train CSV: {train_csv}")
    print(f"✓ Val CSV: {val_csv}")

    # --- Etape 2 : lancer le fine-tuning ---
    print("\n" + "="*80)
    print("▶️ Démarrage du fine-tuning...")
    print("="*80 + "\n")
    
    from src.training.finetune_toxicity import main as ft_main
    
    # Construire les arguments proprement (sans bidouille sys.argv)
    sys_argv_backup = sys.argv.copy()
    try:
        sys.argv = [
            "finetune_toxicity.py",
            "--train_csv", train_csv,
            "--val_csv", val_csv,
            "--pretrained_model", str(pretrained_model),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--smiles_column", args.smiles_column,
            "--save_dir", args.save_dir,
            "--patience", str(args.patience),
        ]
        ft_main()
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = sys_argv_backup


if __name__ == "__main__":
    main()