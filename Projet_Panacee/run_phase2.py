# -*- coding: utf-8 -*-
"""
Lanceur Phase 2 - Fine-tuning toxicite.

Usage :
    python run_phase2.py --download                              # telecharge Tox21 + lance
    python run_phase2.py --download --run_name kaggle_run01      # nom du run (Kaggle/dashboard)
    python run_phase2.py --train_csv data/train.csv --val_csv data/val.csv
    python run_phase2.py --pretrained_model checkpoints/phase1/sovereign_encoder_v1.pth

Sur Kaggle, définir PANACEE_PUSH_URL + PANACEE_PUSH_TOKEN avant de lancer :
    import os
    os.environ["PANACEE_PUSH_URL"]   = "https://xxxx.ngrok.io"
    os.environ["PANACEE_PUSH_TOKEN"] = "mon_token_secret"
"""
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'
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
    p.add_argument("--run_name", type=str, default=None,
                   help="Nom du run (dashboard + dossier checkpoints). "
                        "Prend le dessus sur --save_dir et fixe PANACEE_PUSH_RUN.")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase2"), help="Save directory")
    p.add_argument("--patience", type=int, default=PHASE2["patience"], help="Early stopping patience")
    p.add_argument("--max_molecules", type=int, default=None, help="Limite de molecules (runs rapides Kaggle)")
    p.add_argument("--cv_folds", type=int, default=0, help="Cross-validation scaffold (0=split simple)")
    p.add_argument("--ema", type=int, default=1, help="1=EMA des poids actif, 0=desactive")
    args = p.parse_args()

    # --run_name : identifiant du run dans le dashboard (LiveLogger PANACEE_PUSH_RUN).
    # NE change PAS save_dir → Phase 3 trouve toujours checkpoints/phase2/best_toxicity_model.pth
    if args.run_name:
        safe_name = "".join(c for c in args.run_name if c.isalnum() or c in ("-", "_"))
        os.environ.setdefault("PANACEE_PUSH_RUN", safe_name)

    # --- Etape 0 : Verifications preliminaires ---
    print("="*80)
    print("🚀 PHASE 2 - FINE-TUNING TOXICITE")
    print("="*80)
    push_url = os.environ.get("PANACEE_PUSH_URL", "")
    push_run = os.environ.get("PANACEE_PUSH_RUN", "")
    if push_url:
        print(f"  📡 Push Kaggle → {push_url}  (run={push_run or 'auto'})")
    else:
        print("  💾 Entraînement local (pas de push distant configuré)")

    # Modèle pré-entraîné Phase 1 : OPTIONNEL.
    # S'il est absent, on fine-tune un encodeur initialisé aléatoirement.
    # Cela permet à la Phase 2 de tourner seule sur Kaggle (SKIP_PHASE1=True).
    pretrained_model = args.pretrained_model or str(CHECKPOINT_DIR / "phase1" / PHASE1["checkpoint_name"])
    if os.path.exists(pretrained_model):
        print(f"✓ Modèle Phase 1 trouvé: {pretrained_model}")
    else:
        print(f"⚠️  Pas de checkpoint Phase 1 ({pretrained_model})")
        print("   → Fine-tuning depuis un encodeur aléatoire (Phase 2 autonome).")
        print("   → Pour de meilleurs résultats, lance d'abord: python run_phase1.py --download")
        pretrained_model = None

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
            print("✓ Données Tox21 téléchargées")
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
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--smiles_column", args.smiles_column,
            "--save_dir", args.save_dir,
            "--patience", str(args.patience),
        ]
        if pretrained_model is not None:
            sys.argv += ["--pretrained_model", str(pretrained_model)]
        if args.max_molecules is not None:
            sys.argv += ["--max_molecules", str(args.max_molecules)]
        sys.argv += ["--cv_folds", str(args.cv_folds), "--ema", str(args.ema)]
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
