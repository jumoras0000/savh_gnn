# -*- coding: utf-8 -*-
"""
Lanceur Phase 1 - Pre-entrainement MGM (Masked Graph Modeling) sur ZINC.

Pipeline :
    1. Recupere le CSV ZINC (local s'il existe, sinon telechargement auto).
    2. Canonise / deduplique -> data/processed/pretrain_dataset.pt
    3. Lance le pre-entra, sauvegarde checkpoints/phase1/sovereign_encoder_v1.pth

Usage :
    python run_phase1.py --download                       # telecharge ZINC + lance
    python run_phase1.py --download --max_molecules 50000 # subset (rapide Kaggle)
    python run_phase1.py --zinc_csv chemin/vers/zinc.csv  # CSV local
"""
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINT_DIR, PHASE1, PROCESSED_DIR, RAW_DIR

ZINC_FILENAME = "250k_rndm_zinc_drugs_clean_3.csv"

# Mirrors publics du dataset ZINC 250k (colonne "smiles")
ZINC_URLS = [
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv",
]


def _find_local_zinc():
    """Cherche le CSV ZINC dans les emplacements connus."""
    candidates = [
        PROJECT_ROOT / ZINC_FILENAME,
        PROJECT_ROOT.parent / ZINC_FILENAME,   # racine du repo (dev local)
        RAW_DIR / ZINC_FILENAME,
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def _download_zinc(dest: Path):
    """Telecharge le CSV ZINC depuis les mirrors publics."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in ZINC_URLS:
        try:
            print(f"  Telechargement depuis {url} ...")
            urllib.request.urlretrieve(url, dest)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  OK -> {dest} ({dest.stat().st_size/1e6:.1f} Mo)")
                return dest
        except Exception as e:
            last_err = e
            print(f"  echec ({e})")
    raise RuntimeError(f"Impossible de telecharger ZINC. Derniere erreur: {last_err}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 1 - Pre-entrainement MGM (ZINC)")
    p.add_argument("--download", action="store_true",
                   help="Telecharge ZINC si absent localement")
    p.add_argument("--zinc_csv", type=str, default=None,
                   help="Chemin d'un CSV ZINC local (colonne 'smiles')")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase1"))
    p.add_argument("--epochs", type=int, default=PHASE1["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE1["batch_size"])
    p.add_argument("--lr", type=float, default=PHASE1["lr"])
    p.add_argument("--mask_prob", type=float, default=PHASE1["mask_prob"])
    p.add_argument("--patience", type=int, default=PHASE1["patience"])
    p.add_argument("--max_molecules", type=int, default=None,
                   help="Limite de molecules (runs rapides Kaggle)")
    p.add_argument("--objective", type=str, default="mgm", choices=["mgm", "graphcl"],
                   help="mgm = masked graph modeling | graphcl = pre-entrainement contrastif")
    args = p.parse_args()

    print("=" * 80)
    print("🚀 PHASE 1 - PRE-ENTRAINEMENT MGM (ZINC)")
    print("=" * 80)

    # --- Etape 1 : obtenir le CSV ZINC ---
    if args.zinc_csv:
        zinc_csv = Path(args.zinc_csv)
        if not zinc_csv.exists():
            print(f"❌ CSV ZINC introuvable: {zinc_csv}")
            sys.exit(1)
    else:
        zinc_csv = _find_local_zinc()
        if zinc_csv is None:
            if args.download:
                print("\n🔄 ZINC absent localement, telechargement...")
                try:
                    zinc_csv = _download_zinc(RAW_DIR / ZINC_FILENAME)
                except Exception as e:
                    print(f"❌ {e}")
                    sys.exit(1)
            else:
                print("❌ CSV ZINC introuvable. Relance avec --download ou --zinc_csv <path>.")
                sys.exit(1)
    print(f"✓ ZINC CSV: {zinc_csv}")

    # --- Etape 2 : preparer le dataset (.pt) ---
    print("\n🔧 Preparation du dataset (canonisation + deduplication)...")
    from src.preprocessing.zinc_loader import process_zinc_dataset

    pretrain_pt = PROCESSED_DIR / "pretrain_dataset.pt"
    try:
        process_zinc_dataset(str(zinc_csv), str(pretrain_pt), args.max_molecules)
    except Exception as e:
        print(f"❌ Erreur preparation ZINC: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"✓ Dataset pret: {pretrain_pt}")

    # --- Etape 3 : lancer le pre-entrainement ---
    print("\n" + "=" * 80)
    print("▶️ Demarrage du pre-entrainement MGM...")
    print("=" * 80 + "\n")

    from src.training.pretrain_gnn import main as pretrain_main

    sys_argv_backup = sys.argv.copy()
    try:
        sys.argv = [
            "pretrain_gnn.py",
            "--data_path", str(pretrain_pt),
            "--save_dir", args.save_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--mask_prob", str(args.mask_prob),
            "--patience", str(args.patience),
            "--objective", args.objective,
        ]
        if args.max_molecules is not None:
            sys.argv += ["--max_molecules", str(args.max_molecules)]
        pretrain_main()
    except KeyboardInterrupt:
        print("\n⚠️ Entrainement interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERREUR lors du pre-entrainement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = sys_argv_backup

    print("\n" + "=" * 80)
    print("✅ PHASE 1 TERMINEE")
    print(f"   Checkpoint: {Path(args.save_dir) / PHASE1['checkpoint_name']}")
    print("   Etape suivante: python run_phase2.py --download")
    print("=" * 80)


if __name__ == "__main__":
    main()
