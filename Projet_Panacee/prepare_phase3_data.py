# -*- coding: utf-8 -*-
"""
Préparation déterministe des données Phase 3 — sans training.

Reproduit l'étape de téléchargement/scaffold-split global que `train_phase3.py`
fait en interne quand on passe `--download`, mais isolée pour permettre à
Phase 2 de s'entraîner UNIQUEMENT sur les SMILES du train Phase 3 (évite la
fuite supervisée détectée par `audit_phase2_exposure`).

Pipeline rigoureux :
    1. python prepare_phase3_data.py             → crée splits + data_paths.json + split_manifest.json
    2. python run_phase2.py --phase3_data_dir data/external/phase3  → Phase 2 sur train Phase 3
    3. python run_phase3.py --skip_checks        → Phase 3 complet (pas de re-download)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_DIR  # noqa: E402
from src.preprocessing.multi_property_loader import (  # noqa: E402
    download_all_phase3_data,
    verify_global_split_integrity,
    verify_phase3_source_provenance,
)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Prépare les splits globaux Phase 3 (sans training)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Dossier cible (défaut : data/external/phase3)")
    p.add_argument("--force", action="store_true",
                   help="Recrée les splits même si data_paths.json existe déjà")
    args = p.parse_args()

    data_dir = args.data_dir or str(EXTERNAL_DIR / "phase3")
    paths_file = os.path.join(data_dir, "data_paths.json")

    if os.path.exists(paths_file) and not args.force:
        print(f"✓ data_paths.json existe déjà : {paths_file}")
        print("  (passer --force pour re-télécharger)")
        with open(paths_file, "r") as f:
            data_paths = json.load(f)
    else:
        print("=" * 70)
        print("📥 Téléchargement + scaffold split global Phase 3")
        print("=" * 70)
        os.makedirs(data_dir, exist_ok=True)
        data_paths = download_all_phase3_data(data_dir)

        serializable = {k: v for k, v in data_paths.items() if v is not None}
        with open(paths_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n✓ data_paths.json → {paths_file}")

    # Vérifications d'intégrité (identique à ce que train_phase3.py exige)
    print("\n🔍 Vérifications d'intégrité...")
    audit = verify_global_split_integrity(data_paths)
    print(f"  ✓ Pas de fuite de scaffold : {audit['counts']}")

    manifest_path = Path(data_dir) / "split_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifeste absent : {manifest_path}. "
            "download_all_phase3_data devrait l'avoir créé. "
            "Vérifier src/preprocessing/multi_property_loader.repartition_global_scaffolds"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_audit = verify_phase3_source_provenance(manifest, data_dir)
    print(f"  ✓ Provenance quarantaine : {source_audit}")

    print()
    print("=" * 70)
    print("✅ PHASE 3 DATA PRÊTE")
    print("=" * 70)
    print(f"\nFichiers :")
    for name, parts in data_paths.items():
        if parts:
            for split, path in parts.items():
                print(f"  {name}/{split}: {path}")
    print(f"\n  data_paths.json      : {paths_file}")
    print(f"  split_manifest.json  : {manifest_path}")
    print()
    print("Prochaines étapes :")
    print(f"  python run_phase2.py --phase3_data_dir {data_dir} --epochs 800 --batch_size 128")
    print(f"  python run_phase3.py --epochs 3000 --batch_size 128 --skip_checks")


if __name__ == "__main__":
    main()
