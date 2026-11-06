# -*- coding: utf-8 -*-
"""
Télécharge des tranches ZINC20 supplémentaires (drug-like, in-stock).

ZINC20 fournit des subsets par propriétés (LogP, MW, charge nette).
On cible les tranches drug-like (MW 200-500, LogP -1 à 5) pour élargir le screening.

Usage :
    python download_zinc_extended.py [--max_mols 500000]
"""
import argparse
import gzip
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


ZINC_TRANCHES_URLS = [
    "https://zinc15.docking.org/tranches/download/AAENL.smi",
    "https://zinc15.docking.org/tranches/download/ABENL.smi",
    "https://zinc15.docking.org/tranches/download/ACENL.smi",
    "https://zinc15.docking.org/tranches/download/ADENL.smi",
    "https://zinc15.docking.org/tranches/download/AEENL.smi",
    "https://zinc15.docking.org/tranches/download/BAENL.smi",
    "https://zinc15.docking.org/tranches/download/BBENL.smi",
    "https://zinc15.docking.org/tranches/download/BCENL.smi",
    "https://zinc15.docking.org/tranches/download/BDENL.smi",
    "https://zinc15.docking.org/tranches/download/BEENL.smi",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_mols", type=int, default=500000)
    ap.add_argument("--out", type=str, default=str(
        Path(PROJECT_ROOT) / "data" / "raw" / "zinc20_druglike_extended.csv"))
    args = ap.parse_args()

    try:
        import urllib.request
    except ImportError:
        print("urllib non disponible")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smiles_set = set()
    print(f"▶ Téléchargement ZINC20 tranches drug-like (max {args.max_mols} mol)")

    for url in ZINC_TRANCHES_URLS:
        if len(smiles_set) >= args.max_mols:
            break
        name = url.split("/")[-1]
        print(f"  {name}...", end=" ", flush=True)
        try:
            resp = urllib.request.urlopen(url, timeout=60)
            data = resp.read()
            if name.endswith(".gz"):
                data = gzip.decompress(data)
            lines = data.decode("utf-8", errors="ignore").strip().split("\n")
            added = 0
            for line in lines:
                if len(smiles_set) >= args.max_mols:
                    break
                parts = line.strip().split()
                if parts:
                    smi = parts[0]
                    if smi not in smiles_set and len(smi) > 3:
                        smiles_set.add(smi)
                        added += 1
            print(f"{added} mol (total: {len(smiles_set)})")
        except Exception as e:
            print(f"ERREUR: {e}")

    # Sauvegarder en CSV
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("smiles\n")
        for smi in smiles_set:
            f.write(f"{smi}\n")

    print(f"\n✅ {len(smiles_set)} molécules → {out_path}")
    print(f"   Pour screener : python screen_library.py --library {out_path}")


if __name__ == "__main__":
    main()
