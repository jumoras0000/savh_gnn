"""Audit de prédictions Phase 4 sur un jeu test externe (CSV figé)."""
from __future__ import annotations

import argparse
import json

from src.validation.phase4_audit import audit_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV avec smiles,y_true,predicted_probability,split")
    parser.add_argument("--threshold", required=True, type=float, help="Seuil prédéfini avant test")
    parser.add_argument("--endpoint", required=True, help="Ex. HIV_active ou NR-AR")
    args = parser.parse_args()
    print(json.dumps(audit_csv(args.csv, threshold=args.threshold, endpoint=args.endpoint),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
