"""Audit et décodage local de résultats expérimentaux aveugles.

CSV attendu : blind_id,assay_id,replicate,ec50_um,cc50_um.
Valeurs : nombre positif en µM, ou >nombre pour une limite de mesure.
Le script ne génère jamais de données biologiques.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.translational import evaluate_assay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    report = evaluate_assay(args.csv, ROOT)
    directory = ROOT / "data" / "assay_results"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (report["source_sha256"] + ".json")
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise FileExistsError("Rapport déjà enregistré avec un contenu différent")
    output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_measured": report["n_measured"],
                      "n_expected": report["n_expected"], "issues": report["issues"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
