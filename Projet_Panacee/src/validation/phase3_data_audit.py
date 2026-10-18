"""Read-only audit of Phase 3 split composition and label availability.

Run: python -m src.validation.phase3_data_audit --data_dir data/external/phase3
The report contains counts only, not molecular identities or model predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.multi_property_loader import (
    _chemical_key,
    verify_global_split_integrity,
)


CLASSIFICATION_DATASETS = {"tox21", "bbbp", "clintox", "hiv"}
SPLITS = ("train", "val", "test")


def audit_phase3_data(data_paths):
    """Inspect per-task coverage, prevalence, and the empty-scaffold artifact."""
    integrity = verify_global_split_integrity(data_paths)
    result = {"integrity": integrity, "datasets": {},
              "regression_baselines": {},
              "limitations": [
                  "Phase 2 encoder exposure to validation/test is not audited here",
                  "Identical empty Murcko scaffolds group acyclic molecules in one split",
              ]}
    for name, paths in sorted(data_paths.items()):
        result["datasets"][name] = {}
        for split in SPLITS:
            frame = pd.read_csv(paths[split])
            empty_scaffolds = sum(_chemical_key(s)[1] == "" for s in frame["smiles"])
            columns = {}
            for column in frame.columns:
                if column in {"smiles", "source_smiles", "__source_row", "__source_id"}:
                    continue
                parsed = pd.to_numeric(frame[column], errors="coerce")
                raw = parsed.to_numpy(dtype=float)
                if (frame[column].notna() & ~np.isfinite(raw)).any():
                    raise ValueError(f"{name}/{split}/{column}: étiquette non numérique ou non finie")
                observed = raw[np.isfinite(raw)]
                stats = {"observed": int(len(observed)),
                         "missing": int(len(raw) - len(observed))}
                if name in CLASSIFICATION_DATASETS:
                    if len(observed) and not np.isin(observed, (0, 1)).all():
                        raise ValueError(f"{name}/{split}/{column}: étiquettes non binaires")
                    positives = int(observed.sum())
                    stats.update({"positive": positives,
                                  "negative": int(len(observed) - positives),
                                  "prevalence": (positives / len(observed)
                                                 if len(observed) else None)})
                else:
                    stats.update({"mean": float(observed.mean()) if len(observed) else None,
                                  "std": float(observed.std()) if len(observed) else None})
                columns[column] = stats
            result["datasets"][name][split] = {
                "molecules": int(len(frame)),
                "empty_scaffold_molecules": int(empty_scaffolds),
                "labels": columns,
            }
        if name not in CLASSIFICATION_DATASETS:
            train = pd.read_csv(paths["train"])
            test = pd.read_csv(paths["test"])
            for column in result["datasets"][name]["train"]["labels"]:
                y_train = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
                y_test = pd.to_numeric(test[column], errors="coerce").to_numpy(dtype=float)
                y_train, y_test = y_train[np.isfinite(y_train)], y_test[np.isfinite(y_test)]
                if not len(y_train) or not len(y_test):
                    continue
                baseline = float(y_train.mean())
                mse = float(np.mean((y_test - baseline) ** 2))
                test_variance = float(np.var(y_test))
                result["regression_baselines"][name] = {
                    "column": column,
                    "train_mean_test_rmse": float(np.sqrt(mse)),
                    "train_mean_test_r2": (float(1 - mse / test_variance)
                                           if test_variance > 1e-12 else None),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit read-only des partitions Phase 3")
    parser.add_argument("--data_dir", default="data/external/phase3")
    args = parser.parse_args()
    paths_file = Path(args.data_dir) / "data_paths.json"
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    print(json.dumps(audit_phase3_data(paths), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
