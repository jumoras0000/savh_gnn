# -*- coding: utf-8 -*-
"""
Ensemble 5-seed Phase 3 — Deep Ensemble pour gain +0.5-1% AUC + incertitude épistémique.

Lance Phase 3 training N fois avec seeds différents, puis agrège via DeepEnsemble.
Chaque run utilise :
  - Même partitions scaffold global (data_paths.json gelé)
  - Même Phase 2 checkpoint (encodeur commun)
  - Seeds différents pour init random + ordre batch

Checkpoints sauvegardés dans checkpoints/phase3/seed_<N>/

Usage :
    python run_phase3_ensemble.py --n_seeds 5 --epochs 500 --batch_size 128
    python run_phase3_ensemble.py --evaluate_only   # juste évaluation des ckpts existants
"""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def train_seed(seed: int, epochs: int, batch_size: int, base_save_dir: Path):
    """Lance un run Phase 3 avec un seed spécifique."""
    save_dir = base_save_dir / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PANACEE_SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, "run_phase3.py",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--skip_checks",
        "--save_dir", str(save_dir),
    ]
    print(f"\n═══ Seed {seed} ═══")
    print(f"  Commande : {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ⚠ Seed {seed} crashed (code {result.returncode})")
        return None
    ckpt_path = save_dir / "panacee_phase3_complete.pth"
    if not ckpt_path.exists():
        # Try alternative name
        for alt in save_dir.glob("*phase3*.pth"):
            ckpt_path = alt
            break
    if ckpt_path.exists():
        print(f"  ✓ Seed {seed} → {ckpt_path}")
        return str(ckpt_path)
    print(f"  ⚠ Seed {seed} : checkpoint introuvable dans {save_dir}")
    return None


def evaluate_ensemble(checkpoint_paths: list[str], device_str: str = None):
    """Évalue l'ensemble en moyennant les prédictions + reporte std."""
    import numpy as np
    import pandas as pd
    import torch

    from src.config import CHECKPOINT_DIR, EXTERNAL_DIR, PHASE3
    from src.models.encoder import MolecularEncoder
    from src.models.ensemble import DeepEnsemble
    from src.models.multi_property_head import MultiPropertyPredictor
    from src.preprocessing.graph_builder import smiles_to_graph
    from src.utils.safe_load import safe_load_checkpoint
    from src.validation.pharma_metrics import full_screening_report
    from src.validation.drugbank_antiretroviral import (
        canonicalize_arv_smiles, validate_model_recall,
    )
    from torch_geometric.data import Batch

    device = torch.device(device_str) if device_str else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n═══ Évaluation ensemble ({len(checkpoint_paths)} modèles) sur {device} ═══")

    def _factory():
        encoder = MolecularEncoder()
        return MultiPropertyPredictor(encoder, PHASE3["property_heads"], hidden_dim=256)

    # Charger ensemble
    ensemble = DeepEnsemble.from_checkpoints(
        checkpoint_paths, model_factory=_factory, device=str(device),
    )

    # Charger HIV test
    data_paths_file = EXTERNAL_DIR / "phase3" / "data_paths.json"
    with open(data_paths_file) as f:
        data_paths = json.load(f)
    hiv_test = pd.read_csv(data_paths["hiv"]["test"])
    hiv_train = pd.read_csv(data_paths["hiv"]["train"])

    # Prédictions ensemble batch
    all_mean = []
    all_std = []
    valid_smi = []
    batch_size = 128
    smiles_list = hiv_test["smiles"].tolist()
    for i in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[i:i + batch_size]
        graphs, ok_mask = [], []
        for s in chunk:
            try:
                g = smiles_to_graph(s)
                if g is not None:
                    graphs.append(g); ok_mask.append(True)
                else:
                    ok_mask.append(False)
            except Exception:
                ok_mask.append(False)
        if not graphs:
            continue
        batch = Batch.from_data_list(graphs).to(device)
        means, stds = ensemble.predict(batch, apply_sigmoid_to_classif=True)
        all_mean.append(means["efficacy"].cpu().numpy().ravel())
        all_std.append(stds["efficacy"].cpu().numpy().ravel())
        for j, ok in enumerate(ok_mask):
            if ok:
                valid_smi.append(chunk[j])

    y_proba = np.concatenate(all_mean)
    y_std = np.concatenate(all_std)
    y_true = hiv_test[hiv_test["smiles"].isin(valid_smi)]["HIV_active"].values

    print(f"\n  Prédictions ensemble : {len(y_proba)} molécules HIV test")
    print(f"  Std épistémique moyenne : {y_std.mean():.4f}")
    print(f"  Std épistémique max     : {y_std.max():.4f}")

    # Pharma report
    pharma = full_screening_report(
        y_true, y_proba,
        smiles=valid_smi,
        training_actives=hiv_train[hiv_train["HIV_active"] == 1]["smiles"].tolist(),
    )
    print("\n  Métriques ensemble :")
    for k, v in pharma.items():
        if isinstance(v, (int, float, type(None))):
            print(f"    {k}: {v}")

    # DrugBank validation
    arvs = canonicalize_arv_smiles()
    arv_smi_list = [d["smiles_canonical"] for d in arvs]
    arv_graphs = []
    valid_arv = []
    for s in arv_smi_list:
        try:
            g = smiles_to_graph(s)
            if g is not None:
                arv_graphs.append(g); valid_arv.append(s)
        except Exception:
            pass
    if arv_graphs:
        batch = Batch.from_data_list(arv_graphs).to(device)
        means_arv, _ = ensemble.predict(batch, apply_sigmoid_to_classif=True)
        arv_proba = means_arv["efficacy"].cpu().numpy().ravel()
        combined = pd.DataFrame({
            "smiles": list(valid_arv) + valid_smi,
            "hiv_score": np.concatenate([arv_proba, y_proba]),
        })
        gate = validate_model_recall(combined, min_recall=25)
        print(f"\n  DrugBank gate ensemble : {gate['verdict']} "
              f"({gate['recall']}/{gate['total']} ARV retrouvés top-10%)")

    # Sauvegarde rapport
    report = {
        "n_models": len(checkpoint_paths),
        "checkpoints": checkpoint_paths,
        "pharma_hiv_test": pharma,
        "epistemic_uncertainty": {
            "mean_std": float(y_std.mean()),
            "max_std": float(y_std.max()),
            "median_std": float(np.median(y_std)),
        },
        "drugbank_gate": gate if arv_graphs else None,
    }
    out = Path("phase3_ensemble_report.json")
    with open(out, "w", encoding="utf-8") as f:
        def _d(o):
            import numpy as _np
            if isinstance(o, (_np.integer, _np.floating)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return str(o)
        json.dump(report, f, indent=2, default=_d, ensure_ascii=False)
    print(f"\n✅ Rapport ensemble : {out}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Ensemble 5-seed Phase 3")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=500,
                   help="Epochs par seed (réduit vs run principal pour accélérer)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--base_save_dir", type=str,
                   default=str(Path("checkpoints") / "phase3_ensemble"))
    p.add_argument("--evaluate_only", action="store_true",
                   help="Skip training, évalue juste les ckpts existants")
    args = p.parse_args()

    base_dir = Path(args.base_save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    if args.evaluate_only:
        for seed in range(args.n_seeds):
            for ckpt in (base_dir / f"seed_{seed}").glob("*.pth"):
                if "phase3" in ckpt.name.lower():
                    checkpoints.append(str(ckpt))
                    break
    else:
        for seed in range(args.n_seeds):
            ckpt = train_seed(seed, args.epochs, args.batch_size, base_dir)
            if ckpt:
                checkpoints.append(ckpt)

    print(f"\n═══ Checkpoints collectés : {len(checkpoints)}/{args.n_seeds} ═══")
    for c in checkpoints:
        print(f"  • {c}")

    if len(checkpoints) < 2:
        print("\n⚠ Pas assez de checkpoints pour un ensemble (minimum 2)")
        sys.exit(1)

    evaluate_ensemble(checkpoints)


if __name__ == "__main__":
    main()
