"""
Chargeur de données de toxicité (Tox21 / SIDER) – v2.

Corrections :
  - Calcul automatique de pos_weight pour déséquilibre de classes.
  - Détection robuste de la colonne SMILES et des colonnes de tâches.
  - graph_builder.smiles_to_graph retourne None → filtré automatiquement.
"""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from pathlib import Path
from typing import List, Optional

from src.preprocessing.graph_builder import smiles_to_graph


# ══════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════
_SMILES_CANDIDATES = ["smiles", "SMILES", "ids", "canonical_smiles", "mol", "molecule"]
_EXCLUDE_COLS = {
    "mol_id", "id", "ids", "compound_id", "name",
    *(f"w{i}" for i in range(1, 28)),
}


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class ToxicityDataset(Dataset):
    """
    Dataset multi-tâches pour classification de toxicité.
    Les graphes sont construits une seule fois à l'initialisation.
    """

    def __init__(
        self,
        csv_path: str,
        smiles_column: str = "smiles",
        task_columns: Optional[List[str]] = None,
        max_molecules: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # ── Trouver la colonne SMILES ─────────────────────────────────
        self.smiles_column = self._find_smiles_column(df, smiles_column)

        if max_molecules:
            df = df.head(max_molecules)

        # ── Trouver les colonnes de tâches ────────────────────────────
        self.task_columns = task_columns or self._detect_task_columns(df)
        print(f"  {len(df)} molécules, {len(self.task_columns)} tâches")

        # ── Construire les graphes ────────────────────────────────────
        self.graphs: list = []
        self.labels: list = []
        self._build(df)

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _find_smiles_column(df, hint):
        if hint in df.columns:
            return hint
        for c in _SMILES_CANDIDATES:
            if c in df.columns:
                return c
        raise ValueError(f"Colonne SMILES introuvable parmi {list(df.columns)[:10]}")

    def _detect_task_columns(self, df):
        """Heuristique : colonnes numériques qui ne sont pas SMILES/id/poids."""
        exclude = _EXCLUDE_COLS | {self.smiles_column}
        cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32]
        ]
        return cols

    def _build(self, df):
        invalid = 0
        for _, row in df.iterrows():
            g = smiles_to_graph(row[self.smiles_column])
            if g is None:
                invalid += 1
                continue
            labels = [
                float("nan") if pd.isna(row[c]) else float(row[c])
                for c in self.task_columns
            ]
            self.graphs.append(g)
            self.labels.append(labels)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        if invalid:
            print(f"  {invalid} SMILES invalides ignores")

    # ─────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def get_task_names(self):
        return self.task_columns

    def get_num_tasks(self):
        return len(self.task_columns)

    def get_pos_weight(self) -> torch.Tensor:
        """neg_count / pos_count par tâche (pour BCEWithLogitsLoss)."""
        pw = []
        for t in range(self.labels.shape[1]):
            col = self.labels[:, t]
            valid = col[~torch.isnan(col)]
            pos = (valid == 1).sum().float()
            neg = (valid == 0).sum().float()
            pw.append((neg / pos).item() if pos > 0 else 1.0)
        return torch.tensor(pw, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════════════

def collate_toxicity_batch(batch):
    graphs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════
# Téléchargement (DeepChem)
# ══════════════════════════════════════════════════════════════════════

def _download_dataset(loader_fn, name, save_dir):
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError("pip install deepchem")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Use Raw featurizer to get SMILES directly (not fingerprints)
    tasks, datasets, _ = loader_fn(featurizer="Raw", splitter="scaffold")
    splits = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}

    paths = {}
    for split_name, ds in splits.items():
        # Extract SMILES from X (Raw featurizer may return Mol objects or strings)
        from rdkit import Chem
        smiles_list = []
        for x in ds.X:
            if isinstance(x, Chem.rdchem.Mol):
                smiles_list.append(Chem.MolToSmiles(x))
            elif isinstance(x, str):
                smiles_list.append(x)
            else:
                smiles_list.append(str(x))

        # Build DataFrame with only SMILES + target columns
        df = pd.DataFrame({"smiles": smiles_list})
        labels = ds.y
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        for i, task_name in enumerate(tasks):
            df[task_name] = labels[:, i]

        p = save_path / f"{name}_{split_name}.csv"
        df.to_csv(p, index=False)
        paths[split_name] = str(p)
        print(f"  {split_name}: {p} ({len(df)} mol, {len(tasks)} tasks)")
    return paths


def download_tox21_data(save_dir="data/external/tox21"):
    import deepchem as dc
    return _download_dataset(dc.molnet.load_tox21, "tox21", save_dir)


def download_sider_data(save_dir="data/external/sider"):
    import deepchem as dc
    return _download_dataset(dc.molnet.load_sider, "sider", save_dir)
