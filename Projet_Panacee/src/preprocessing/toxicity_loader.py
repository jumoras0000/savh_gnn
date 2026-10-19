"""
Chargeur de données de toxicité (Tox21 / SIDER) – v2.

Corrections :
  - Calcul automatique de pos_weight pour déséquilibre de classes.
  - Détection robuste de la colonne SMILES et des colonnes de tâches.
  - graph_builder.smiles_to_graph retourne None → filtré automatiquement.
"""
import sys

# Console Windows en UTF-8, SANS remplacer sys.stdout (ce qui casserait la
# capture de pytest et toute redirection). reconfigure() n'existe que sur le
# vrai TextIOWrapper d'une console : sous capture/pipe, l'AttributeError est
# ignorée (l'encodage est de toute façon géré via PYTHONIOENCODING).
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

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

        if max_molecules is not None:
            if max_molecules < 1:
                raise ValueError("max_molecules doit être positif")
            df = df.sample(n=min(max_molecules, len(df)), random_state=42)

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
            if any(not np.isnan(value) and value not in (0.0, 1.0) for value in labels):
                raise ValueError("Phase 2 : étiquette non binaire ou non finie")
            self.graphs.append(g)
            self.labels.append(labels)

        if not self.graphs or not self.task_columns:
            raise ValueError("Phase 2 : aucun graphe valide ou aucune tâche")
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
    graphs, labels = zip(*batch, strict=False)
    return Batch.from_data_list(graphs), torch.stack(labels)


# ══════════════════════════════════════════════════════════════════════
# Téléchargement (direct CSV — bypass DeepChem/tensorflow)
# ══════════════════════════════════════════════════════════════════════

_TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

_TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def _scaffold_split(df, frac_train=0.8, frac_val=0.1, seed=42):
    """Scaffold split via RDKit — fallback random si RDKit échoue."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffolds: dict = {}
        for i, smi in enumerate(df["smiles"]):
            try:
                mol = Chem.MolFromSmiles(smi)
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = smi
            scaffolds.setdefault(scaffold, []).append(i)

        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
        n = len(df)
        train_cutoff = int(frac_train * n)
        val_cutoff = int((frac_train + frac_val) * n)

        train_idx, val_idx, test_idx = [], [], []
        for group in scaffold_sets:
            if len(train_idx) < train_cutoff:
                train_idx.extend(group)
            elif len(train_idx) + len(val_idx) < val_cutoff:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        return (df.iloc[train_idx].reset_index(drop=True),
                df.iloc[val_idx].reset_index(drop=True),
                df.iloc[test_idx].reset_index(drop=True))

    except Exception as e:
        print(f"  Scaffold split échoué ({e}), fallback random split")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        t, v = int(0.8 * n), int(0.9 * n)
        return df.iloc[:t], df.iloc[t:v], df.iloc[v:]


def download_tox21_data(save_dir="data/external/tox21"):
    """Télécharge Tox21 depuis S3 et fait un scaffold split — sans DeepChem."""
    import urllib.request, gzip, shutil

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    raw_csv = save_path / "tox21_raw.csv"

    # Cherche CSV déjà présent (téléchargé manuellement ou précédemment)
    if not raw_csv.exists():
        gz_path = save_path / "tox21.csv.gz"
        print(f"  Téléchargement Tox21 depuis {_TOX21_URL} ...")
        urllib.request.urlretrieve(_TOX21_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(raw_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)
        print(f"  OK → {raw_csv}")

    df = pd.read_csv(raw_csv)
    # Normalise colonne SMILES (peut être dernière colonne)
    if "smiles" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "smiles"})

    # Garde uniquement SMILES + tâches connues
    task_cols = [c for c in _TOX21_TASKS if c in df.columns]
    df = df[["smiles"] + task_cols].dropna(subset=["smiles"])

    train_df, val_df, test_df = _scaffold_split(df)

    paths = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = save_path / f"tox21_{split_name}.csv"
        split_df.to_csv(p, index=False)
        paths[split_name] = str(p)
        print(f"  {split_name}: {p} ({len(split_df)} mol, {len(task_cols)} tasks)")

    return paths


def download_sider_data(save_dir="data/external/sider"):
    """SIDER : téléchargement direct sans DeepChem."""
    import urllib.request

    _SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
    import gzip, shutil

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    raw_csv = save_path / "sider_raw.csv"
    if not raw_csv.exists():
        gz_path = save_path / "sider.csv.gz"
        print(f"  Téléchargement SIDER depuis {_SIDER_URL} ...")
        urllib.request.urlretrieve(_SIDER_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(raw_csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)

    df = pd.read_csv(raw_csv)
    if "smiles" not in df.columns:
        df = df.rename(columns={df.columns[0]: "smiles"})

    task_cols = [c for c in df.columns if c != "smiles"]
    df = df[["smiles"] + task_cols].dropna(subset=["smiles"])

    train_df, val_df, test_df = _scaffold_split(df)

    paths = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = save_path / f"sider_{split_name}.csv"
        split_df.to_csv(p, index=False)
        paths[split_name] = str(p)
        print(f"  {split_name}: {p} ({len(split_df)} mol, {len(task_cols)} tasks)")

    return paths
