"""
data_loaders.py – Chargeurs de données (ZINC, Tox21, multi-propriétés)
=======================================================================
Tous les DataLoaders dans un seul fichier pour compatibilité Colab.

Sources supportées :
  • ZINC        → pré-entraînement Phase 1 (250 k molécules drogues)
  • Tox21       → classification toxicité Phase 2  (12 tâches)
  • ESOL        → solubilité aqueuse (régression)
  • Lipophilicity → LogP (régression)
  • BBBP        → passage barrière hémato-encéphalique (classification)
  • ClinTox     → toxicité clinique (classification)
  • HIV         → activité antivirale (classification)
  • CSV custom  → format libre avec colonne SMILES

Téléchargement via DeepChem (scaffold split automatique).
"""
import os, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple

from graph_builder import smiles_to_graph, mask_atoms
from config import (
    SMILES_COLUMN_CANDIDATES, DATA_DIR, EXTERNAL_DIR,
    NUM_WORKERS, PIN_MEMORY,
)

# ══════════════════════════════════════════════════════════════════════
# ZONE 1 – ZINC (pré-entraînement Phase 1)
# ══════════════════════════════════════════════════════════════════════

def process_zinc_to_pt(
    csv_path: str,
    output_pt: str,
    max_molecules: Optional[int] = None,
) -> str:
    """
    Lit le CSV ZINC, canonise, déduplique et sauvegarde un .pt.

    Args:
        csv_path      : chemin vers 250k_rndm_zinc_drugs_clean_3.csv
        output_pt     : chemin de sortie (ex: data/zinc_pretrain.pt)
        max_molecules : limiter pour tests rapides

    Returns:
        chemin du .pt créé
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    print(f"[ZINC] Chargement : {csv_path}")
    df = pd.read_csv(csv_path)

    # Détecter la colonne SMILES
    smiles_col = _find_smiles_col(df)
    smiles_raw = df[smiles_col].dropna().tolist()
    if max_molecules:
        smiles_raw = smiles_raw[:max_molecules]

    print(f"  → {len(smiles_raw)} SMILES bruts lus")

    valid, n_invalid = [], 0
    for s in tqdm(smiles_raw, desc="Canonisation"):
        try:
            mol = Chem.MolFromSmiles(str(s).strip())
            if mol is not None:
                Chem.SanitizeMol(mol)
                valid.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                n_invalid += 1
        except Exception:
            n_invalid += 1

    unique = list(set(valid))
    print(f"  → {len(valid)} valides | {n_invalid} invalides | {len(unique)} uniques")

    os.makedirs(os.path.dirname(os.path.abspath(output_pt)), exist_ok=True)
    payload = {
        "smiles": unique,
        "num_molecules": len(unique),
        "source": csv_path,
    }
    torch.save(payload, output_pt)

    stats_path = output_pt.replace(".pt", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_raw": len(smiles_raw),
            "valid": len(valid),
            "invalid": n_invalid,
            "unique": len(unique),
        }, f, indent=2)

    print(f"  → Sauvegardé : {output_pt}")
    return output_pt


class PretrainDataset(Dataset):
    """Dataset Phase 1 : SMILES → graphe masqué + features cibles."""

    def __init__(self, smiles_list: List[str], mask_prob: float = 0.15):
        self.smiles   = smiles_list
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = smiles_to_graph(self.smiles[idx])
        if graph is None:
            # Chercher un SMILES valide à proximité
            for offset in range(1, len(self.smiles)):
                graph = smiles_to_graph(self.smiles[(idx + offset) % len(self.smiles)])
                if graph is not None:
                    break
            if graph is None:
                raise RuntimeError("Aucun SMILES valide dans le dataset")

        graph_masked, masked_idx, masked_feat = mask_atoms(graph, self.mask_prob)
        return graph_masked, masked_idx, masked_feat


def collate_pretrain(batch):
    """Collate Phase 1 : fusionne les graphes en un Batch PyG."""
    graphs, indices, features = zip(*batch)
    batch_graph   = Batch.from_data_list(graphs)
    all_features  = torch.cat(features, dim=0)
    return batch_graph, list(indices), all_features


def make_pretrain_loaders(
    pt_path: str,
    val_split: float = 0.10,
    batch_size: int = 64,
    mask_prob: float = 0.15,
    max_molecules: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Crée les DataLoaders train/val pour Phase 1."""
    data = torch.load(pt_path, weights_only=False)
    smiles = data["smiles"]
    if max_molecules:
        smiles = smiles[:max_molecules]

    n_val   = int(len(smiles) * val_split)
    n_train = len(smiles) - n_val

    train_ds = PretrainDataset(smiles[:n_train], mask_prob)
    val_ds   = PretrainDataset(smiles[n_train:], mask_prob)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pretrain,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pretrain,
    )
    print(f"[PretrainLoaders] train={len(train_ds)} | val={len(val_ds)}")
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════
# ZONE 2 – TOXICITÉ (Phase 2, Tox21 / SIDER)
# ══════════════════════════════════════════════════════════════════════

_EXCLUDE_COLS = {
    "mol_id", "id", "ids", "compound_id", "name",
    *(f"w{i}" for i in range(1, 28)),
}


class ToxicityDataset(Dataset):
    """
    Dataset multi-tâches pour la classification de toxicité.
    Compatible Tox21, SIDER, ClinTox, ou tout CSV personnalisé.
    Les labels NaN sont tolérés (ignorés dans la loss).
    """

    def __init__(
        self,
        csv_path: str,
        smiles_column: str = "smiles",
        task_columns: Optional[List[str]] = None,
        max_molecules: Optional[int] = None,
    ):
        df = pd.read_csv(csv_path)
        self.smiles_column = _find_smiles_col(df, hint=smiles_column)

        if max_molecules:
            df = df.head(max_molecules)

        self.task_columns = task_columns or _detect_task_cols(df, self.smiles_column)
        print(f"  [ToxDataset] {len(df)} mol | {len(self.task_columns)} tâches")

        self.graphs: List = []
        self.labels: Optional[torch.Tensor] = None
        self._build(df)

    # ─────────────────────────────────────────────────────────────────
    def _build(self, df: pd.DataFrame):
        raw_labels, n_invalid = [], 0
        for _, row in df.iterrows():
            g = smiles_to_graph(row[self.smiles_column])
            if g is None:
                n_invalid += 1
                continue
            lbl = [
                float("nan") if pd.isna(row[c]) else float(row[c])
                for c in self.task_columns
            ]
            self.graphs.append(g)
            raw_labels.append(lbl)

        self.labels = torch.tensor(raw_labels, dtype=torch.float32)
        if n_invalid:
            print(f"  [ToxDataset] {n_invalid} SMILES invalides ignorés")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def get_task_names(self) -> List[str]:
        return self.task_columns

    def get_num_tasks(self) -> int:
        return len(self.task_columns)

    def get_pos_weight(self) -> torch.Tensor:
        """Poids pour corriger le déséquilibre de classes (neg/pos par tâche)."""
        pw = []
        for t in range(self.labels.shape[1]):
            col   = self.labels[:, t]
            valid = col[~torch.isnan(col)]
            pos   = (valid == 1).sum().float()
            neg   = (valid == 0).sum().float()
            pw.append((neg / pos).item() if pos > 0 else 1.0)
        return torch.tensor(pw, dtype=torch.float32)


def collate_toxicity(batch):
    graphs, labels = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(labels)


def make_tox_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
    batch_size: int = 64,
    task_columns: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Crée les DataLoaders train/val/test pour Phase 2."""
    train_ds = ToxicityDataset(train_csv, task_columns=task_columns)
    val_ds   = ToxicityDataset(val_csv,   task_columns=task_columns)
    test_ds  = ToxicityDataset(test_csv,  task_columns=task_columns) if test_csv else None

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            collate_fn=collate_toxicity,
        )

    return (
        _loader(train_ds, True),
        _loader(val_ds,   False),
        _loader(test_ds,  False) if test_ds else None,
    )


# ══════════════════════════════════════════════════════════════════════
# ZONE 3 – MULTI-PROPRIÉTÉS (Phase 3)
# ══════════════════════════════════════════════════════════════════════

# Mapping : colonnes CSV → groupes de propriétés
PROPERTY_GROUPS = {
    "toxicity": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
    "efficacy"           : ["HIV_active"],
    "solubility"         : ["measured log solubility in mols per litre"],
    "lipophilicity"      : ["exp"],
    "bioavailability"    : ["p_np"],
    "metabolic_stability": ["FDA_APPROVED", "CT_TOX"],
}


class MultiPropertyDataset(Dataset):
    """
    Dataset multi-propriétés fusionné (Phase 3).
    Supporte toutes les sources MoleculeNet.
    Les propriétés absentes sont NaN → ignorées dans la loss.
    """

    def __init__(self, df: pd.DataFrame):
        self.graphs: List = []
        self.labels: dict = {k: [] for k in PROPERTY_GROUPS}
        self._build(df)

    def _build(self, df: pd.DataFrame):
        smiles_col = _find_smiles_col(df)
        n_invalid  = 0

        for _, row in df.iterrows():
            g = smiles_to_graph(row[smiles_col])
            if g is None:
                n_invalid += 1
                continue

            self.graphs.append(g)
            for prop, cols in PROPERTY_GROUPS.items():
                vals = []
                for c in cols:
                    if c in df.columns and not pd.isna(row.get(c, float("nan"))):
                        vals.append(float(row[c]))
                    else:
                        vals.append(float("nan"))
                self.labels[prop].append(vals)

        # Convertir en tenseurs
        for prop in PROPERTY_GROUPS:
            self.labels[prop] = torch.tensor(
                self.labels[prop], dtype=torch.float32
            )

        if n_invalid:
            print(f"  [MultiPropDataset] {n_invalid} SMILES invalides ignorés")
        print(f"  [MultiPropDataset] {len(self.graphs)} molécules chargées")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], {k: v[idx] for k, v in self.labels.items()}


def collate_multi_property(batch):
    graphs, label_dicts = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    labels = {
        k: torch.stack([d[k] for d in label_dicts])
        for k in label_dicts[0]
    }
    return batch_graph, labels


def make_multi_property_loaders(
    data_dict: dict,
    split: str = "train",
    batch_size: int = 32,
) -> DataLoader:
    """Crée un DataLoader multi-propriétés depuis un dict de DataFrames fusionnés."""
    df = data_dict.get(split)
    if df is None:
        raise ValueError(f"Split '{split}' introuvable dans data_dict")
    ds = MultiPropertyDataset(df)
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_multi_property,
    )


# ══════════════════════════════════════════════════════════════════════
# ZONE 4 – TÉLÉCHARGEMENT VIA DEEPCHEM
# ══════════════════════════════════════════════════════════════════════

def download_dataset_deepchem(
    name: str,
    save_dir: str,
) -> dict:
    """
    Télécharge et sauvegarde en CSV un dataset MoleculeNet via DeepChem.

    Args:
        name    : 'tox21' | 'esol' | 'lipo' | 'bbbp' | 'clintox' | 'hiv'
        save_dir: dossier de destination

    Returns:
        dict {'train': path, 'val': path, 'test': path}
    """
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError("pip install deepchem")

    from rdkit import Chem as _Chem

    os.makedirs(save_dir, exist_ok=True)
    _loaders = {
        "tox21"  : dc.molnet.load_tox21,
        "esol"   : dc.molnet.load_delaney,
        "lipo"   : dc.molnet.load_lipo,
        "bbbp"   : dc.molnet.load_bbbp,
        "clintox": dc.molnet.load_clintox,
        "hiv"    : dc.molnet.load_hiv,
    }

    if name not in _loaders:
        raise ValueError(f"Dataset inconnu: {name}. Options: {list(_loaders)}")

    print(f"  [DeepChem] Téléchargement '{name}'...")
    tasks, datasets, _ = _loaders[name](featurizer="Raw", splitter="scaffold")
    splits = {"train": datasets[0], "val": datasets[1], "test": datasets[2]}

    paths = {}
    for split_name, ds in splits.items():
        smiles_list = []
        for x in ds.X:
            if isinstance(x, _Chem.rdchem.Mol):
                smiles_list.append(_Chem.MolToSmiles(x))
            else:
                smiles_list.append(str(x))

        df   = pd.DataFrame({"smiles": smiles_list})
        labs = ds.y if ds.y.ndim == 2 else ds.y.reshape(-1, 1)
        for i, t in enumerate(tasks):
            col = t if name == "tox21" else f"{name}_{t}"
            df[col] = labs[:, i]

        p = os.path.join(save_dir, f"{name}_{split_name}.csv")
        df.to_csv(p, index=False)
        paths[split_name] = p
        print(f"    {split_name}: {len(df)} mol → {p}")

    return paths


def download_all_phase3_data(base_dir: Optional[str] = None) -> dict:
    """
    Télécharge tous les datasets Phase 3 (tox21, esol, lipo, bbbp, clintox, hiv).

    Returns:
        dict {dataset_name: {split: csv_path}}
    """
    if base_dir is None:
        base_dir = os.path.join(EXTERNAL_DIR, "phase3")

    all_paths = {}
    for name in ["tox21", "esol", "lipo", "bbbp", "clintox", "hiv"]:
        try:
            paths = download_dataset_deepchem(name, os.path.join(base_dir, name))
            all_paths[name] = paths
            print(f"  ✓ {name}")
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            all_paths[name] = None

    return all_paths


def merge_phase3_datasets(data_paths: dict, split: str = "train") -> pd.DataFrame:
    """
    Fusionne les DataFrames de tous les datasets par SMILES (outer join).
    Les propriétés absentes deviennent NaN.
    """
    merged = None
    for name, paths in data_paths.items():
        if paths is None or split not in paths:
            continue
        path = paths[split]
        if not os.path.exists(path):
            continue

        df        = pd.read_csv(path)
        smiles_c  = _find_smiles_col(df)
        df        = df.rename(columns={smiles_c: "smiles"})

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="smiles", how="outer",
                              suffixes=("", f"_{name}"))

    if merged is None:
        raise RuntimeError("Aucun dataset Phase 3 disponible")

    print(f"  [Merge] {split}: {len(merged)} mol | {len(merged.columns)-1} colonnes")
    return merged


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES INTERNES
# ══════════════════════════════════════════════════════════════════════

def _find_smiles_col(df: pd.DataFrame, hint: str = "smiles") -> str:
    """Détecte la colonne SMILES dans un DataFrame."""
    if hint in df.columns:
        return hint
    for c in SMILES_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Colonne SMILES introuvable. Colonnes disponibles : {list(df.columns)[:10]}"
    )


def _detect_task_cols(df: pd.DataFrame, smiles_col: str) -> List[str]:
    """Heuristique : colonnes numériques hors SMILES/id/poids."""
    exclude = _EXCLUDE_COLS | {smiles_col}
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
