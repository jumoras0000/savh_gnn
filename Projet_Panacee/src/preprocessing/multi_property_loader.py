"""
Chargeur de données multi-propriétés – Phase 3.

Sources de données supportées :
  - Tox21 (toxicité, 12 tâches) via DeepChem
  - ESOL (solubilité aqueuse) via DeepChem
  - Lipophilicity (LogP) via DeepChem
  - BBBP (bioavailabilité / passage barrière hémato-encéphalique) via DeepChem
  - ClinTox (toxicité clinique) via DeepChem
  - HIV (activité anti-VIH) via DeepChem
  - CSV personnalisé (format libre avec colonne SMILES)

Chaque molécule est enrichie avec toutes les propriétés disponibles.
Les propriétés manquantes sont marquées NaN et ignorées dans la loss.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_DIR, SMILES_COLUMN_CANDIDATES
from src.preprocessing.graph_builder import smiles_to_graph

# ══════════════════════════════════════════════════════════════════════
# Téléchargement des datasets
# ══════════════════════════════════════════════════════════════════════

def download_dataset(name: str, save_dir: str) -> dict:
    """
    Télécharge un dataset via DeepChem et sauvegarde en CSV.

    Args:
        name: nom du dataset (tox21, esol, lipo, bbbp, clintox, hiv)
        save_dir: dossier de destination

    Returns:
        dict avec 'train', 'val', 'test' → chemins CSV
    """
    try:
        import deepchem as dc
    except ImportError as e:
        raise ImportError(f"DeepChem ne peut pas être importé : {e}") from e

    os.makedirs(save_dir, exist_ok=True)
    loaders = {
        "tox21": dc.molnet.load_tox21,
        "esol": dc.molnet.load_delaney,
        "lipo": dc.molnet.load_lipo,
        "bbbp": dc.molnet.load_bbbp,
        "clintox": dc.molnet.load_clintox,
        "hiv": dc.molnet.load_hiv,
    }

    if name not in loaders:
        raise ValueError(f"Dataset inconnu : {name}. Choix : {list(loaders.keys())}")

    print(f"  Téléchargement de {name}...")
    tasks, datasets, _transformers = loaders[name](featurizer="Raw", splitter="scaffold")
    train_ds, val_ds, test_ds = datasets

    paths = {}
    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        # Extraire les SMILES et les labels
        from rdkit import Chem
        smiles_list = []
        for x in ds.X:
            if isinstance(x, Chem.rdchem.Mol):
                smiles_list.append(Chem.MolToSmiles(x))
            elif isinstance(x, str):
                smiles_list.append(x)
            else:
                smiles_list.append(str(x))

        df = pd.DataFrame({"smiles": smiles_list})
        labels = ds.y
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        for i, task_name in enumerate(tasks):
            col_name = f"{name}_{task_name}" if name != "tox21" else task_name
            df[col_name] = labels[:, i]

        csv_path = os.path.join(save_dir, f"{name}_{split_name}.csv")
        df.to_csv(csv_path, index=False)
        paths[split_name] = csv_path
        print(f"    {split_name}: {len(df)} molécules → {csv_path}")

    return paths


def download_all_phase3_data(base_dir: str | None = None) -> dict:
    """
    Télécharge tous les datasets nécessaires pour Phase 3.

    Returns:
        dict {dataset_name: {split: csv_path}}
    """
    if base_dir is None:
        base_dir = str(EXTERNAL_DIR / "phase3")

    # Vérifier l'import réel : find_spec('deepchem') ne garantit pas que ses
    # dépendances sont utilisables dans cet environnement Python.
    try:
        import deepchem  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            f"DeepChem est installé mais inutilisable : {type(e).__name__}: {e}"
        ) from e

    os.makedirs(base_dir, exist_ok=True)

    all_paths = {}
    datasets_to_download = ["tox21", "esol", "lipo", "bbbp", "clintox", "hiv"]

    for name in datasets_to_download:
        try:
            save_dir = os.path.join(base_dir, name)
            paths = download_dataset(name, save_dir)
            all_paths[name] = paths
            print(f"  ✓ {name} téléchargé")
        except Exception as e:
            print(f"  ✗ Erreur {name}: {e}")
            all_paths[name] = None

    if not any(all_paths.values()):
        raise RuntimeError("Aucun dataset Phase 3 téléchargé ; voir les erreurs ci-dessus")
    return all_paths


# ══════════════════════════════════════════════════════════════════════
# Fusion des données
# ══════════════════════════════════════════════════════════════════════

def merge_datasets(data_paths: dict, split: str = "train") -> pd.DataFrame:
    """
    Fusionne les datasets par SMILES (outer join).
    Les propriétés manquantes deviennent NaN.

    Args:
        data_paths: dict {name: {split: path}} de download_all_phase3_data
        split: 'train', 'val', ou 'test'

    Returns:
        DataFrame avec colonne 'smiles' et toutes les propriétés
    """
    merged = None

    for name, paths in data_paths.items():
        if paths is None or split not in paths:
            continue

        csv_path = paths[split]
        if not os.path.exists(csv_path):
            print(f"  WARN: {csv_path} introuvable, ignoré")
            continue

        df = pd.read_csv(csv_path)

        # Détecter la colonne SMILES
        smiles_col = None
        for candidate in SMILES_COLUMN_CANDIDATES:
            if candidate in df.columns:
                smiles_col = candidate
                break

        if smiles_col is None:
            print(f"  WARN: pas de colonne SMILES dans {csv_path}")
            continue

        df = df.rename(columns={smiles_col: "smiles"})

        if merged is None:
            merged = df
        else:
            # Outer join sur SMILES
            merged = pd.merge(merged, df, on="smiles", how="outer", suffixes=("", f"_{name}"))

    if merged is None:
        raise ValueError("Aucun dataset n'a pu être chargé")

    print(f"  Dataset fusionné ({split}): {len(merged)} molécules, {len(merged.columns)-1} propriétés")
    return merged


# ══════════════════════════════════════════════════════════════════════
# Dataset PyTorch
# ══════════════════════════════════════════════════════════════════════

class MultiPropertyDataset(Dataset):
    """
    Dataset multi-propriétés pour Phase 3.

    Chaque échantillon contient :
      - data : graphe PyG
      - labels : dict {property_group: tensor}

    Les labels manquants sont NaN et seront ignorés par la loss.
    Les targets de régression sont normalisés (z-score sur le train set).
    """

    # Mapping des colonnes vers les groupes de propriétés
    PROPERTY_GROUPS = {
        "toxicity": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "efficacy": ["hiv_HIV_active"],
        # Solubilité : uniquement la mesure expérimentale (Delaney 2004).
        # La colonne "ESOL predicted" est un baseline calculé, pas un label.
        "solubility": ["esol_measured log solubility in mols per litre"],
        "lipophilicity": ["lipo_exp"],
        # BBBP = perméabilité barrière hémato-encéphalique (Blood-Brain Barrier
        # Penetration), PAS biodisponibilité orale. Nom corrigé.
        "bbb_penetration": ["bbbp_p_np"],
        # ClinTox CT_TOX = toxicité clinique (échec en essai clinique pour
        # toxicité). PAS stabilité métabolique. Nom corrigé.
        "clinical_toxicity": ["clintox_CT_TOX"],
    }

    REGRESSION_GROUPS = {"solubility", "lipophilicity"}

    def __init__(self, dataframe: pd.DataFrame, smiles_column: str = "smiles",
                 regression_stats: dict | None = None):
        """
        Args:
            dataframe: DataFrame avec colonne SMILES et propriétés
            smiles_column: nom de la colonne SMILES
            regression_stats: stats de normalisation {group: {"mean": float, "std": float}}
                              None = calcule sur ce dataset (mode train).
                              Fournir les stats du train set pour val/test.
        """
        self.smiles_column = smiles_column
        self.df = dataframe.reset_index(drop=True)

        # Identifier les colonnes de propriétés disponibles
        self.available_props = {}
        for group_name, candidates in self.PROPERTY_GROUPS.items():
            found_cols = [c for c in candidates if c in self.df.columns]
            if found_cols:
                self.available_props[group_name] = found_cols

        print("  Propriétés disponibles :")
        for group, cols in self.available_props.items():
            print(f"    {group}: {len(cols)} colonnes")

        # Normalisation z-score des targets de régression
        if regression_stats is not None:
            self.regression_stats = regression_stats
        else:
            self.regression_stats = self._compute_regression_stats()
        if self.regression_stats:
            print("  Normalisation régression :")
            for g, s in self.regression_stats.items():
                print(f"    {g}: mean={s['mean']:.4f}, std={s['std']:.4f}")

        # Construire les graphes une seule fois
        self.graphs = []
        self.labels_list = []
        self.valid_indices = []

        print("  Construction des graphes...")
        n_fail = 0
        for idx in range(len(self.df)):
            smiles = str(self.df.iloc[idx][self.smiles_column])
            graph = smiles_to_graph(smiles)
            if graph is None:
                n_fail += 1
                continue

            # Extraire les labels pour chaque groupe
            labels = {}
            for group_name, cols in self.available_props.items():
                values = []
                for col in cols:
                    val = self.df.iloc[idx].get(col, float("nan"))
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        values.append(float("nan"))
                t = torch.tensor(values, dtype=torch.float32)
                if group_name in self.regression_stats:
                    stats = self.regression_stats[group_name]
                    mask = ~torch.isnan(t)
                    t[mask] = (t[mask] - stats["mean"]) / stats["std"]
                labels[group_name] = t

            self.graphs.append(graph)
            self.labels_list.append(labels)
            self.valid_indices.append(idx)

        print(f"  {len(self.graphs)} graphes valides, {n_fail} échecs")

    def _compute_regression_stats(self) -> dict:
        """Calcule mean/std sur les colonnes de régression (train set)."""
        stats = {}
        for group_name in self.REGRESSION_GROUPS:
            if group_name not in self.available_props:
                continue
            cols = self.available_props[group_name]
            vals = self.df[cols].values.astype(float).flatten()
            valid = vals[~np.isnan(vals)]
            if len(valid) < 2:
                continue
            mean = float(np.mean(valid))
            std = float(np.std(valid))
            if std < 1e-8:
                std = 1.0
            stats[group_name] = {"mean": mean, "std": std}
        return stats

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels_list[idx]

    def get_property_dims(self):
        """Retourne le nombre de dimensions par groupe de propriétés."""
        return {name: len(cols) for name, cols in self.available_props.items()}

    def get_pos_weight(self, group: str = "toxicity"):
        """Calcule pos_weight pour un groupe de classification."""
        if group not in self.available_props:
            return None

        cols = self.available_props[group]
        vals = self.df[cols].values.astype(float)
        pos_weights = []
        for i in range(vals.shape[1]):
            col_vals = vals[:, i]
            mask = ~np.isnan(col_vals)
            if mask.sum() == 0:
                pos_weights.append(1.0)
                continue
            positives = col_vals[mask].sum()
            negatives = mask.sum() - positives
            pw = negatives / max(positives, 1)
            pos_weights.append(min(pw, 10.0))  # Plafonner à 10
        return torch.tensor(pos_weights, dtype=torch.float32)


class TaskBalancedSampler(torch.utils.data.Sampler):
    """Sampler qui équilibre les tâches dans chaque batch.

    Problème : avec un outer join, chaque molécule n'a de labels que pour 1 dataset.
    Un batch aléatoire peut n'avoir que des molécules Tox21 → gradient nul pour
    les autres tâches. Ce sampler intercale les sources pour diversifier chaque batch.
    """

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._build_task_indices()

    def _build_task_indices(self):
        self.task_indices = {}
        for idx, labels in enumerate(self.dataset.labels_list):
            for group_name, t in labels.items():
                if (~torch.isnan(t)).any():
                    self.task_indices.setdefault(group_name, []).append(idx)

    def __iter__(self):
        task_names = list(self.task_indices.keys())
        pools = {}
        for t in task_names:
            arr = np.array(self.task_indices[t])
            if self.shuffle:
                np.random.shuffle(arr)
            pools[t] = list(arr)

        indices = []
        per_task = max(1, self.batch_size // len(task_names))
        while any(pools.values()):
            batch = []
            for t in task_names:
                take = min(per_task, len(pools[t]))
                batch.extend(pools[t][:take])
                pools[t] = pools[t][take:]
            if self.shuffle:
                np.random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def collate_multi_property(batch):
    """
    Collate function pour MultiPropertyDataset.
    Retourne : (PyG Batch, dict de labels empilés)
    """
    graphs, labels_list = zip(*batch, strict=False)

    # Batch des graphes
    pyg_batch = Batch.from_data_list(list(graphs))

    # Empiler les labels par groupe
    merged_labels = {}
    all_groups = set()
    for lb in labels_list:
        all_groups.update(lb.keys())

    for group in all_groups:
        tensors = [lb[group] for lb in labels_list if group in lb]
        if tensors:
            merged_labels[group] = torch.stack(tensors)

    return pyg_batch, merged_labels
