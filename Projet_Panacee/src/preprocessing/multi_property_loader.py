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
import urllib.request
import hashlib
import json
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

_DATASET_FILES = {
    "tox21": ("tox21.csv.gz", [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ]),
    "esol": ("delaney-processed.csv", ["measured log solubility in mols per litre"]),
    "lipo": ("Lipophilicity.csv", ["exp"]),
    "bbbp": ("BBBP.csv", ["p_np"]),
    "clintox": ("clintox.csv.gz", ["FDA_APPROVED", "CT_TOX"]),
    "hiv": ("HIV.csv", ["HIV_active"]),
}
_DATASET_BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"


def download_dataset(name: str, save_dir: str) -> dict:
    """Télécharge le CSV MoleculeNet et produit un split par scaffold RDKit."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if name not in _DATASET_FILES:
        raise ValueError(f"Dataset inconnu : {name}. Choix : {list(_DATASET_FILES)}")

    filename, task_columns = _DATASET_FILES[name]
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    raw_path = save_path / filename
    if not raw_path.exists():
        print(f"  Téléchargement de {name}...")
        pending = save_path / (filename + ".part")
        try:
            urllib.request.urlretrieve(_DATASET_BASE_URL + filename, pending)
            pending.replace(raw_path)
        finally:
            pending.unlink(missing_ok=True)

    df = pd.read_csv(raw_path)
    missing = set(["smiles", *task_columns]) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: colonnes absentes du CSV : {sorted(missing)}")
    # Conserver aussi les SMILES strictement identiques : leurs labels peuvent
    # être discordants et doivent passer par l'audit de réconciliation global.
    missing_smiles = int(df["smiles"].isna().sum())
    df = df[["smiles", *task_columns]].dropna(subset=["smiles"])
    df = df.copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    blank_smiles = int((df["smiles"] == "").sum())
    df = df[df["smiles"] != ""].reset_index(drop=True)
    for task in task_columns:
        numeric = pd.to_numeric(df[task], errors="coerce")
        invalid = df[task].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"{name}: {int(invalid.sum())} étiquettes non numériques dans {task}")
        if not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: étiquettes infinies dans {task}")
        df[task] = numeric
    if missing_smiles or blank_smiles:
        print(f"  ⚠ {name}: {missing_smiles} SMILES manquants et {blank_smiles} vides exclus")

    # Les molécules qui partagent un scaffold restent dans le même split.
    groups = {}
    valid_rows = []
    for idx, smiles in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(scaffold, []).append(idx)
        valid_rows.append(idx)
    invalid_smiles = len(df) - len(valid_rows)
    if invalid_smiles:
        print(f"  ⚠ {name}: {invalid_smiles} SMILES non analysables par RDKit exclus")
    if not valid_rows:
        raise ValueError(f"{name}: aucun SMILES valide")

    indices = {"train": [], "val": [], "test": []}
    n = len(valid_rows)
    for group in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
        if len(indices["train"]) + len(group) <= int(0.8 * n):
            target = "train"
        elif len(indices["val"]) + len(group) <= int(0.1 * n):
            target = "val"
        else:
            target = "test"
        indices[target].extend(group)

    if not indices["train"] or not indices["val"] or not indices["test"]:
        raise ValueError(f"{name}: split scaffold vide ; vérifier le CSV")

    paths = {}
    for split_name, row_ids in indices.items():
        csv_path = save_path / f"{name}_{split_name}.csv"
        split_df = df.iloc[row_ids].rename(
            columns={task: f"{name}_{task}" for task in task_columns} if name != "tox21" else {}
        )
        split_df.to_csv(csv_path, index=False)
        paths[split_name] = str(csv_path)
        print(f"    {split_name}: {len(split_df)} molécules → {csv_path}")
    return paths


def download_all_phase3_data(base_dir: str | None = None) -> dict:
    """
    Télécharge tous les datasets nécessaires pour Phase 3.

    Returns:
        dict {dataset_name: {split: csv_path}}
    """
    if base_dir is None:
        base_dir = str(EXTERNAL_DIR / "phase3")

    os.makedirs(base_dir, exist_ok=True)

    all_paths = {}
    datasets_to_download = ["tox21", "esol", "lipo", "bbbp", "clintox", "hiv"]

    failures = []
    for name in datasets_to_download:
        try:
            save_dir = os.path.join(base_dir, name)
            paths = download_dataset(name, save_dir)
            all_paths[name] = paths
            print(f"  ✓ {name} téléchargé")
        except Exception as e:
            print(f"  ✗ Erreur {name}: {e}")
            all_paths[name] = None
            failures.append(name)

    if failures:
        raise RuntimeError(f"Datasets Phase 3 incomplets : {', '.join(failures)}")
    repartition_global_scaffolds(all_paths, base_dir)
    return all_paths


def _chemical_key(smiles: str) -> tuple[str, str]:
    """Identité canonique et scaffold utilisés pour tous les jeux Phase 3."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"SMILES invalide dans les partitions Phase 3 : {str(smiles)[:80]}")
    return (Chem.MolToSmiles(mol, isomericSmiles=True),
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))


def verify_global_split_integrity(data_paths: dict) -> dict:
    """Bloque toute fuite d'identité ou de scaffold entre jeux/splits."""
    owner: dict[str, str] = {}
    scaffold_owner: dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for paths in data_paths.values():
        if not paths:
            raise ValueError("Dataset Phase 3 manquant")
        for split in counts:
            if split not in paths:
                raise ValueError(f"Partition {split} manquante")
            frame = pd.read_csv(paths[split], usecols=["smiles"])
            for smiles in frame["smiles"]:
                canonical, scaffold = _chemical_key(smiles)
                if canonical in owner and owner[canonical] != split:
                    raise ValueError("Fuite de molécule canonique entre splits Phase 3")
                if scaffold in scaffold_owner and scaffold_owner[scaffold] != split:
                    raise ValueError("Fuite de scaffold entre splits Phase 3")
                owner[canonical] = split
                scaffold_owner[scaffold] = split
                counts[split] += 1
    if any(n == 0 for n in counts.values()):
        raise ValueError("Partition Phase 3 vide")
    return {"counts": counts, "unique_molecules": len(owner),
            "unique_scaffolds": len(scaffold_owner), "global_scaffold_disjoint": True}


def _assign_global_scaffolds(groups: dict[str, set[str]]) -> dict[str, str]:
    """Attribue chaque scaffold une seule fois, de façon déterministe."""
    total = len(set().union(*groups.values()))
    if not total:
        raise ValueError("Aucune molécule valide pour le découpage global")
    sizes = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for scaffold, molecules in sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if sizes["train"] + len(molecules) <= int(0.8 * total):
            split = "train"
        elif sizes["val"] + len(molecules) <= int(0.1 * total):
            split = "val"
        else:
            split = "test"
        sizes[split] += len(molecules)
        for key in molecules:
            if key in assignment:
                raise ValueError("Même molécule associée à plusieurs scaffolds")
            assignment[key] = split
    if any(n == 0 for n in sizes.values()):
        raise ValueError("Découpage global impossible : partition vide")
    return assignment


def repartition_global_scaffolds(data_paths: dict, base_dir: str) -> dict:
    """Recrée des partitions cohérentes entre les six jeux, puis les vérifie."""
    datasets = {}
    groups: dict[str, set[str]] = {}
    duplicate_audit = {}
    for name, paths in data_paths.items():
        frame = pd.concat([pd.read_csv(paths[split]) for split in ("train", "val", "test")],
                          ignore_index=True)
        canonical = []
        scaffolds = {}
        for smiles in frame["smiles"]:
            key, scaffold = _chemical_key(smiles)
            canonical.append(key)
            scaffolds[key] = scaffold
        frame["source_smiles"] = frame["smiles"]
        frame["smiles"] = canonical
        # Une même clé canonique peut représenter des stéréoisomères dont le
        # SMILES ne précise pas la stéréochimie. Ne pas moyenner leurs labels.
        regression = name in {"esol", "lipo"}
        resolved = []
        quarantined = []
        repeated = 0
        for key, records in frame.groupby("smiles", sort=False):
            if len(records) > 1:
                repeated += 1
            row = {"smiles": key}
            conflict = False
            for col in frame.columns.drop(["smiles", "source_smiles"]):
                values = pd.to_numeric(records[col], errors="coerce").dropna()
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"{name}: valeur non finie pour {key}, colonne {col}")
                unique = values.unique()
                if len(unique) > 1:
                    if not regression:
                        raise ValueError(
                            f"{name}: étiquettes discordantes pour {key}, colonne {col} "
                            f"({len(unique)} valeurs)"
                        )
                    conflict = True
                row[col] = values.iloc[0] if len(values) else np.nan
            if conflict:
                quarantined.append({"canonical_smiles": key, "observations": [
                    {"source_smiles": str(record["source_smiles"]),
                     "labels": {col: (None if pd.isna(record[col]) else float(record[col]))
                                for col in frame.columns.drop(["smiles", "source_smiles"])}}
                    for _, record in records.iterrows()
                ]})
            else:
                resolved.append(row)
                groups.setdefault(scaffolds[key], set()).add(key)
        datasets[name] = pd.DataFrame(resolved, columns=frame.columns.drop("source_smiles"))
        duplicate_audit[name] = {"repeated_compounds": repeated,
                                 "quarantined_compounds": len(quarantined),
                                 "resolution": "quarantine_discordant" if regression else "identical_only",
                                 "conflicts": quarantined}
        if quarantined:
            print(f"  ⚠ {name}: {len(quarantined)} composés ambigus exclus des splits; observations dans split_manifest.json")

    assignment = _assign_global_scaffolds(groups)
    prepared = {}
    for name, frame in datasets.items():
        target = frame["smiles"].map(assignment)
        for split, path in data_paths[name].items():
            output = frame.loc[target == split]
            if output.empty:
                raise ValueError(f"{name}: partition {split} vide après découpage global")
            prepared[(name, split)] = (path, output)

    hashes = {}
    for (name, split), (path, output) in prepared.items():
        output.to_csv(path, index=False)
        hashes[f"{name}_{split}"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    audit = verify_global_split_integrity(data_paths)
    manifest = {"method": "global_canonical_murcko_v1", "audit": audit,
                "duplicate_resolution": duplicate_audit,
                "sha256_generated_splits": hashes,
                "warning": "Provenance externe et validité biologique non vérifiées"}
    (Path(base_dir) / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


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
