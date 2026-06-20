"""
Scaffold split (Bemis-Murcko) + k-fold scaffold cross-validation.

Le split aléatoire surestime les performances en chimio-informatique : des
molécules très proches se retrouvent dans train ET val. Le scaffold split
regroupe par squelette moléculaire → évaluation plus honnête de la
généralisation à des structures nouvelles.

Fonctions :
    generate_scaffold(smiles)              -> str (SMILES du scaffold)
    scaffold_split(smiles, frac_train)     -> (train_idx, val_idx)
    scaffold_kfold(smiles, k)              -> list[(train_idx, val_idx)]
"""
from collections import defaultdict
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Squelette de Bemis-Murcko d'un SMILES (chaîne vide si invalide)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return ""
    return scaffold


def _group_by_scaffold(smiles_list: List[str]):
    """Regroupe les indices par scaffold, triés par taille de groupe décroissante."""
    groups = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        groups[scaffold].append(idx)
    # Gros groupes d'abord → plus déterministe / équilibré
    return sorted(groups.values(), key=lambda x: (len(x), x[0]), reverse=True)


def scaffold_split(
    smiles_list: List[str], frac_train: float = 0.8
) -> Tuple[List[int], List[int]]:
    """
    Split scaffold-based. Les molécules d'un même scaffold restent ensemble.

    Returns:
        (train_indices, val_indices)
    """
    scaffold_sets = _group_by_scaffold(smiles_list)
    n_total = len(smiles_list)
    n_train_target = int(frac_train * n_total)

    train_idx, val_idx = [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        else:
            val_idx.extend(group)
    return train_idx, val_idx


def scaffold_kfold(
    smiles_list: List[str], k: int = 5
) -> List[Tuple[List[int], List[int]]]:
    """
    k-fold cross-validation par scaffold (stratifiée par taille de groupe).

    Chaque scaffold est assigné à UN seul fold (gloutonnement, au fold le moins
    rempli) → aucune fuite de scaffold entre train et val d'un même fold.

    Returns:
        liste de k tuples (train_indices, val_indices)
    """
    assert k >= 2, "k doit être >= 2"
    scaffold_sets = _group_by_scaffold(smiles_list)

    folds: List[List[int]] = [[] for _ in range(k)]
    for group in scaffold_sets:
        # Assigner au fold actuellement le plus petit (équilibrage)
        smallest = min(range(k), key=lambda f: len(folds[f]))
        folds[smallest].extend(group)

    splits = []
    for f in range(k):
        val_idx = folds[f]
        train_idx = [i for g in range(k) if g != f for i in folds[g]]
        splits.append((train_idx, val_idx))
    return splits
