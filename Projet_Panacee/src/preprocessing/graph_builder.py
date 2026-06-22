"""
Module de conversion SMILES → Graphes moléculaires (PyTorch Geometric).
Features normalisées pour stabilité numérique.
"""
import warnings

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

# ── Constantes de normalisation ──────────────────────────────────────
_MAX_ATOMIC_NUM = 118.0
_MAX_DEGREE     = 6.0
_MAX_HYBRID     = 6.0
_MAX_VALENCE    = 6.0
_MAX_HS         = 8.0

# Dimension attendue des features (doit correspondre à config.ATOM_FEATURE_DIM)
ATOM_DIM = 9
BOND_DIM = 6

# ── Vocabulaire de TYPES d'atomes (pré-entraînement MGM par classification) ──
# L'identité de l'élément (C/N/O/S/halogènes…) ne se déduit PAS de la topologie
# du graphe, contrairement aux features structurelles (degré, cycle, valence).
# Prédire ce type force donc l'encodeur à apprendre une vraie chimie locale.
# Éléments fréquents en chimie médicinale ; tout le reste → classe « autre ».
ATOM_TYPE_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34]  # C N O F P S Cl Br I B Si Se
_ATOM_TYPE_TO_CLASS = {z: i for i, z in enumerate(ATOM_TYPE_LIST)}
ATOM_TYPE_OTHER = len(ATOM_TYPE_LIST)              # index de la classe « autre »
ATOM_TYPE_VOCAB_SIZE = len(ATOM_TYPE_LIST) + 1     # +1 pour « autre »


def atomic_num_to_class(z: int) -> int:
    """Numéro atomique → index de classe dans le vocabulaire de types d'atomes."""
    return _ATOM_TYPE_TO_CLASS.get(int(round(z)), ATOM_TYPE_OTHER)


def smiles_to_graph(smiles: str):
    """
    Convertit un SMILES en graphe PyTorch Geometric avec features normalisées.

    Returns ``None`` au lieu de lever une exception quand le SMILES est
    invalide, ce qui permet de filtrer proprement dans les datasets.

    Features atomiques (9-dim, toutes dans [0, 1]) :
        0  numéro atomique / 118
        1  degré / 6
        2  charge formelle (clampée [-2,2] puis /4 + 0.5)
        3  hybridation / 6
        4  aromatique  (0 / 1)
        5  électrons radicaux (0 / 1 clampé)
        6  valence implicite / 6
        7  total H / 8
        8  dans un cycle (0 / 1)

    Features de liaison (6-dim, binaires) :
        single, double, triple, aromatic, conjugated, in_ring
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Features atomiques ────────────────────────────────────────────
    atom_features = []
    for atom in mol.GetAtoms():
        charge = max(-2.0, min(2.0, atom.GetFormalCharge()))
        features = [
            atom.GetAtomicNum()            / _MAX_ATOMIC_NUM,
            atom.GetDegree()               / _MAX_DEGREE,
            (charge + 2.0)                 / 4.0,
            int(atom.GetHybridization())   / _MAX_HYBRID,
            float(atom.GetIsAromatic()),
            min(atom.GetNumRadicalElectrons(), 1),
            atom.GetImplicitValence()      / _MAX_VALENCE,
            atom.GetTotalNumHs()           / _MAX_HS,
            float(atom.IsInRing()),
        ]
        atom_features.append(features)

    # Clamp [0,1] : garantit l'invariant de normalisation même pour les cas
    # limites (hybridation OTHER=7 -> 7/6>1, degré/valence atypiques, etc.)
    x = torch.tensor(atom_features, dtype=torch.float).clamp_(0.0, 1.0)

    # ── Arêtes (bidirectionnelles) ────────────────────────────────────
    edge_src, edge_dst, edge_attr_list = [], [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bf = [
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        # i → j et j → i
        edge_src += [i, j]
        edge_dst += [j, i]
        edge_attr_list += [bf, bf]

    if len(edge_src) == 0:
        # Molécule à un seul atome : self-loop pour éviter graphe vide
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr  = torch.zeros(1, BOND_DIM, dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def mask_atoms(graph: Data, mask_prob: float = 0.15):
    """
    Masque aléatoirement des atomes (stratégie BERT-like pour graphes).

    80 % → zéros, 10 % → bruit aléatoire, 10 % → inchangé.
    Ceci empêche le modèle de se fier au signal "tout-zéro" et améliore
    la qualité des représentations apprises (Hu et al. 2020).

    La CIBLE de pré-entraînement est le **type d'atome** (classe d'élément),
    qui ne fuit pas par la topologie — l'objectif est donc réellement difficile.

    Returns:
        (graph_masked, masked_indices_list, masked_features_tensor, masked_types_tensor)
        - masked_features_tensor : [M, ATOM_DIM]  features originales (cible auxiliaire)
        - masked_types_tensor    : [M]            classe d'élément (cible principale)
    """
    num_atoms = graph.x.size(0)
    num_masked = max(1, int(num_atoms * mask_prob))

    perm = torch.randperm(num_atoms)[:num_masked]
    masked_features = graph.x[perm].clone()

    # Cible principale : type d'atome. feature[0] = numéro_atomique / 118.
    masked_types = torch.tensor(
        [atomic_num_to_class(masked_features[i, 0].item() * _MAX_ATOMIC_NUM)
         for i in range(masked_features.size(0))],
        dtype=torch.long,
    )

    graph_masked = graph.clone()

    for idx in perm:
        r = torch.rand(1).item()
        if r < 0.8:
            graph_masked.x[idx] = 0.0                        # 80 % zéros
        elif r < 0.9:
            graph_masked.x[idx] = torch.rand(ATOM_DIM)       # 10 % bruit
        # 10 % restants : inchangé

    return graph_masked, perm.tolist(), masked_features, masked_types
