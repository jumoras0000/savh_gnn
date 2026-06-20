"""
graph_builder.py – SMILES → Graphes PyTorch Geometric
======================================================
Conversion robuste avec features normalisées et stratégie BERT-like
de masquage (Hu et al. 2020, ICLR "Strategies for Pre-training GNNs").

Features atomiques (9-dim, toutes dans [0, 1]) :
  0  numéro atomique / 118
  1  degré          / 6
  2  charge formelle (clampée [-2,2] puis (val+2)/4)
  3  hybridation    / 6
  4  aromatique     (0/1)
  5  électrons radicaux (clampé à 1)
  6  valence implicite / 6
  7  total H           / 8
  8  dans un cycle     (0/1)

Features de liaison (6-dim, binaires) :
  single, double, triple, aromatic, conjugated, in_ring
"""
import warnings
import torch
from torch_geometric.data import Data
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")

# ── Constantes de normalisation ──────────────────────────────────────
_MAX_ATOMIC_NUM = 118.0
_MAX_DEGREE     = 6.0
_MAX_HYBRID     = 6.0
_MAX_VALENCE    = 6.0
_MAX_HS         = 8.0

ATOM_DIM = 9
BOND_DIM = 6


# ══════════════════════════════════════════════════════════════════════
# Construction du graphe
# ══════════════════════════════════════════════════════════════════════

def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convertit un SMILES en objet PyG Data.

    Retourne ``None`` pour tout SMILES invalide (pas d'exception levée),
    ce qui permet un filtrage propre dans les DataLoaders.
    """
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
    except Exception:
        return None

    if mol is None:
        return None

    # ── Features atomiques ────────────────────────────────────────────
    atom_feats = []
    for atom in mol.GetAtoms():
        charge = float(max(-2, min(2, atom.GetFormalCharge())))
        atom_feats.append([
            atom.GetAtomicNum()          / _MAX_ATOMIC_NUM,
            atom.GetDegree()             / _MAX_DEGREE,
            (charge + 2.0)               / 4.0,
            int(atom.GetHybridization()) / _MAX_HYBRID,
            float(atom.GetIsAromatic()),
            float(min(atom.GetNumRadicalElectrons(), 1)),
            atom.GetImplicitValence()    / _MAX_VALENCE,
            atom.GetTotalNumHs()         / _MAX_HS,
            float(atom.IsInRing()),
        ])
    x = torch.tensor(atom_feats, dtype=torch.float)  # [N, 9]

    # ── Features de liaison (arêtes bidirectionnelles) ────────────────
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        feat = [
            float(bt == Chem.BondType.SINGLE),
            float(bt == Chem.BondType.DOUBLE),
            float(bt == Chem.BondType.TRIPLE),
            float(bt == Chem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]
        src  += [i, j]
        dst  += [j, i]
        edge_feats += [feat, feat]  # arête non-dirigée → copie i→j et j→i

    if len(src) == 0:
        # Molécule mono-atomique : self-loop pour éviter un graphe vide
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr  = torch.zeros(1, BOND_DIM, dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


# ══════════════════════════════════════════════════════════════════════
# Masquage d'atomes (Pré-entraînement MGM)
# ══════════════════════════════════════════════════════════════════════

def mask_atoms(graph: Data, mask_prob: float = 0.15):
    """
    Masquage BERT-like sur les atomes d'un graphe (Hu et al. 2020).

    Stratégie :
      80 % → zéros          (token [MASK])
      10 % → bruit aléatoire (token aléatoire)
      10 % → inchangé        (permet au modèle de se fier au contexte)

    Returns:
        (graph_masked, masked_indices_list, masked_features_tensor)
    """
    num_atoms  = graph.x.size(0)
    num_masked = max(1, int(num_atoms * mask_prob))

    perm            = torch.randperm(num_atoms)[:num_masked]
    masked_features = graph.x[perm].clone()   # features originales à prédire

    graph_masked = graph.clone()
    for idx in perm.tolist():
        r = torch.rand(1).item()
        if r < 0.80:
            graph_masked.x[idx] = torch.zeros(ATOM_DIM)   # [MASK]
        elif r < 0.90:
            graph_masked.x[idx] = torch.rand(ATOM_DIM)    # bruit

    return graph_masked, perm.tolist(), masked_features
