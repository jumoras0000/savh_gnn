# -*- coding: utf-8 -*-
"""
Interprétabilité GNN — attention maps + GNNExplainer sur les hits.

Extrait les poids d'attention par arête de chaque couche EdgeAttentionConv,
agrège en importance par atome, et génère des visualisations SVG/PNG
avec les sous-structures activantes surlignées (RDKit).

Usage :
    python interpret_hits.py [--hits_csv PATH] [--top_k 20] [--out_dir PATH]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

from src.config import CHECKPOINT_DIR
from src.preprocessing.graph_builder import smiles_to_graph
from src.models.encoder import MolecularEncoder, EdgeAttentionConv
from src.models.multi_property_head import MultiPropertyPredictor
from stack_gnn_rf import load_gnn


def extract_attention_weights(model, data, device):
    """
    Forward pass avec extraction des attention weights de chaque couche.

    Returns:
        node_embeddings: [N, hidden_dim] après message passing
        layer_attentions: list of [E, heads] attention weights par couche
        edge_index: [2, E]
    """
    model.eval()
    encoder = model.encoder

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    chirality = data.chirality.to(device) if hasattr(data, 'chirality') and data.chirality is not None else None

    # Embedding initial
    h = encoder.atom_embedding(x)
    if encoder.use_chirality and chirality is not None:
        h = h + encoder.chirality_embedding(chirality.long())

    vn_feat = None
    if encoder.use_virtual_node:
        n_graphs = 1
        vn_feat = encoder.virtual_node.vn_embedding.weight.expand(n_graphs, -1).clone()

    layer_attentions = []

    for conv, norm, drop in zip(encoder.convs, encoder.norms, encoder.drops, strict=False):
        h_res = h
        if encoder.use_virtual_node:
            h, vn_feat = encoder.virtual_node.pool_and_broadcast(h, batch, vn_feat)

        if isinstance(conv, EdgeAttentionConv):
            # Extraire les attention weights manuellement
            H, D = conv.heads, conv.head_dim
            x_i = h[edge_index[1]]  # target
            x_j = h[edge_index[0]]  # source

            xi = conv.lin_i(x_i).view(-1, H, D)
            xj = conv.lin_j(x_j).view(-1, H, D)
            ee = conv.lin_e(edge_attr).view(-1, H, D)

            a = F.leaky_relu(xi + xj + ee, negative_slope=0.2)
            alpha = (a * conv.att).sum(dim=-1)  # [E, heads]
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))
            layer_attentions.append(alpha.detach().cpu().numpy())

            # Forward normal
            h = conv(h, edge_index, edge_attr)
        else:
            h = conv(h, edge_index, edge_attr)

        h = norm(h)
        h = F.silu(h)
        h = drop(h)
        h = h + h_res

    return h.detach().cpu().numpy(), layer_attentions, edge_index.cpu().numpy()


def attention_to_atom_importance(layer_attentions, edge_index, n_atoms):
    """
    Agrège les attention weights en importance par atome.
    Pour chaque atome i, importance = moyenne des alpha entrants sur toutes les couches et têtes.
    """
    importance = np.zeros(n_atoms)
    counts = np.zeros(n_atoms)

    for alpha in layer_attentions:
        # alpha: [E, heads], edge_index: [2, E]
        alpha_mean = alpha.mean(axis=1)  # moyenne sur les têtes
        targets = edge_index[1]  # nœuds cibles
        for e_idx in range(len(targets)):
            importance[targets[e_idx]] += alpha_mean[e_idx]
            counts[targets[e_idx]] += 1

    counts = np.maximum(counts, 1)
    importance = importance / counts
    # Normaliser [0, 1]
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())
    return importance


def attention_to_bond_importance(layer_attentions, edge_index, mol):
    """
    Agrège les attention weights en importance par liaison.
    """
    n_bonds = mol.GetNumBonds()
    bond_importance = np.zeros(n_bonds)
    bond_counts = np.zeros(n_bonds)

    bond_map = {}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond_map[(i, j)] = b.GetIdx()
        bond_map[(j, i)] = b.GetIdx()

    for alpha in layer_attentions:
        alpha_mean = alpha.mean(axis=1)
        for e_idx in range(edge_index.shape[1]):
            src, tgt = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
            if (src, tgt) in bond_map:
                bid = bond_map[(src, tgt)]
                bond_importance[bid] += alpha_mean[e_idx]
                bond_counts[bid] += 1

    bond_counts = np.maximum(bond_counts, 1)
    bond_importance = bond_importance / bond_counts
    if bond_importance.max() > bond_importance.min():
        bond_importance = (bond_importance - bond_importance.min()) / (bond_importance.max() - bond_importance.min())
    return bond_importance


def generate_attention_svg(smiles, atom_importance, bond_importance, out_path, title=""):
    """Génère un SVG avec atomes/liaisons colorés par importance."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # Couleurs : importance haute = rouge, basse = blanc
    atom_colors = {}
    for i in range(min(n_atoms, len(atom_importance))):
        v = float(atom_importance[i])
        atom_colors[i] = (1.0, 1.0 - v * 0.8, 1.0 - v * 0.8)

    bond_colors = {}
    for i in range(min(n_bonds, len(bond_importance))):
        v = float(bond_importance[i])
        bond_colors[i] = (1.0, 1.0 - v * 0.7, 1.0 - v * 0.7)

    # Top-5 atomes les plus importants
    top_atoms = list(np.argsort(-atom_importance[:n_atoms])[:5])

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(600, 400)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    if title:
        opts.legendFontSize = 14

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightBonds=list(range(n_bonds)),
        highlightBondColors=bond_colors,
        legend=title,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)


def interpret_molecule(model, smiles, device, out_dir, idx=0, score=None):
    """Interprète une molécule : attention map + SVG."""
    data = smiles_to_graph(smiles)
    if data is None:
        print(f"  ⚠ SMILES invalide: {smiles[:50]}")
        return None

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    node_emb, layer_attns, edge_idx = extract_attention_weights(model, data, device)
    atom_imp = attention_to_atom_importance(layer_attns, edge_idx, n_atoms)
    bond_imp = attention_to_bond_importance(layer_attns, edge_idx, mol)

    # Top-3 atomes + leur type
    top3_idx = np.argsort(-atom_imp)[:3]
    top3_info = []
    for ai in top3_idx:
        atom = mol.GetAtomWithIdx(int(ai))
        top3_info.append({
            "idx": int(ai),
            "symbol": atom.GetSymbol(),
            "importance": float(atom_imp[ai]),
            "aromatic": atom.GetIsAromatic(),
            "in_ring": atom.IsInRing(),
        })

    # Identifier les sous-structures activantes (atomes > 0.7 importance)
    active_atoms = [int(i) for i in range(n_atoms) if atom_imp[i] > 0.7]
    active_bonds = [int(i) for i in range(mol.GetNumBonds()) if bond_imp[i] > 0.7]

    title = f"Hit #{idx+1}"
    if score is not None:
        title += f" (score={score:.3f})"

    svg_path = out_dir / f"hit_{idx+1:03d}_attention.svg"
    generate_attention_svg(smiles, atom_imp, bond_imp, svg_path, title=title)

    return {
        "rank": idx + 1,
        "smiles": smiles,
        "score": float(score) if score is not None else None,
        "n_atoms": n_atoms,
        "top3_atoms": top3_info,
        "n_active_atoms": len(active_atoms),
        "n_active_bonds": len(active_bonds),
        "active_atom_indices": active_atoms,
        "atom_importance": atom_imp.tolist(),
        "bond_importance": bond_imp.tolist(),
    }


def summarize_pharmacophores(results):
    """Analyse les motifs récurrents dans les sous-structures activantes."""
    from collections import Counter

    atom_type_counter = Counter()
    ring_count = 0
    aromatic_count = 0

    for r in results:
        if r is None:
            continue
        for a in r["top3_atoms"]:
            atom_type_counter[a["symbol"]] += 1
            if a["in_ring"]:
                ring_count += 1
            if a["aromatic"]:
                aromatic_count += 1

    total = sum(atom_type_counter.values())
    print("\n" + "═" * 60)
    print("RÉSUMÉ PHARMACOPHORIQUE (top-3 atomes par hit)")
    print("═" * 60)
    print(f"  Total atomes analysés: {total}")
    for sym, cnt in atom_type_counter.most_common(10):
        print(f"  {sym:>3} : {cnt:4d} ({100*cnt/max(total,1):.1f}%)")
    print(f"  En cycle: {ring_count}/{total} ({100*ring_count/max(total,1):.1f}%)")
    print(f"  Aromatique: {aromatic_count}/{total} ({100*aromatic_count/max(total,1):.1f}%)")
    return dict(atom_type_counter)


def main():
    ap = argparse.ArgumentParser(description="Interprétabilité GNN — attention maps")
    ap.add_argument("--hits_csv", type=str,
                    default=str(CHECKPOINT_DIR / "phase4" / "library_hits_clean.csv"))
    ap.add_argument("--top_k", type=int, default=20,
                    help="Nombre de hits à interpréter")
    ap.add_argument("--model_dir", type=str,
                    default=str(CHECKPOINT_DIR / "model_A_prime_augmented"))
    ap.add_argument("--out_dir", type=str,
                    default=str(CHECKPOINT_DIR / "interpretability"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Charger modèle
    import json as json_mod
    model_dir = Path(args.model_dir)
    with open(model_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json_mod.load(f)
    gnn = load_gnn(manifest["gnn_ckpt"], device)

    # Charger hits
    import pandas as pd
    hits = pd.read_csv(args.hits_csv)
    print(f"Hits chargés: {len(hits)}")

    # Colonnes score
    score_col = None
    for c in ["score_anti_hiv", "score", "rank_avg"]:
        if c in hits.columns:
            score_col = c
            break

    top_k = min(args.top_k, len(hits))
    top_hits = hits.head(top_k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n▶ Interprétation des {top_k} meilleurs hits\n")
    results = []
    for i, row in top_hits.iterrows():
        smi = row["smiles"] if "smiles" in row else row.iloc[0]
        sc = float(row[score_col]) if score_col else None
        print(f"  Hit {i+1}/{top_k}: {smi[:60]}...")
        r = interpret_molecule(gnn, smi, device, out_dir, idx=i, score=sc)
        results.append(r)

    valid = [r for r in results if r is not None]
    print(f"\n✅ {len(valid)}/{top_k} hits interprétés")

    # Résumé pharmacophorique
    pharma_summary = summarize_pharmacophores(valid)

    # Sauvegarder rapport JSON
    report = {
        "model": str(manifest["gnn_ckpt"]),
        "n_hits_interpreted": len(valid),
        "top_k": top_k,
        "pharmacophore_summary": pharma_summary,
        "hits": valid,
    }
    report_path = out_dir / "interpretation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {report_path}")
    print(f"✅ SVGs: {out_dir}/hit_*_attention.svg")


if __name__ == "__main__":
    main()
