# -*- coding: utf-8 -*-
"""Tests de l'objectif de pré-entraînement Phase 1 (MGM par classification du type d'atome)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from src.preprocessing.graph_builder import (  # noqa: E402
    ATOM_TYPE_OTHER,
    ATOM_TYPE_VOCAB_SIZE,
    atomic_num_to_class,
    mask_atoms,
    smiles_to_graph,
)


def test_atomic_num_to_class_known_elements():
    assert atomic_num_to_class(6) == 0   # C = première classe
    assert atomic_num_to_class(7) == 1   # N
    assert atomic_num_to_class(8) == 2   # O
    # Élément rare → classe « autre »
    assert atomic_num_to_class(78) == ATOM_TYPE_OTHER  # Pt
    assert atomic_num_to_class(0) == ATOM_TYPE_OTHER


def test_vocab_size_consistent():
    # 12 éléments listés + 1 classe « autre »
    assert ATOM_TYPE_VOCAB_SIZE == ATOM_TYPE_OTHER + 1
    assert ATOM_TYPE_VOCAB_SIZE == 13


def test_mask_atoms_returns_types_matching_elements():
    # Éthanol CCO : 2 carbones + 1 oxygène
    graph = smiles_to_graph("CCO")
    graph_masked, idx, feats, types = mask_atoms(graph, mask_prob=1.0)  # masque tout
    assert types.dtype == torch.long
    assert types.shape[0] == feats.shape[0] == len(idx)
    # Recalcule la classe attendue depuis le numéro atomique d'origine
    expected = [atomic_num_to_class(round(feats[i, 0].item() * 118)) for i in range(feats.size(0))]
    assert types.tolist() == expected
    # Doit contenir au moins du C (classe 0) et de l'O (classe 2)
    assert 0 in types.tolist() and 2 in types.tolist()


def test_mgm_forward_shapes_and_loss():
    from torch_geometric.data import Batch

    from src.config import (
        ATOM_FEATURE_DIM,
        ATTENTION_HEADS,
        BOND_FEATURE_DIM,
        CONV_TYPE,
        DROPOUT,
        HIDDEN_DIM,
        NUM_GNN_LAYERS,
        OUTPUT_DIM,
    )
    from src.models.encoder import MolecularEncoder
    from src.models.mgm_head import MaskedGraphModel, MGMHead

    graphs, idxs, types = [], [], []
    for smi in ("CC(=O)Nc1ccc(O)cc1", "CCO"):
        gm, mi, _mf, mt = mask_atoms(smiles_to_graph(smi), 0.3)
        graphs.append(gm)
        idxs.append(mi)
        types.append(mt)
    batch = Batch.from_data_list(graphs)
    targets = torch.cat(types, dim=0)

    enc = MolecularEncoder(atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
                           num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
                           output_dim=OUTPUT_DIM, dropout=DROPOUT,
                           conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS)
    model = MaskedGraphModel(enc, MGMHead(hidden_dim=HIDDEN_DIM, num_classes=ATOM_TYPE_VOCAB_SIZE))

    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, idxs)
    # Alignement : un logit par atome masqué, vocab_size classes
    assert logits.shape == (targets.shape[0], ATOM_TYPE_VOCAB_SIZE)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss)
