# -*- coding: utf-8 -*-
"""Multi-GPU Phase 1 (opt-in) — propriétés vérifiables SANS 2 GPU.

On valide les invariants critiques du chemin parallèle :
  1. le wrapper produit EXACTEMENT les mêmes clés de state_dict que le modèle
     mono-GPU → checkpoints compatibles Phase 2/3 (pas de préfixe « module. ») ;
  2. PyG décale bien « masked_node_index » en indices GLOBAUX à la collation ;
  3. le forward du wrapper renvoie des (logits, types) ALIGNÉS.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit")
pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from src.config import (  # noqa: E402
    ATOM_FEATURE_DIM,
    ATTENTION_HEADS,
    BOND_FEATURE_DIM,
    CONV_TYPE,
    DROPOUT,
    HIDDEN_DIM,
    NUM_GNN_LAYERS,
    OUTPUT_DIM,
)
from src.models.encoder import MolecularEncoder  # noqa: E402
from src.models.mgm_head import MaskedGraphModel, MGMHead  # noqa: E402
from src.preprocessing.graph_builder import ATOM_TYPE_VOCAB_SIZE  # noqa: E402
from src.training.pretrain_gnn import (  # noqa: E402
    MGMParallelWrapper,
    PretrainDataset,
    _multi_gpu_active,
    _unwrap,
)


def _make_modules():
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_GNN_LAYERS,
        edge_dim=BOND_FEATURE_DIM, output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )
    mgm_head = MGMHead(hidden_dim=HIDDEN_DIM, num_classes=ATOM_TYPE_VOCAB_SIZE)
    return encoder, mgm_head


def test_multi_gpu_inactive_without_env_or_cuda():
    # CPU → jamais actif, quel que soit l'env.
    os.environ.pop("PANACEE_MULTI_GPU", None)
    assert _multi_gpu_active(torch.device("cpu")) is False
    os.environ["PANACEE_MULTI_GPU"] = "1"
    try:
        assert _multi_gpu_active(torch.device("cpu")) is False  # pas de CUDA
    finally:
        os.environ.pop("PANACEE_MULTI_GPU", None)


def test_unwrap_plain_and_wrapped():
    encoder, mgm_head = _make_modules()
    plain = MaskedGraphModel(encoder, mgm_head)
    assert _unwrap(plain) is plain                      # pas de .module

    class _FakeDP(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

    wrapped = _FakeDP(plain)
    assert _unwrap(wrapped) is plain                    # déballe .module


def test_wrapper_state_dict_keys_match_single_gpu():
    """Invariant CLÉ : mêmes clés → checkpoint chargeable par Phase 2/3."""
    encoder, mgm_head = _make_modules()
    single = MaskedGraphModel(encoder, mgm_head)
    wrapper = MGMParallelWrapper(encoder, mgm_head)
    assert set(single.state_dict().keys()) == set(wrapper.state_dict().keys())
    # et l'encodeur garde le préfixe « encoder. » attendu par les phases suivantes
    assert any(k.startswith("encoder.") for k in wrapper.state_dict())


def test_pyg_offsets_masked_node_index_to_global():
    """PyG doit décaler masked_node_index (local) en indices GLOBAUX au batch."""
    from torch_geometric.data import Batch

    ds = PretrainDataset(["CCO", "c1ccccc1", "CCN"], mask_prob=0.5,
                         for_parallel=True, desc="test")
    items = [ds[i] for i in range(len(ds))]
    batch = Batch.from_data_list(items)

    # Reconstruire l'offset attendu (somme cumulée des nœuds).
    offsets, acc = [], 0
    for it in items:
        offsets.append(acc)
        acc += it.x.size(0)
    expected = torch.cat([items[i].masked_node_index + offsets[i]
                          for i in range(len(items))])

    assert torch.equal(batch.masked_node_index, expected)
    assert int(batch.masked_node_index.max()) < batch.x.size(0)   # indices valides
    total = sum(it.masked_node_index.numel() for it in items)
    assert batch.masked_types.numel() == total                    # cibles alignées


def test_wrapper_forward_returns_aligned_logits_and_types():
    from torch_geometric.data import Batch

    encoder, mgm_head = _make_modules()
    wrapper = MGMParallelWrapper(encoder, mgm_head).eval()

    ds = PretrainDataset(["CCO", "c1ccccc1", "CCN", "CCC"], mask_prob=0.5,
                         for_parallel=True, desc="test")
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))])

    with torch.no_grad():
        logits, types = wrapper(batch)
    assert logits.shape[0] == types.shape[0]              # alignés
    assert logits.shape[1] == ATOM_TYPE_VOCAB_SIZE
    assert types.dtype == torch.long
