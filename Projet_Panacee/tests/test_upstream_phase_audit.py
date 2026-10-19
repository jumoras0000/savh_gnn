"""Regression checks for Phase 1/2 provenance and selection stability."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.preprocessing.graph_builder import smiles_to_graph
from src.training.finetune_toxicity import (
    phase2_epoch_eligible, phase2_exposure, validate_phase2_warm_start,
)
from src.training.pretrain_gnn import PretrainDataset
from src.training.train_phase3 import audit_phase2_exposure


def test_phase1_validation_masks_are_repeatable_without_freezing_train():
    val = PretrainDataset(["CCN(CC)CCO"], mask_prob=0.8, fixed_mask_seed=42)
    first = val[0]
    torch.rand(100)
    second = val[0]
    assert first[1] == second[1]
    assert torch.equal(first[0].x, second[0].x)
    assert torch.equal(first[2], second[2])
    train = PretrainDataset(["CCN(CC)CCO"], mask_prob=0.8)
    assert train.fixed_mask_seed is None


def test_phase2_exposure_includes_train_and_validation_for_downstream_audit(tmp_path):
    train = [(smiles_to_graph("CCO"), torch.tensor([1.0]))]
    val = [(smiles_to_graph("c1ccccc1"), torch.tensor([0.0]))]
    exposure = phase2_exposure(train, val)
    assert exposure["canonical_smiles"] == ["CCO", "c1ccccc1"]
    assert "" in exposure["scaffolds"]
    assert exposure["includes_validation_selection"] is True

    paths = {}
    for split, smiles in (("val", "OCC"), ("test", "C1CCCCC1")):
        path = tmp_path / f"{split}.csv"
        pd.DataFrame({"smiles": [smiles]}).to_csv(path, index=False)
        paths[split] = str(path)
    with pytest.raises(ValueError, match="Fuite supervisée"):
        audit_phase2_exposure({"phase2_exposure": exposure}, {"tox21": paths})
    assert audit_phase2_exposure({}, {"tox21": paths})["status"] == "unverified"
    clean_exposure = phase2_exposure(
        [(smiles_to_graph("C1CCC1"), torch.tensor([1.0]))],
        [(smiles_to_graph("C1CCCC1"), torch.tensor([0.0]))],
    )
    assert audit_phase2_exposure(
        {"phase2_exposure": clean_exposure}, {"tox21": paths}
    )["status"] == "verified_no_supervised_overlap"


def test_phase2_warm_start_refuses_legacy_or_changed_data():
    exposure = {"canonical_smiles": ["CCO"], "scaffolds": [""],
                "includes_validation_selection": True}
    with pytest.raises(ValueError, match="Warm-start"):
        validate_phase2_warm_start({"model_state_dict": {}}, exposure, ["NR-AR"])
    checkpoint = {"phase2_exposure": exposure, "task_names": ["NR-AR"],
                  "model_state_dict": {"weight": torch.tensor([1.0])}}
    assert validate_phase2_warm_start(checkpoint, exposure, ["NR-AR"])
    with pytest.raises(ValueError, match="Warm-start"):
        validate_phase2_warm_start(checkpoint, exposure, ["NR-AhR"])
    assert not phase2_epoch_eligible(3, 10)
    assert phase2_epoch_eligible(11, 10)


def test_phase2_pretraining_checkpoint_rejects_incomplete_encoder(tmp_path):
    from src.models.encoder import MolecularEncoder
    from src.training.finetune_toxicity import _load_pretrained_encoder

    path = tmp_path / "bad_phase1.pth"
    torch.save({"model_state_dict": {"encoder.atom_embedding.0.weight":
                                     torch.zeros(8, 9)}}, path)
    encoder = MolecularEncoder(hidden_dim=8, output_dim=8, num_layers=1,
                               attention_heads=2)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_pretrained_encoder(encoder, path)


def test_aligned_phase2_csvs_are_subsets_of_phase3_train(tmp_path, monkeypatch):
    import run_phase2
    from src.preprocessing.multi_property_loader import _DATASET_FILES

    data_dir = tmp_path / "phase3"
    data_dir.mkdir()
    train = [f"C1{'C' * n}1" for n in range(2, 12)]
    val, test = ["c1ccccc1"], ["c1ccncc1"]
    paths, hashes, raw_hashes = {}, {}, {}
    for name in ("tox21", "esol", "lipo", "bbbp", "clintox", "hiv"):
        paths[name] = {}
        raw_dir = data_dir / name
        raw_dir.mkdir()
        raw_file = raw_dir / _DATASET_FILES[name][0]
        pd.DataFrame({"smiles": train + val + test}).to_csv(raw_file, index=False)
        raw_hashes[name] = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        for split, smiles in (("train", train), ("val", val), ("test", test)):
            path = data_dir / f"{name}_{split}.csv"
            pd.DataFrame({"smiles": smiles, "NR-AR": [i % 2 for i in range(len(smiles))]}).to_csv(
                path, index=False)
            paths[name][split] = str(path)
            hashes[f"{name}_{split}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data_dir / "data_paths.json").write_text(json.dumps(paths), encoding="utf-8")
    (data_dir / "split_manifest.json").write_text(json.dumps(
        {"sha256_generated_splits": hashes, "sha256_raw_datasets": raw_hashes}),
        encoding="utf-8")
    monkeypatch.setattr(run_phase2, "EXTERNAL_DIR", tmp_path / "generated")
    train_path, val_path = run_phase2.prepare_phase2_from_phase3(data_dir)
    phase2_smiles = set(pd.read_csv(train_path).smiles) | set(pd.read_csv(val_path).smiles)
    assert phase2_smiles == set(train)
    assert not phase2_smiles.intersection(val + test)


def test_phase1_one_epoch_cpu_checkpoint(tmp_path, monkeypatch):
    from src.training import pretrain_gnn

    data_path = tmp_path / "pretrain.pt"
    torch.save({"smiles": ["CCO", "CCN", "CCC", "CCCl", "CCF", "CCS", "COC", "CNC"]},
               data_path)
    for name, value in (("HIDDEN_DIM", 8), ("OUTPUT_DIM", 8),
                        ("NUM_GNN_LAYERS", 1), ("ATTENTION_HEADS", 2)):
        monkeypatch.setattr(pretrain_gnn, name, value)
    monkeypatch.setitem(pretrain_gnn.PHASE1, "val_split", 0.25)
    monkeypatch.setitem(pretrain_gnn.PHASE1, "warmup_epochs", 1)
    monkeypatch.setattr(pretrain_gnn, "LOG_DIR", tmp_path / "logs")
    save_dir = tmp_path / "phase1"
    monkeypatch.setattr(sys, "argv", ["pretrain_gnn.py", "--data_path", str(data_path),
                    "--save_dir", str(save_dir), "--epochs", "1", "--batch_size", "2",
                    "--device", "cpu"])
    pretrain_gnn.main()
    checkpoint = torch.load(save_dir / pretrain_gnn.PHASE1["checkpoint_name"],
                            map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["model_state_dict"]


def test_phase2_one_epoch_cpu_checkpoint_records_exposure(tmp_path, monkeypatch):
    from src.preprocessing.toxicity_loader import ToxicityDataset
    from src.training import finetune_toxicity

    train_path, val_path = tmp_path / "train.csv", tmp_path / "val.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN", "CCC", "CCF"],
                  "NR-AR": [0, 1, 0, 1]}).to_csv(train_path, index=False)
    pd.DataFrame({"smiles": ["c1ccccc1", "C1CCCCC1"],
                  "NR-AR": [0, 1]}).to_csv(val_path, index=False)
    train_ds = ToxicityDataset(str(train_path))
    val_ds = ToxicityDataset(str(val_path), task_columns=train_ds.get_task_names())
    for name, value in (("HIDDEN_DIM", 8), ("OUTPUT_DIM", 8),
                        ("NUM_GNN_LAYERS", 1), ("ATTENTION_HEADS", 2)):
        monkeypatch.setattr(finetune_toxicity, name, value)
    monkeypatch.setitem(finetune_toxicity.PHASE2, "freeze_encoder_epochs", 0)
    monkeypatch.setitem(finetune_toxicity.PHASE2, "warmup_epochs", 1)
    args = SimpleNamespace(batch_size=2, epochs=1, patience=1, pretrained_model=None,
                           warm_start=0, save_epochs=0)
    save_dir = tmp_path / "phase2"
    finetune_toxicity.train_one_run(
        train_ds, val_ds, 1, ["NR-AR"], train_ds.get_pos_weight(),
        args, torch.device("cpu"), str(save_dir), use_ema=False,
    )
    checkpoint = torch.load(save_dir / "best_toxicity_model.pth",
                            map_location="cpu", weights_only=False)
    assert checkpoint["selection_eligible"] is True
    assert len(checkpoint["phase2_exposure"]["canonical_smiles"]) == 6


def test_phase2_rejects_nonbinary_labels_before_training(tmp_path):
    from src.preprocessing.toxicity_loader import ToxicityDataset

    path = tmp_path / "bad_labels.csv"
    pd.DataFrame({"smiles": ["CCO", "CCN"], "NR-AR": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non binaire"):
        ToxicityDataset(str(path), task_columns=["NR-AR"])
