"""A tiny, synthetic CPU launch exercises the real Phase 3 entry point."""

import hashlib
import json
import sys

import pandas as pd
import torch

from src.models.encoder import MolecularEncoder
from src.preprocessing.multi_property_loader import _DATASET_FILES
from src.training import train_phase3


def test_phase3_complete_launch_with_chiral_schema(tmp_path, monkeypatch):
    train = ["CCO", "CCC", "CCN", "CC(=O)O"]
    val = ["c1ccccc1", "C1CCCCC1"]
    test = ["c1ccncc1", "C1CCCC1"]
    all_smiles = train + val + test
    paths = {}
    raw_hashes = {}
    split_hashes = {}
    for name, (filename, raw_columns) in _DATASET_FILES.items():
        folder = tmp_path / "data" / name
        folder.mkdir(parents=True)
        raw = pd.DataFrame({"smiles": all_smiles})
        for col in raw_columns:
            raw[col] = [i % 2 if name not in {"esol", "lipo"}
                        else float(i) for i in range(len(all_smiles))]
        raw_path = folder / filename
        raw.to_csv(raw_path, index=False)
        raw_hashes[name] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        paths[name] = {}
        for split, smiles in (("train", train), ("val", val), ("test", test)):
            frame = raw[raw.smiles.isin(smiles)].copy()
            if name != "tox21":
                frame = frame.rename(columns={col: f"{name}_{col}" for col in raw_columns})
            path = folder / f"{name}_{split}.csv"
            frame.to_csv(path, index=False)
            paths[name][split] = str(path)
            split_hashes[f"{name}_{split}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    data_dir = tmp_path / "data"
    (data_dir / "data_paths.json").write_text(json.dumps(paths), encoding="utf-8")
    (data_dir / "split_manifest.json").write_text(json.dumps({
        "sha256_generated_splits": split_hashes,
        "sha256_raw_datasets": raw_hashes,
    }), encoding="utf-8")

    for name, value in (("HIDDEN_DIM", 8), ("OUTPUT_DIM", 8),
                        ("NUM_GNN_LAYERS", 1), ("ATTENTION_HEADS", 2)):
        monkeypatch.setattr(train_phase3, name, value)
    for name, value in (("freeze_encoder_epochs", 0), ("warmup_epochs", 1),
                        ("reasoner_hidden_dim", 8), ("reasoner_num_heads", 2),
                        ("reasoner_num_layers", 1), ("max_molecules_combo", 2),
                        ("gradient_accumulation_steps", 1), ("ema_decay", 0)):
        monkeypatch.setitem(train_phase3.PHASE3, name, value)
    monkeypatch.setattr(train_phase3, "LOG_DIR", tmp_path / "logs")
    legacy_encoder = MolecularEncoder(hidden_dim=8, output_dim=8,
                                      num_layers=1, attention_heads=2)
    phase2_path = tmp_path / "phase2.pth"
    torch.save({"model_state_dict": {
        f"encoder.{name}": value for name, value in legacy_encoder.state_dict().items()
    }}, phase2_path)
    save_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_phase3.py", "--data_dir", str(data_dir),
                    "--pretrained_model", str(phase2_path), "--save_dir", str(save_dir),
                    "--epochs", "1", "--patience", "1", "--batch_size", "2",
                    "--save_epochs", "0", "--warm_start", "0", "--device", "cpu"])
    train_phase3.main()
    checkpoint = torch.load(save_dir / "panacee_phase3_complete.pth",
                            map_location="cpu", weights_only=False)
    assert checkpoint["training_stage"] == "complete"
    assert checkpoint["config"]["feature_schema"] == "atom9_bond6_cip_atom_v2"
    assert checkpoint["selection_policy"] == "multitask"
    assert checkpoint["epoch"] == 1
