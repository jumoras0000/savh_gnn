"""Small real GNN checkpoint/inference tests (CPU, synthetic weights/data)."""
import math

import pytest
import torch

from src.analysis.combinatorial_engine import PanaceeAnalyzer
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor
from src.models.reasoner import MolecularReasoner
from src.preprocessing.graph_builder import smiles_to_graph


@pytest.fixture
def tiny_checkpoint(tmp_path):
    encoder = MolecularEncoder(hidden_dim=8, output_dim=4, num_layers=1,
                               attention_heads=2, use_virtual_node=False)
    model = MultiPropertyPredictor(encoder, hidden_dim=8)
    reasoner = MolecularReasoner(mol_dim=8, hidden_dim=8, num_heads=2,
                                num_layers=1, max_molecules=3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "reasoner_state_dict": reasoner.state_dict(),
        "training_stage": "complete",
        "config": {"hidden_dim": 8, "output_dim": 4, "num_gnn_layers": 1,
                   "attention_heads": 2, "use_virtual_node": False},
        "reasoner_config": {"hidden_dim": 8, "num_heads": 2, "num_layers": 1,
                            "max_molecules": 3},
        "temperatures": {"efficacy": 2.0},
        "regression_stats": {"solubility": {"mean": -3.0, "std": 2.0}},
    }
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)
    return path


def test_single_molecule_mc_dropout_does_not_train_batchnorm(tiny_checkpoint):
    analyzer = PanaceeAnalyzer(str(tiny_checkpoint), device="cpu")
    before = {name: value.clone() for name, value in analyzer.model.named_buffers()}
    result = analyzer.predict_properties("CCO", mc_samples=1)
    assert math.isfinite(result["efficacy"]["probabilite_activite"])
    assert result["uncertainty"]["efficacy_std"] == 0.0
    for name, value in analyzer.model.named_buffers():
        assert torch.equal(value, before[name])
    assert not analyzer.model.training


def test_ensemble_preserves_checkpoint_architecture_and_probability_units(tiny_checkpoint):
    analyzer = PanaceeAnalyzer(str(tiny_checkpoint), device="cpu")
    single = analyzer.predict_properties("CCO")
    analyzer.load_ensemble([str(tiny_checkpoint), str(tiny_checkpoint)])
    result = analyzer.predict_properties("CCO")
    assert result["efficacy"] == single["efficacy"]
    assert result["uncertainty"]["efficacy_std"] == 0


def test_ensemble_rejects_mismatched_normalization(tiny_checkpoint, tmp_path):
    from src.utils.safe_load import safe_load_checkpoint
    checkpoint = safe_load_checkpoint(tiny_checkpoint)
    checkpoint["regression_stats"]["solubility"]["std"] = 9.0
    other = tmp_path / "other.pth"
    torch.save(checkpoint, other)
    analyzer = PanaceeAnalyzer(str(tiny_checkpoint), device="cpu")
    with pytest.raises(ValueError, match="regression_stats"):
        analyzer.load_ensemble([str(tiny_checkpoint), str(other)])


def test_untrained_synergy_dose_confidence_are_not_presented(tiny_checkpoint):
    analyzer = PanaceeAnalyzer(str(tiny_checkpoint), device="cpu")
    result = analyzer.analyze_combination(["CCO", "CCN"])
    assert result["dose_recommendations"] == []
    assert result["synergy_matrix"] is None
    assert result["synergy_pairs"] == []
    assert result["confidence"] is None
    with pytest.raises(ValueError, match="Maximum"):
        analyzer.analyze_combination(["C", "CC", "CCC", "CCCC"])


@pytest.mark.parametrize("smiles", ["", "   ", None, 42])
def test_empty_or_nonstring_smiles_cannot_create_graph(smiles):
    assert smiles_to_graph(smiles) is None


def test_chiral_encoder_distinguishes_enantiomers_without_changing_legacy_schema():
    from torch_geometric.data import Batch

    right = smiles_to_graph("N[C@H](C)C(=O)O")
    left = smiles_to_graph("N[C@@H](C)C(=O)O")
    assert torch.equal(right.x, left.x)
    assert torch.equal(right.edge_index, left.edge_index)
    assert torch.equal(right.edge_attr, left.edge_attr)
    assert not torch.equal(right.chirality, left.chirality)

    torch.manual_seed(17)
    old = MolecularEncoder(hidden_dim=8, output_dim=8, num_layers=1,
                           attention_heads=2, dropout=0, use_virtual_node=False)
    upgraded = MolecularEncoder(hidden_dim=8, output_dim=8, num_layers=1,
                                attention_heads=2, dropout=0, use_virtual_node=False,
                                use_chirality=True)
    result = upgraded.load_state_dict(old.state_dict(), strict=False)
    assert result.missing_keys == ["chirality_embedding.weight"]
    assert result.unexpected_keys == []
    old.eval()
    upgraded.eval()
    with torch.no_grad():
        old_outputs = [old(b.x, b.edge_index, b.edge_attr,
                           torch.zeros(b.x.size(0), dtype=torch.long)) for b in (right, left)]
        new_outputs = [upgraded(b.x, b.edge_index, b.edge_attr,
                                torch.zeros(b.x.size(0), dtype=torch.long), b.chirality)
                       for b in (right, left)]
    assert torch.allclose(old_outputs[0], old_outputs[1])
    assert not torch.allclose(new_outputs[0], new_outputs[1])
    batch = Batch.from_data_list([right, left])
    assert batch.chirality.numel() == batch.x.size(0)


def test_chiral_phase3_checkpoint_reloads_with_explicit_schema(tmp_path):
    encoder = MolecularEncoder(hidden_dim=8, output_dim=4, num_layers=1,
                               attention_heads=2, use_virtual_node=False,
                               use_chirality=True)
    model = MultiPropertyPredictor(encoder, hidden_dim=8)
    checkpoint = {"model_state_dict": model.state_dict(),
                  "training_stage": "complete",
                  "config": {"hidden_dim": 8, "output_dim": 4,
                             "num_gnn_layers": 1, "attention_heads": 2,
                             "use_virtual_node": False, "use_chirality": True,
                             "feature_schema": "atom9_bond6_cip_atom_v2"}}
    path = tmp_path / "chiral.pth"
    torch.save(checkpoint, path)
    analyzer = PanaceeAnalyzer(str(path), device="cpu")
    assert analyzer.encoder.use_chirality
    assert analyzer.predict_properties("N[C@H](C)C(=O)O") is not None


def test_reasoner_rejects_all_masked_or_oversized_input():
    model = MolecularReasoner(mol_dim=8, hidden_dim=8, num_heads=2, num_layers=1, max_molecules=3)
    with pytest.raises(ValueError, match="masquée"):
        model(torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="limites"):
        model(torch.zeros(1, 4, 8))


def test_phase2_unfreezes_virtual_node():
    from src.models.toxicity_classifier import ToxicityClassifier
    encoder = MolecularEncoder(hidden_dim=8, output_dim=8, num_layers=2, attention_heads=2)
    model = ToxicityClassifier(encoder=encoder, hidden_dim=8, freeze_encoder=True)
    model.train()
    assert not encoder.training
    model.gradual_unfreeze(11, 10)
    assert not any(p.requires_grad for p in encoder.convs[0].parameters())
    model.gradual_unfreeze(12, 10)
    assert all(p.requires_grad for p in encoder.virtual_node.parameters())


def test_real_gnn_training_unfreezes_encoder_and_evaluates():
    from torch.utils.data import DataLoader
    from src.models.multi_property_head import MultiPropertyLoss
    from src.preprocessing.multi_property_loader import collate_multi_property
    from src.training.train_phase3 import train_one_epoch, evaluate, phase3_parameter_groups
    torch.manual_seed(7)
    encoder = MolecularEncoder(hidden_dim=8, output_dim=4, num_layers=1,
                               attention_heads=2, dropout=0)
    model = MultiPropertyPredictor(encoder, hidden_dim=8, dropout=0, freeze_encoder=True)
    criterion = MultiPropertyLoss()
    groups = phase3_parameter_groups(model, 0.001, 0.0) + phase3_parameter_groups(criterion, 0.001, 0.0)
    optimizer = torch.optim.AdamW(groups)
    samples = []
    for i, smiles in enumerate(["CCO", "CCN", "CCC", "CC(=O)O"]):
        target = {name: torch.tensor([float(i % 2)]) for name in model.head_names}
        target["toxicity"] = target["toxicity"].expand(12)
        samples.append((smiles_to_graph(smiles), target))
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_multi_property)
    initial = encoder.atom_embedding[0].weight.detach().clone()
    model.gradual_unfreeze(1, 0)
    ema = torch.optim.swa_utils.AveragedModel(model, use_buffers=True)
    metrics = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"),
                              1.0, 1, accum_steps=2, ema_model=ema)
    assert metrics["_optimizer_updates"] == 1
    assert not torch.equal(initial, encoder.atom_embedding[0].weight)
    val_metrics, predictions, _ = evaluate(ema, loader, criterion, torch.device("cpu"))
    assert math.isfinite(val_metrics["_loss"])
    assert all(torch.isfinite(value).all() for value in predictions.values())
