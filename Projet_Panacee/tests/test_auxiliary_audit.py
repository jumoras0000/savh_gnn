import sqlite3

import numpy as np
import pytest
import torch


def test_database_closes_connections_and_prevents_orphans(tmp_path, monkeypatch):
    from webapp import store
    monkeypatch.setenv("PANACEE_DB", str(tmp_path / "chat.db"))
    assert store.conversation_exists("missing") is False
    with store._conn() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_message("missing", "user", "hello")
    conversation = store.create_conversation()
    store.add_message(conversation["id"], "user", "hello")
    assert len(store.get_messages(conversation["id"])) == 1
    store.delete_conversation(conversation["id"])
    assert store.get_messages(conversation["id"]) == []


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], np.zeros((1, 1, 1))])
def test_uncertainty_refuses_empty_nonfinite_or_wrong_shape(samples):
    from src.utils.uncertainty import summarize
    with pytest.raises(ValueError):
        summarize(samples)


def test_graphcl_filters_invalid_molecules_without_substituting_other_samples():
    from src.training.graphcl import GraphCLDataset
    dataset = GraphCLDataset(["CCO", "", "CCN"])
    assert len(dataset) == 2
    assert dataset.invalid_count == 1
    assert dataset.smiles_list == ["CCO", "CCN"]


def test_graphcl_keeps_bonds_bidirectional():
    from src.preprocessing.graph_builder import smiles_to_graph
    from src.training.graphcl import augment_graph
    graph = smiles_to_graph("CCCOCC")
    for seed in range(5):
        torch.manual_seed(seed)
        view = augment_graph(graph, edge_drop=0.7)
        edges = set(map(tuple, view.edge_index.t().tolist()))
        assert all((b, a) in edges for a, b in edges)


@pytest.mark.parametrize("temperature", [0, -1, float("nan")])
def test_graphcl_rejects_invalid_temperature(temperature):
    from src.training.graphcl import nt_xent_loss
    with pytest.raises(ValueError):
        nt_xent_loss(torch.ones(2, 4), torch.ones(2, 4), temperature)


def test_graphcl_rejects_zero_information_singleton_batch():
    from src.training.graphcl import nt_xent_loss
    with pytest.raises(ValueError):
        nt_xent_loss(torch.ones(1, 4), torch.ones(1, 4))


def test_phase2_nonfinite_gradient_cannot_update_parameters():
    from src.training.finetune_toxicity import train_one_epoch
    class Batch:
        def to(self, device):
            return self
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.weight.register_hook(lambda gradient: gradient * float("inf"))
        def forward(self, batch):
            return self.weight.expand(2, 1)
    model = Model()
    with pytest.raises(RuntimeError, match="non-finite"):
        train_one_epoch(model, [(Batch(), torch.ones(2, 1))],
                        torch.optim.SGD(model.parameters(), lr=0.1),
                        lambda output, target: output.mean(), torch.device("cpu"), 1.0, 1)
    assert model.weight.item() == 1.0


def test_ensemble_rejects_inconsistent_tasks():
    from src.models.ensemble import DeepEnsemble
    class Model(torch.nn.Module):
        def __init__(self, key):
            super().__init__()
            self.key = key
        def forward(self, batch):
            return {self.key: torch.ones(1, 1)}
    with pytest.raises(ValueError, match="mêmes tâches"):
        DeepEnsemble([Model("a"), Model("b")]).predict(None)


@pytest.mark.parametrize("parallel", [False, True])
def test_phase1_empty_validation_does_not_report_perfect_loss(parallel):
    from src.training import pretrain_gnn
    evaluate = pretrain_gnn.evaluate_parallel if parallel else pretrain_gnn.evaluate
    with pytest.raises(RuntimeError, match="aucun atome"):
        evaluate(torch.nn.Linear(2, 2), [], torch.device("cpu"))


def test_advanced_dose_search_requires_measured_response():
    from src.models.advanced_reasoner import AdvancedMolecularReasoner
    analyzer = AdvancedMolecularReasoner(None, None, torch.device("cpu"))
    with pytest.raises(ValueError, match="dose-réponse"):
        analyzer.full_analysis(torch.zeros(2, 8), use_bayesian=True)
