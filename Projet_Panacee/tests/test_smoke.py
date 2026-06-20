# -*- coding: utf-8 -*-
"""
Smoke-test rapide du pipeline GNN (sans deepchem, ~quelques secondes sur CPU).

Verifie que les briques essentielles fonctionnent end-to-end :
  - construction de graphes (SMILES -> PyG),
  - encodeur (attention + mpnn) + pooling normalise,
  - classifier toxicite + loss masquee + EMA + 1 step backward,
  - scaffold split / k-fold,
  - GraphCL (augment + NT-Xent),
  - tete MGM (Phase 1).

Lancer :  python -m tests.test_smoke    (depuis Projet_Panacee/)
Sortie 0 si tout passe, 1 sinon (utilisable en CI).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Batch

SMILES = [
    "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccccc1", "CCO",
    "C1CCCCC1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "O=C(O)c1ccccc1O", "CCN(CC)CC",
    "C1=CC=C(C=C1)C=O", "CC(=O)Nc1ccc(O)cc1", "OCC(O)CO", "CC(C)(C)O",
]

_ok = True


def check(name, cond):
    global _ok
    print(("  OK   " if cond else "  FAIL ") + name)
    _ok = _ok and bool(cond)


def run():
    from src.preprocessing.graph_builder import smiles_to_graph, mask_atoms
    from src.models.encoder import MolecularEncoder
    from src.models.toxicity_classifier import ToxicityClassifier, MultiTaskBCELoss
    from src.models.mgm_head import MaskedGraphModel, MGMHead
    from src.utils.ema import ModelEMA
    from src.preprocessing.scaffold_split import scaffold_split, scaffold_kfold
    from src.training.graphcl import augment_graph, nt_xent_loss, GraphCLModel, collate_two_views

    print("== graphes ==")
    graphs = [smiles_to_graph(s) for s in SMILES]
    check("tous les SMILES -> graphes", all(g is not None for g in graphs))
    check("features atome=9, liaison=6", graphs[0].x.shape[1] == 9 and graphs[0].edge_attr.shape[1] == 6)
    batch = Batch.from_data_list(graphs)

    print("== encodeur (attention + mpnn) ==")
    for ct in ("attention", "mpnn"):
        enc = MolecularEncoder(num_layers=3, conv_type=ct)
        out = enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        check(f"[{ct}] sortie [B,256] sans NaN",
              tuple(out.shape) == (len(graphs), 256) and not torch.isnan(out).any())

    print("== classifier + loss masquee + EMA + backward ==")
    model = ToxicityClassifier(encoder=MolecularEncoder(num_layers=3), num_tasks=12)
    labels = torch.randint(0, 2, (len(graphs), 12)).float()
    labels[0, 0] = float("nan")
    crit = MultiTaskBCELoss(pos_weight=torch.ones(12))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ModelEMA(model, decay=0.9)
    loss = crit(model(batch), labels)
    loss.backward(); opt.step(); ema.update(model)
    check("loss finie malgre NaN", torch.isfinite(loss).item())
    before = model.classifier[0].weight.detach().clone()
    ema.store(model); ema.copy_to(model)
    changed = not torch.allclose(before, model.classifier[0].weight)
    ema.restore(model)
    check("EMA copy/restore", changed and torch.allclose(before, model.classifier[0].weight))

    print("== scaffold split / kfold ==")
    tr, va = scaffold_split(SMILES, 0.7)
    check("split disjoint et complet", not (set(tr) & set(va)) and len(tr) + len(va) == len(SMILES))
    folds = scaffold_kfold(SMILES, k=3)
    allv = sorted(i for _, v in folds for i in v)
    check("kfold couvre tout 1x", allv == list(range(len(SMILES))))

    print("== GraphCL ==")
    gcl = GraphCLModel(MolecularEncoder(num_layers=2), proj_dim=64)
    b1, b2 = collate_two_views([(augment_graph(g), augment_graph(g)) for g in graphs])
    l = nt_xent_loss(gcl(b1), gcl(b2))
    l.backward()
    check("nt_xent fini + backward", torch.isfinite(l).item() and gcl.proj.net[0].weight.grad is not None)

    print("== MGM head ==")
    gm, mi, _ = mask_atoms(graphs[2], 0.3)
    mgm = MaskedGraphModel(MolecularEncoder(num_layers=2), MGMHead())
    bm = Batch.from_data_list([gm])
    preds = mgm(bm.x, bm.edge_index, bm.edge_attr, bm.batch, [mi])
    check("MGM predit [M,9]", preds.shape[1] == 9 and preds.shape[0] == len(mi))


if __name__ == "__main__":
    run()
    print("\n" + ("==> SMOKE TEST OK" if _ok else "==> SMOKE TEST ECHEC"))
    sys.exit(0 if _ok else 1)
