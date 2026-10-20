"""
GraphCL – pré-entraînement contrastif de graphes (You et al. 2020).

Principe : pour chaque molécule on génère DEUX vues augmentées, on les encode,
et on rapproche les deux vues d'une même molécule (positives) tout en éloignant
les autres (négatives) via une perte NT-Xent (InfoNCE).

Augmentations (sans réindexation, donc robustes) :
  - attribute masking : mise à zéro de features de certains atomes,
  - edge dropping     : suppression aléatoire d'arêtes.

L'encodeur ainsi pré-entraîné se sauvegarde au MÊME format que la Phase 1 MGM
(model_state_dict avec préfixe "encoder."), donc la Phase 2 le charge tel quel.
"""
import json
import math
import warnings
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from src.models.encoder import MolecularEncoder
from src.preprocessing.graph_builder import smiles_to_graph

# ──────────────────────────────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────────────────────────────

def augment_graph(data: Data, edge_drop: float = 0.15, attr_mask: float = 0.15) -> Data:
    if not 0 <= edge_drop <= 1 or not 0 <= attr_mask <= 1:
        raise ValueError("Les probabilités d'augmentation doivent être dans [0, 1]")
    x = data.x.clone()

    # Attribute masking : met à zéro les features de quelques atomes
    n = x.size(0)
    if attr_mask > 0 and n > 1:
        k = max(1, int(n * attr_mask))
        idx = torch.randperm(n, device=x.device)[:k]
        x[idx] = 0.0

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # Edge dropping : supprime une fraction d'arêtes
    e = edge_index.size(1)
    if edge_drop > 0 and e > 2:
        # A chemical bond is represented by two directed edges; drop both.
        pairs = edge_index.t().sort(dim=1).values
        _, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        keep_pairs = torch.rand(int(inverse.max()) + 1, device=edge_index.device) > edge_drop
        if not keep_pairs.any():
            keep_pairs[0] = True
        keep = keep_pairs[inverse]
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphCLDataset(Dataset):
    """Retourne deux vues augmentées de chaque molécule."""

    def __init__(self, smiles_list, edge_drop=0.15, attr_mask=0.15):
        self.smiles_list = [s for s in smiles_list if smiles_to_graph(s) is not None]
        self.invalid_count = len(smiles_list) - len(self.smiles_list)
        if self.invalid_count:
            warnings.warn(f"GraphCL : {self.invalid_count} SMILES invalides exclus", stacklevel=2)
        if not self.smiles_list:
            raise ValueError("GraphCL : aucun SMILES valide")
        self.edge_drop = edge_drop
        self.attr_mask = attr_mask

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        g = smiles_to_graph(self.smiles_list[idx])
        if g is None:
            raise RuntimeError("SMILES devenu invalide après validation du dataset")
        v1 = augment_graph(g, self.edge_drop, self.attr_mask)
        v2 = augment_graph(g, self.edge_drop, self.attr_mask)
        return v1, v2


def collate_two_views(batch):
    v1, v2 = zip(*batch, strict=False)
    return Batch.from_data_list(list(v1)), Batch.from_data_list(list(v2))


# ──────────────────────────────────────────────────────────────────────
# Modèle + perte
# ──────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.SiLU(), nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraphCLModel(nn.Module):
    def __init__(self, encoder: MolecularEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionHead(encoder.output_dim, proj_dim)

    def forward(self, batch):
        emb = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.proj(emb)


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    """InfoNCE symétrique : positives = (i, i+N)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Température contrastive positive et finie requise")
    if z1.ndim != 2 or z1.shape != z2.shape or z1.shape[0] < 2:
        raise ValueError("Deux vues de même taille avec au moins deux graphes requises")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    n = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                    # [2N, D]
    sim = (z @ z.t()) / temperature                   # [2N, 2N]
    sim.fill_diagonal_(float("-inf"))                 # exclure self-similarité
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


# ──────────────────────────────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────────────────────────────

def run_graphcl_pretraining(smiles_list, args, device, save_dir, config):
    """Pré-entraîne l'encodeur par GraphCL, sauvegarde au format Phase 1."""
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

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / config["checkpoint_name"]

    ds = GraphCLDataset(smiles_list)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_two_views, drop_last=True,
    )

    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )
    model = GraphCLModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"GraphCL : {len(smiles_list)} molecules, conv={CONV_TYPE}, AMP={use_amp}")
    best_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"[GraphCL] Epoch {epoch}")
        for v1, v2 in pbar:
            v1 = v1.to(device)
            v2 = v2.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = nt_xent_loss(z1.float(), z2.float())
                if not torch.isfinite(loss).all():
                    raise RuntimeError("GraphCL : loss non finie")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2)
                if not torch.isfinite(loss).all():
                    raise RuntimeError("GraphCL : loss non finie")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
                optimizer.step()

            total += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if n == 0:
            raise RuntimeError("GraphCL : aucun batch valide pendant l'epoch")
        avg = total / n
        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))
        print(f"Epoch {epoch:3d}/{args.epochs} | nt_xent={avg:.4f} | ETA {eta}")
        history.append({"epoch": epoch, "loss": avg})

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),  # contient encoder.* et proj.*
                "loss": avg,
                "objective": "graphcl",
                "config": {
                    "hidden_dim": HIDDEN_DIM, "num_layers": NUM_GNN_LAYERS,
                    "output_dim": OUTPUT_DIM, "conv_type": CONV_TYPE,
                    "attention_heads": ATTENTION_HEADS,
                },
            }, best_path)
            print(f"  -> meilleur modele GraphCL sauvegarde ({best_path.name})")

    print(f"\nGraphCL termine en {timedelta(seconds=int(time.time() - t0))} | best nt_xent={best_loss:.4f}")
    log_path = save_dir / f"graphcl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Log -> {log_path}")
