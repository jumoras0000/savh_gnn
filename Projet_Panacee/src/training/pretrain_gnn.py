"""
Phase 1 - Pre-entrainement MGM (Masked Graph Modeling) - v2.

Corrections :
  1. Warmup lineaire + cosine annealing scheduler.
  2. Grad clipping configurable.
  3. Early stopping sur val loss.
  4. Utilise MaskedGraphModel._encode_nodes (node-level).
  5. Toutes les constantes lues depuis config.py.
"""
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -- Path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ATOM_FEATURE_DIM,
    ATTENTION_HEADS,
    BOND_FEATURE_DIM,
    CHECKPOINT_DIR,
    CONV_TYPE,
    DEVICE,
    DROPOUT,
    HIDDEN_DIM,
    LOG_DIR,
    NUM_GNN_LAYERS,
    NUM_WORKERS,
    OUTPUT_DIM,
    PHASE1,
    PIN_MEMORY,
)
from src.models.encoder import MolecularEncoder
from src.models.mgm_head import MaskedGraphModel, MGMHead
from src.preprocessing.graph_builder import (
    ATOM_TYPE_VOCAB_SIZE,
    mask_atoms,
    smiles_to_graph,
)
from src.utils.live_logger import LiveLogger

# ======================================================================
# Dataset
# ======================================================================

class PretrainDataset(Dataset):
    def __init__(self, smiles_list, mask_prob=0.15):
        self.smiles_list = smiles_list
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        graph = smiles_to_graph(self.smiles_list[idx])
        if graph is None:
            for j in range(1, len(self.smiles_list)):
                graph = smiles_to_graph(self.smiles_list[(idx + j) % len(self.smiles_list)])
                if graph is not None:
                    break
            if graph is None:
                raise RuntimeError("Aucun SMILES valide dans le dataset")
        graph_masked, masked_idx, _masked_feat, masked_types = mask_atoms(graph, self.mask_prob)
        return graph_masked, masked_idx, masked_types


def collate_fn(batch):
    from torch_geometric.data import Batch
    graphs, indices, types = zip(*batch, strict=False)
    batch_graph = Batch.from_data_list(graphs)
    all_types = torch.cat(types, dim=0)
    return batch_graph, list(indices), all_types


# ======================================================================
# Scheduler avec warmup
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            alpha = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = self.lr_min + (base_lr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Train / Eval loops
# ======================================================================

def train_one_epoch(model, loader, optimizer, device, grad_clip, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    n_atoms = 0
    n_correct = 0
    use_amp = scaler is not None and scaler.is_enabled()
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_graph, masked_indices, masked_types in pbar:
        batch_graph = batch_graph.to(device)
        masked_types = masked_types.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(
                    batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr,
                    batch_graph.batch, masked_indices,
                )
                loss = criterion(logits.float(), masked_types)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(
                batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr,
                batch_graph.batch, masked_indices,
            )
            loss = criterion(logits, masked_types)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = masked_types.size(0)
        total_loss += loss.item() * bs
        n_atoms += bs
        n_correct += (logits.argmax(dim=1) == masked_types).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{n_correct / max(n_atoms, 1):.3f}")

    return total_loss / max(n_atoms, 1), n_correct / max(n_atoms, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_atoms = 0
    n_correct = 0
    criterion = nn.CrossEntropyLoss()

    for batch_graph, masked_indices, masked_types in loader:
        batch_graph = batch_graph.to(device)
        masked_types = masked_types.to(device)

        logits = model(
            batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr,
            batch_graph.batch, masked_indices,
        )
        loss = criterion(logits, masked_types)
        bs = masked_types.size(0)
        total_loss += loss.item() * bs
        n_atoms += bs
        n_correct += (logits.argmax(dim=1) == masked_types).sum().item()

    return total_loss / max(n_atoms, 1), n_correct / max(n_atoms, 1)


# ======================================================================
# Main
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 1 - Pre-training MGM")
    p.add_argument("--data_path", type=str, required=True, help="pretrain_dataset.pt")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase1"))
    p.add_argument("--epochs", type=int, default=PHASE1["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE1["batch_size"])
    p.add_argument("--lr", type=float, default=PHASE1["lr"])
    p.add_argument("--mask_prob", type=float, default=PHASE1["mask_prob"])
    p.add_argument("--patience", type=int, default=PHASE1["patience"])
    p.add_argument("--max_molecules", type=int, default=None)
    p.add_argument("--objective", type=str, default="mgm", choices=["mgm", "graphcl"],
                   help="mgm = masked graph modeling | graphcl = contrastif")
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase1"
    log_dir.mkdir(parents=True, exist_ok=True)

    # -- Donnees --
    print("Chargement du dataset ...")
    data = torch.load(args.data_path, weights_only=False)
    smiles_list = data["smiles"]
    if args.max_molecules:
        smiles_list = smiles_list[: args.max_molecules]
    print(f"  {len(smiles_list)} molecules | objectif = {args.objective}")

    # -- Branche GraphCL (pre-entrainement contrastif) --
    if args.objective == "graphcl":
        from src.training.graphcl import run_graphcl_pretraining
        run_graphcl_pretraining(smiles_list, args, device, args.save_dir, PHASE1)
        return

    split = int(len(smiles_list) * (1 - PHASE1["val_split"]))
    train_ds = PretrainDataset(smiles_list[:split], args.mask_prob)
    val_ds = PretrainDataset(smiles_list[split:], args.mask_prob)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_fn,
    )

    # -- Modele --
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,
        edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        conv_type=CONV_TYPE,
        attention_heads=ATTENTION_HEADS,
    )
    mgm_head = MGMHead(hidden_dim=HIDDEN_DIM, num_classes=ATOM_TYPE_VOCAB_SIZE)
    model = MaskedGraphModel(encoder, mgm_head).to(device)

    num_params = sum(pp.numel() for pp in model.parameters())
    print(f"  Modele: {num_params:,} params")

    # -- Optimiseur / Scheduler --
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=PHASE1["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer, PHASE1["warmup_epochs"], args.epochs, PHASE1["lr_min"],
    )

    # -- AMP (mixed precision) : gros gain de vitesse sur GPU P100 --
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  AMP (mixed precision) active")

    # -- Boucle d'entrainement --
    best_val_loss = float("inf")
    no_improve = 0
    history = []
    best_path = Path(args.save_dir) / PHASE1["checkpoint_name"]
    latest_path = Path(args.save_dir) / "checkpoint_latest.pth"

    # Logger temps reel (lu par le tableau de bord pendant l'entrainement)
    live = LiveLogger(Path(args.save_dir) / "live_metrics.jsonl",
                      meta={"phase": "phase1", "objective": args.objective,
                            "epochs_total": args.epochs, "conv_type": CONV_TYPE})

    print(f"\nDemarrage - {args.epochs} epochs, patience={args.patience}")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch - 1)
        lr = scheduler.get_last_lr()[0]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, PHASE1["grad_clip"], epoch, scaler,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Point temps reel (Phase 1 = MGM par classification du type d'atome).
        # train_acc/val_acc = exactitude de prediction du type d'atome masque.
        # Le logging ne doit jamais casser l'entrainement.
        with contextlib.suppress(Exception):
            live.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      "train_acc": train_acc, "val_acc": val_acc,
                      "lr": lr, "best_loss": min(best_val_loss, val_loss)})

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | lr={lr:.2e} | ETA {eta}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": lr,
        })

        # -- Checkpoint --
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": {
                "atom_dim": ATOM_FEATURE_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_GNN_LAYERS,
                "edge_dim": BOND_FEATURE_DIM,
                "output_dim": OUTPUT_DIM,
                "dropout": DROPOUT,
            },
        }
        torch.save(ckpt, latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(ckpt, best_path)
            print(f"  -> Nouveau meilleur modele sauvegarde ({best_path.name})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (patience={args.patience})")
                break

    # -- Sauvegarde finale --
    total_time = time.time() - t0
    print(f"\nTermine en {timedelta(seconds=int(total_time))}")
    print(f"Meilleur val_loss = {best_val_loss:.5f}")

    log_path = log_dir / f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Log -> {log_path}")


if __name__ == "__main__":
    main()
