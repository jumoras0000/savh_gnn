"""
Phase 2 - Fine-tuning classification de toxicite - v2.

Corrections :
  1. LR differencies (encoder vs head).
  2. Warmup + cosine scheduler.
  3. pos_weight dynamique pour desequilibre de classes.
  4. Gradual unfreezing de l'encodeur.
  5. Recherche de seuil optimal par tache.
  6. Early stopping sur ROC-AUC moyen.
"""
import sys
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import MolecularEncoder
from src.models.toxicity_classifier import ToxicityClassifier, MultiTaskBCELoss
from src.preprocessing.toxicity_loader import ToxicityDataset, collate_toxicity_batch
from src.config import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
    HIDDEN_DIM, NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT,
    PHASE2, CHECKPOINT_DIR, LOG_DIR,
)


# ======================================================================
# Scheduler warmup + cosine
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            alpha = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            alpha = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.lr_min + (blr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(logits, targets):
    """ROC-AUC, F1, precision, recall (macro) en ignorant les NaN."""
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()
    valid = ~np.isnan(targets_np)

    aucs, f1s, precs, recs = [], [], [], []
    num_tasks = targets_np.shape[1]

    for t in range(num_tasks):
        mask = valid[:, t]
        if mask.sum() == 0:
            continue
        y = targets_np[mask, t]
        p = probs[mask, t]
        pred = (p > 0.5).astype(int)

        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))
        f1s.append(f1_score(y, pred, zero_division=0))
        precs.append(precision_score(y, pred, zero_division=0))
        recs.append(recall_score(y, pred, zero_division=0))

    return {
        "roc_auc": float(np.mean(aucs)) if aucs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
    }


def find_optimal_thresholds(logits, targets, num_tasks):
    """Cherche le seuil optimal par tache (max F1)."""
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()
    valid = ~np.isnan(targets_np)
    thresholds = []

    for t in range(num_tasks):
        mask = valid[:, t]
        if mask.sum() == 0:
            thresholds.append(0.5)
            continue
        y = targets_np[mask, t]
        p = probs[mask, t]
        best_t, best_f1 = 0.5, 0.0
        for th in np.arange(0.2, 0.8, 0.05):
            f = f1_score(y, (p > th).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = th
        thresholds.append(float(best_t))
    return thresholds


# ======================================================================
# Train / Eval
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch):
    model.train()
    total_loss, n = 0.0, 0
    all_logits, all_targets = [], []

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_data, labels in pbar:
        batch_data = batch_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_data)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach())
        all_targets.append(labels.detach())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_targets = [], []

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels = labels.to(device)
        logits = model(batch_data)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits)
        all_targets.append(labels)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics, all_logits, all_targets


# ======================================================================
# Main
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 2 - Fine-tuning toxicite")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--pretrained_model", type=str, required=True, help="Phase 1 checkpoint")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase2"))
    p.add_argument("--epochs", type=int, default=PHASE2["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE2["batch_size"])
    p.add_argument("--patience", type=int, default=PHASE2["patience"])
    p.add_argument("--smiles_column", type=str, default="smiles")
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase2"
    log_dir.mkdir(parents=True, exist_ok=True)

    # -- Datasets --
    print("Chargement des datasets ...")
    train_ds = ToxicityDataset(args.train_csv, smiles_column=args.smiles_column)
    val_ds = ToxicityDataset(args.val_csv, smiles_column=args.smiles_column,
                             task_columns=train_ds.get_task_names())
    num_tasks = train_ds.get_num_tasks()
    print(f"  {num_tasks} taches, {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_toxicity_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_toxicity_batch,
    )

    # -- Modele --
    print("Construction du modele ...")
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
    )

    # Charger les poids pre-entraines
    ckpt = torch.load(args.pretrained_model, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        if encoder_sd:
            encoder.load_state_dict(encoder_sd, strict=False)
            print("  Encodeur pre-entraine charge")
        else:
            print("  WARN: pas de poids encoder dans le checkpoint")
    else:
        print("  WARN: format checkpoint non reconnu, poids aleatoires")

    model = ToxicityClassifier(
        encoder=encoder, num_tasks=num_tasks,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        freeze_encoder=True,
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    n_train = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  {n_params:,} params total, {n_train:,} entrainables")

    # -- Loss avec pos_weight --
    pos_weight = train_ds.get_pos_weight().to(device)
    criterion = MultiTaskBCELoss(pos_weight=pos_weight)

    # -- Optimiseur (LR differencies) --
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": PHASE2["lr_encoder"]},
        {"params": model.classifier.parameters(), "lr": PHASE2["lr_head"]},
    ], weight_decay=PHASE2["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer, PHASE2["warmup_epochs"], args.epochs, PHASE2["lr_min"],
    )

    # -- Boucle --
    best_auc = 0.0
    no_improve = 0
    history = []
    best_path = Path(args.save_dir) / "best_toxicity_model.pth"
    latest_path = Path(args.save_dir) / "checkpoint_latest.pth"
    freeze_epochs = PHASE2["freeze_encoder_epochs"]

    print(f"\nDemarrage - {args.epochs} epochs, freeze_encoder={freeze_epochs} epochs, patience={args.patience}")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Gradual unfreezing
        if epoch <= freeze_epochs:
            model._freeze_encoder()
        else:
            model.gradual_unfreeze(epoch, freeze_epochs)

        scheduler.step(epoch - 1)
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PHASE2["grad_clip"], epoch,
        )
        val_m, val_logits, val_targets = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"t_loss={train_m['loss']:.4f} v_loss={val_m['loss']:.4f} | "
            f"t_auc={train_m['roc_auc']:.4f} v_auc={val_m['roc_auc']:.4f} | "
            f"v_f1={val_m['f1']:.4f} | lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} | ETA {eta}"
        )

        history.append({
            "epoch": epoch, "train": train_m, "val": val_m,
            "lr_encoder": lr_enc, "lr_head": lr_head,
        })

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_m,
            "best_auc": best_auc,
            "history": history,
            "num_tasks": num_tasks,
            "task_names": train_ds.get_task_names(),
        }
        torch.save(ckpt_data, latest_path)

        if val_m["roc_auc"] > best_auc:
            best_auc = val_m["roc_auc"]
            no_improve = 0
            torch.save(ckpt_data, best_path)
            print(f"  -> Nouveau meilleur AUC={best_auc:.4f} sauvegarde")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (patience={args.patience})")
                break

    # -- Seuils optimaux --
    print("\nRecherche des seuils optimaux ...")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_logits, final_targets = evaluate(model, val_loader, criterion, device)
    opt_thresh = find_optimal_thresholds(final_logits, final_targets, num_tasks)

    for i, (name, th) in enumerate(zip(train_ds.get_task_names(), opt_thresh)):
        print(f"  {name}: threshold={th:.2f}")

    # Sauvegarder les seuils dans le checkpoint
    best_ckpt["optimal_thresholds"] = opt_thresh
    torch.save(best_ckpt, best_path)

    # -- Resume --
    total_time = time.time() - t0
    print(f"\nTermine en {timedelta(seconds=int(total_time))}")
    print(f"Meilleur ROC-AUC = {best_auc:.4f}")

    log_path = log_dir / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Log -> {log_path}")


if __name__ == "__main__":
    main()