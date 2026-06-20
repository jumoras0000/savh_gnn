"""
Phase 2 - Fine-tuning classification de toxicite - v3.

Nouveautes v3 :
  - Encodeur a attention edge-aware (config CONV_TYPE).
  - AMP (mixed precision) sur GPU.
  - EMA des poids (--ema) pour un fine-tuning plus stable.
  - Cross-validation par scaffold (--cv_folds N) pour des metriques honnetes.

Conserve v2 :
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
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import MolecularEncoder
from src.models.toxicity_classifier import ToxicityClassifier, MultiTaskBCELoss
from src.preprocessing.toxicity_loader import ToxicityDataset, collate_toxicity_batch
from src.preprocessing.scaffold_split import scaffold_kfold
from src.utils.ema import ModelEMA
from src.config import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
    HIDDEN_DIM, NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT,
    CONV_TYPE, ATTENTION_HEADS,
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

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch,
                    scaler=None, ema=None):
    model.train()
    total_loss, n = 0.0, 0
    all_logits, all_targets = [], []
    use_amp = scaler is not None and scaler.is_enabled()

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_data, labels in pbar:
        batch_data = batch_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(batch_data)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_data)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach().float())
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
        all_logits.append(logits.float())
        all_targets.append(labels)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics, all_logits, all_targets


# ======================================================================
# Helpers
# ======================================================================

class _GraphLabelDataset(Dataset):
    """Dataset minimal (graphes + labels) pour la cross-validation."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def _pos_weight_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """neg/pos par tache (pour BCEWithLogitsLoss), en ignorant les NaN."""
    pw = []
    for t in range(labels.shape[1]):
        col = labels[:, t]
        valid = col[~torch.isnan(col)]
        pos = (valid == 1).sum().float()
        neg = (valid == 0).sum().float()
        pw.append((neg / pos).item() if pos > 0 else 1.0)
    return torch.tensor(pw, dtype=torch.float32)


def _load_pretrained_encoder(encoder, path):
    """Charge les poids d'encodeur Phase 1 si dispo (sinon aleatoire)."""
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            encoder_sd = {k.replace("encoder.", ""): v
                          for k, v in sd.items() if k.startswith("encoder.")}
            if encoder_sd:
                encoder.load_state_dict(encoder_sd, strict=False)
                print("  Encodeur pre-entraine charge (Phase 1)")
                return
            print("  WARN: pas de poids encoder dans le checkpoint, poids aleatoires")
        else:
            print("  WARN: format checkpoint non reconnu, poids aleatoires")
    else:
        print("  Pas de checkpoint Phase 1 -> encodeur aleatoire (Phase 2 autonome)")


# ======================================================================
# Une execution complete (un split) : build -> train -> best
# ======================================================================

def train_one_run(train_ds, val_ds, num_tasks, task_names, pos_weight,
                  args, device, save_dir, use_ema=True, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    best_path = Path(save_dir) / "best_toxicity_model.pth"
    latest_path = Path(save_dir) / "checkpoint_latest.pth"
    label = f"[{tag}] " if tag else ""

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_toxicity_batch, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_toxicity_batch,
    )

    # -- Modele --
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )
    _load_pretrained_encoder(encoder, args.pretrained_model)

    model = ToxicityClassifier(
        encoder=encoder, num_tasks=num_tasks,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT, freeze_encoder=True,
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"{label}Modele: {n_params:,} params | conv={CONV_TYPE}")

    criterion = MultiTaskBCELoss(pos_weight=pos_weight.to(device))
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": PHASE2["lr_encoder"]},
        {"params": model.classifier.parameters(), "lr": PHASE2["lr_head"]},
    ], weight_decay=PHASE2["weight_decay"])
    scheduler = WarmupCosineScheduler(
        optimizer, PHASE2["warmup_epochs"], args.epochs, PHASE2["lr_min"],
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=0.999) if use_ema else None
    print(f"{label}AMP={use_amp} | EMA={use_ema} | {args.epochs} epochs | patience={args.patience}")

    best_auc = 0.0
    no_improve = 0
    history = []
    freeze_epochs = PHASE2["freeze_encoder_epochs"]
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch <= freeze_epochs:
            model._freeze_encoder()
        else:
            model.gradual_unfreeze(epoch, freeze_epochs)

        scheduler.step(epoch - 1)
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PHASE2["grad_clip"], epoch, scaler, ema,
        )

        # Evaluation avec les poids EMA (si actif)
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        val_m, _, _ = evaluate(model, val_loader, criterion, device)
        if ema is not None:
            ema.restore(model)

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))
        print(
            f"{label}Epoch {epoch:3d}/{args.epochs} | "
            f"t_loss={train_m['loss']:.4f} v_loss={val_m['loss']:.4f} | "
            f"t_auc={train_m['roc_auc']:.4f} v_auc={val_m['roc_auc']:.4f} | "
            f"v_f1={val_m['f1']:.4f} | lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} | ETA {eta}"
        )
        history.append({"epoch": epoch, "train": train_m, "val": val_m,
                        "lr_encoder": lr_enc, "lr_head": lr_head})

        # state_dict a sauvegarder = EMA si actif, sinon poids courants
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
            state_to_save = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model)
        else:
            state_to_save = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": state_to_save,
            "val_metrics": val_m,
            "best_auc": best_auc,
            "history": history,
            "num_tasks": num_tasks,
            "task_names": task_names,
            "config": {
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_GNN_LAYERS,
                "output_dim": OUTPUT_DIM, "conv_type": CONV_TYPE,
                "attention_heads": ATTENTION_HEADS, "ema": use_ema,
            },
        }
        torch.save(ckpt_data, latest_path)

        if val_m["roc_auc"] > best_auc:
            best_auc = val_m["roc_auc"]
            no_improve = 0
            torch.save(ckpt_data, best_path)
            print(f"{label}  -> Nouveau meilleur AUC={best_auc:.4f} sauvegarde")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"{label}Early stopping (patience={args.patience})")
                break

    # -- Seuils optimaux (sur le meilleur modele) --
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    _, final_logits, final_targets = evaluate(model, val_loader, criterion, device)
    opt_thresh = find_optimal_thresholds(final_logits, final_targets, num_tasks)
    best_ckpt["optimal_thresholds"] = opt_thresh
    torch.save(best_ckpt, best_path)

    total_time = time.time() - t0
    print(f"{label}Termine en {timedelta(seconds=int(total_time))} | meilleur ROC-AUC={best_auc:.4f}")
    return best_auc, history


# ======================================================================
# Main
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 2 - Fine-tuning toxicite")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--pretrained_model", type=str, default=None,
                   help="Phase 1 checkpoint (optionnel : sinon encodeur aleatoire)")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase2"))
    p.add_argument("--epochs", type=int, default=PHASE2["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE2["batch_size"])
    p.add_argument("--patience", type=int, default=PHASE2["patience"])
    p.add_argument("--smiles_column", type=str, default="smiles")
    p.add_argument("--max_molecules", type=int, default=None,
                   help="Limite de molecules (runs rapides Kaggle)")
    p.add_argument("--cv_folds", type=int, default=0,
                   help="Cross-validation par scaffold (0 = split simple)")
    p.add_argument("--ema", type=int, default=1, help="1=EMA des poids actif, 0=desactive")
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase2"
    log_dir.mkdir(parents=True, exist_ok=True)
    use_ema = bool(args.ema)

    # -- Datasets --
    print("Chargement des datasets ...")
    train_ds = ToxicityDataset(args.train_csv, smiles_column=args.smiles_column,
                               max_molecules=args.max_molecules)
    val_ds = ToxicityDataset(args.val_csv, smiles_column=args.smiles_column,
                             task_columns=train_ds.get_task_names(),
                             max_molecules=args.max_molecules)
    num_tasks = train_ds.get_num_tasks()
    task_names = train_ds.get_task_names()
    print(f"  {num_tasks} taches, {len(train_ds)} train, {len(val_ds)} val")

    # ── Cross-validation par scaffold ────────────────────────────────
    if args.cv_folds and args.cv_folds >= 2:
        print(f"\n=== Cross-validation scaffold {args.cv_folds}-fold ===")
        graphs = list(train_ds.graphs) + list(val_ds.graphs)
        labels = torch.cat([train_ds.labels, val_ds.labels], dim=0)
        smiles = [getattr(g, "smiles", "") for g in graphs]
        combined = _GraphLabelDataset(graphs, labels)

        fold_aucs = []
        best_fold, best_fold_auc, best_fold_dir = -1, -1.0, None
        for fold, (tr_idx, va_idx) in enumerate(scaffold_kfold(smiles, k=args.cv_folds), start=1):
            fold_dir = Path(args.save_dir) / f"fold{fold}"
            pw = _pos_weight_from_labels(labels[tr_idx])
            auc, _ = train_one_run(
                Subset(combined, tr_idx), Subset(combined, va_idx),
                num_tasks, task_names, pw, args, device, str(fold_dir),
                use_ema=use_ema, tag=f"fold{fold}/{args.cv_folds}",
            )
            fold_aucs.append(auc)
            if auc > best_fold_auc:
                best_fold_auc, best_fold, best_fold_dir = auc, fold, fold_dir

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        print(f"\n=== Resultat CV : ROC-AUC = {mean_auc:.4f} +/- {std_auc:.4f} "
              f"(folds: {[round(a, 4) for a in fold_aucs]}) ===")

        # Copier le meilleur fold comme modele final
        if best_fold_dir is not None:
            src = best_fold_dir / "best_toxicity_model.pth"
            dst = Path(args.save_dir) / "best_toxicity_model.pth"
            if src.exists():
                torch.save(torch.load(src, map_location="cpu", weights_only=False), dst)
                print(f"Meilleur fold = {best_fold} (AUC={best_fold_auc:.4f}) -> {dst}")

        summary = {"cv_folds": args.cv_folds, "fold_aucs": fold_aucs,
                   "mean_auc": mean_auc, "std_auc": std_auc, "best_fold": best_fold}
        log_path = log_dir / f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Log CV -> {log_path}")
        return

    # ── Split simple (defaut) ────────────────────────────────────────
    pos_weight = train_ds.get_pos_weight()
    best_auc, history = train_one_run(
        train_ds, val_ds, num_tasks, task_names, pos_weight,
        args, device, args.save_dir, use_ema=use_ema,
    )

    log_path = log_dir / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Meilleur ROC-AUC = {best_auc:.4f} | Log -> {log_path}")


if __name__ == "__main__":
    main()
