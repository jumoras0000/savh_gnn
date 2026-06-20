"""
trainer.py – Entraîneur unifié (Phase 1, 2, 3)
================================================
Gère les 3 phases d'entraînement :

  Phase 1 : Pré-entraînement auto-supervisé (Masked Graph Modeling)
            → Apprend des représentations moléculaires riches sans labels

  Phase 2 : Fine-tuning classification de toxicité (Tox21, multi-tâche)
            → Transfer learning avec dégel progressif de l'encodeur

  Phase 3 : Entraînement multi-propriétés + module Raisonneur
            → Prédit simultanément 17 propriétés moléculaires

Mécanismes anti-surapprentissage intégrés :
  ✓ Early stopping avec restauration des meilleurs poids
  ✓ Warmup linéaire + cosine annealing (scheduler)
  ✓ Gradient clipping (norme max = 1.0)
  ✓ Weight decay L2 (AdamW)
  ✓ Dropout (couche GNN + têtes)
  ✓ Dégel progressif de l'encodeur (gradual unfreezing)
  ✓ Monitoring train/val gap automatique
  ✓ Mixed Precision (AMP) sur GPU CUDA
"""
import os
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_DIR, LOG_DIR, PHASE1, PHASE2, PHASE3
from gnn_models import build_encoder, WarmupCosineScheduler, count_parameters
from prediction_heads import (
    MGMHead, MaskedGraphModel,
    ToxicityClassifier, MultiPropertyPredictor, MolecularReasoner,
)
from losses import MultiTaskBCELoss, MultiPropertyLoss, ReasonerLoss
from metrics import (
    compute_classification_metrics, compute_regression_metrics,
    EarlyStopping, TrainingHistory,
)

logger = logging.getLogger("panacee.trainer")


# ══════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def setup_logger(phase_name: str, log_dir: str = LOG_DIR) -> logging.Logger:
    """Configure le logger avec sortie console + fichier."""
    os.makedirs(log_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger(f"panacee.{phase_name}")
    log.setLevel(logging.INFO)

    if not log.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s | %(message)s",
                                datefmt="%H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        log.addHandler(ch)

        fh = logging.FileHandler(
            os.path.join(log_dir, f"{phase_name}_{ts}.log"),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
    config_dict: Optional[dict] = None,
):
    """Sauvegarde complète : poids + optimiseur + métadonnées."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics"    : metrics,
        "config"     : config_dict or {},
        "timestamp"  : datetime.now().isoformat(),
    }, path)


def load_encoder_weights(encoder: nn.Module, checkpoint_path: str) -> nn.Module:
    """Charge uniquement les poids de l'encodeur depuis un checkpoint Phase 1."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Chercher le state_dict de l'encodeur dans diverses structures
    sd = ckpt.get("model_state", ckpt)
    # Filtrer seulement les clés de l'encodeur
    encoder_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            encoder_sd[k[len("encoder."):]] = v
        elif not any(k.startswith(p) for p in ["mgm_head.", "classifier.", "shared.", "toxicity_head.", "efficacy_head."]):
            encoder_sd[k] = v

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing:
        logger.warning(f"  Clés manquantes dans l'encodeur: {missing[:5]}")
    if unexpected:
        logger.warning(f"  Clés inattendues ignorées: {unexpected[:5]}")

    logger.info(f"  Encodeur chargé depuis {checkpoint_path}")
    return encoder


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 – PRÉ-ENTRAÎNEMENT MGM
# ══════════════════════════════════════════════════════════════════════

def train_phase1(
    train_loader,
    val_loader,
    arch: str           = "attfp",
    epochs: int         = None,
    batch_size: int     = None,
    lr: float           = None,
    mask_prob: float    = None,
    patience: int       = None,
    save_dir: str       = None,
    device              = None,
    config              = None,
) -> str:
    """
    Phase 1 : pré-entraînement Masked Graph Modeling.

    Returns:
        Chemin du meilleur checkpoint sauvegardé.
    """
    cfg       = config or PHASE1
    epochs    = epochs    or cfg["epochs"]
    lr        = lr        or cfg["lr"]
    patience  = patience  or cfg["patience"]
    save_dir  = save_dir  or os.path.join(CHECKPOINT_DIR, "phase1")
    device    = device    or DEVICE
    log       = setup_logger("phase1")

    os.makedirs(save_dir, exist_ok=True)

    # ── Modèle ────────────────────────────────────────────────────────
    from config import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT
    encoder  = build_encoder(arch=arch).to(device)
    mgm_head = MGMHead(hidden_dim=HIDDEN_DIM, atom_dim=ATOM_FEATURE_DIM).to(device)
    model    = MaskedGraphModel(encoder, mgm_head).to(device)

    log.info(f"Architecture  : {arch.upper()}")
    log.info(f"Paramètres    : {count_parameters(model):,}")
    log.info(f"Device        : {device}")

    # ── Optimisation ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = WarmupCosineScheduler(
        optimizer, cfg["warmup_epochs"], epochs, cfg["lr_min"]
    )
    scaler    = GradScaler() if "cuda" in str(device) else None

    best_path = os.path.join(save_dir, cfg["checkpoint_name"])
    early_stop = EarlyStopping(
        patience=patience, mode="min",
        checkpoint_path=best_path,
    )
    history   = TrainingHistory()
    criterion = nn.MSELoss()

    log.info("=" * 60)
    log.info("DÉMARRAGE PHASE 1 – Masked Graph Modeling")
    log.info("=" * 60)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss, n_atoms = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[P1 Train] E{epoch}/{epochs}", leave=False)

        for batch_g, masked_idx, masked_feat in pbar:
            batch_g    = batch_g.to(device)
            masked_feat = masked_feat.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast():
                    preds = model(batch_g.x, batch_g.edge_index, batch_g.edge_attr,
                                  batch_g.batch, masked_idx)
                    loss  = criterion(preds, masked_feat)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(batch_g.x, batch_g.edge_index, batch_g.edge_attr,
                              batch_g.batch, masked_idx)
                loss  = criterion(preds, masked_feat)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            train_loss += loss.item() * masked_feat.size(0)
            n_atoms    += masked_feat.size(0)
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for batch_g, masked_idx, masked_feat in val_loader:
                batch_g    = batch_g.to(device)
                masked_feat = masked_feat.to(device)
                preds = model(batch_g.x, batch_g.edge_index, batch_g.edge_attr,
                              batch_g.batch, masked_idx)
                val_loss += criterion(preds, masked_feat).item() * masked_feat.size(0)
                n_val    += masked_feat.size(0)

        train_l = train_loss / max(n_atoms, 1)
        val_l   = val_loss   / max(n_val, 1)
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]

        history.record(epoch, train_l, val_l, lr=current_lr)
        elapsed = str(timedelta(seconds=int(time.time() - t_start)))
        log.info(
            f"E{epoch:3d}/{epochs} | Train={train_l:.5f} | Val={val_l:.5f} | "
            f"LR={current_lr:.2e} | ⏱ {elapsed}"
        )

        # Anti-surapprentissage
        of = history.check_overfitting()
        if of["severity"] == "severe":
            log.warning(f"  ⚠ SURAPPRENTISSAGE SÉVÈRE détecté (gap={of['auc_gap']})")

        if early_stop.step(val_l, model, epoch):
            log.info(f"  ↳ Early stopping à epoch {epoch}")
            break

    early_stop.load_best(model)
    history.save(os.path.join(save_dir, "history_phase1.json"))
    history.plot(save_path=os.path.join(save_dir, "curves_phase1.png"))
    log.info(f"Phase 1 terminée. Meilleur modèle : {best_path}")
    return best_path


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 – FINE-TUNING TOXICITÉ
# ══════════════════════════════════════════════════════════════════════

def train_phase2(
    train_loader,
    val_loader,
    pretrained_encoder_path: str,
    arch: str           = "attfp",
    epochs: int         = None,
    patience: int       = None,
    save_dir: str       = None,
    device              = None,
    config              = None,
    num_tasks: int      = 12,
    pos_weight          = None,
    label_smoothing: float = 0.05,
) -> str:
    """
    Phase 2 : fine-tuning classification de toxicité.

    Returns:
        Chemin du meilleur checkpoint sauvegardé.
    """
    cfg      = config or PHASE2
    epochs   = epochs   or cfg["epochs"]
    patience = patience or cfg["patience"]
    save_dir = save_dir or os.path.join(CHECKPOINT_DIR, "phase2")
    device   = device   or DEVICE
    log      = setup_logger("phase2")

    os.makedirs(save_dir, exist_ok=True)

    # ── Modèle ────────────────────────────────────────────────────────
    from config import HIDDEN_DIM, DROPOUT
    encoder  = build_encoder(arch=arch).to(device)
    encoder  = load_encoder_weights(encoder, pretrained_encoder_path)
    model    = ToxicityClassifier(
        encoder, num_tasks=num_tasks,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        freeze_encoder=True,
    ).to(device)

    log.info(f"Encodeur pré-entraîné : {pretrained_encoder_path}")
    log.info(f"Paramètres totaux     : {count_parameters(model):,}")

    # ── Optimisation ──────────────────────────────────────────────────
    optimizer = AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg["lr_encoder"]},
        {"params": model.classifier.parameters(), "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer, cfg["warmup_epochs"], epochs, cfg["lr_min"]
    )
    scaler    = GradScaler() if "cuda" in str(device) else None
    criterion = MultiTaskBCELoss(
        pos_weight=pos_weight, label_smoothing=label_smoothing
    )

    best_path  = os.path.join(save_dir, cfg["checkpoint_name"])
    early_stop = EarlyStopping(
        patience=patience, mode="max",  # on maximise l'AUC
        checkpoint_path=best_path,
    )
    history = TrainingHistory()

    log.info("=" * 60)
    log.info("DÉMARRAGE PHASE 2 – Fine-tuning Toxicité")
    log.info("=" * 60)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # Dégel progressif de l'encodeur
        model.gradual_unfreeze(epoch, cfg["freeze_encoder_epochs"])

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        all_logits, all_targets, t_loss, n = [], [], 0.0, 0
        pbar = tqdm(train_loader, desc=f"[P2 Train] E{epoch}/{epochs}", leave=False)

        for batch_data, labels in pbar:
            batch_data = batch_data.to(device)
            labels     = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast():
                    logits = model(batch_data)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_data)
                loss   = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            t_loss += loss.item() * labels.size(0)
            n      += labels.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(labels.detach().cpu())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        all_logits  = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        train_m = compute_classification_metrics(all_logits, all_targets, t_loss / max(n, 1))

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_logits, val_targets, v_loss, v_n = [], [], 0.0, 0
        with torch.no_grad():
            for batch_data, labels in val_loader:
                batch_data = batch_data.to(device)
                labels     = labels.to(device)
                logits     = model(batch_data)
                loss       = criterion(logits, labels)
                v_loss += loss.item() * labels.size(0)
                v_n    += labels.size(0)
                val_logits.append(logits.cpu())
                val_targets.append(labels.cpu())

        val_logits  = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)
        val_m = compute_classification_metrics(val_logits, val_targets, v_loss / max(v_n, 1))

        scheduler.step(epoch)
        lr_now = scheduler.get_last_lr()[0]
        history.record(epoch, train_m.loss, val_m.loss, train_m.roc_auc, val_m.roc_auc, lr_now)

        elapsed = str(timedelta(seconds=int(time.time() - t_start)))
        log.info(
            f"E{epoch:3d}/{epochs} | "
            f"Train: {train_m.summary()} | "
            f"Val: {val_m.summary()} | "
            f"LR={lr_now:.2e} | ⏱ {elapsed}"
        )

        # Monitoring surapprentissage
        of = history.check_overfitting()
        if of["severity"] != "none":
            log.warning(
                f"  ⚠ Surapprentissage [{of['severity']}] "
                f"gap AUC={of['auc_gap']:.4f}"
            )

        if early_stop.step(val_m.roc_auc, model, epoch):
            log.info(f"  ↳ Early stopping à epoch {epoch}")
            break

    early_stop.load_best(model)
    history.save(os.path.join(save_dir, "history_phase2.json"))
    history.plot(save_path=os.path.join(save_dir, "curves_phase2.png"))
    log.info(f"Phase 2 terminée. Meilleur modèle : {best_path}")
    return best_path


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3 – MULTI-PROPRIÉTÉS
# ══════════════════════════════════════════════════════════════════════

def train_phase3(
    train_loader,
    val_loader,
    pretrained_tox_path: str,
    arch: str           = "attfp",
    epochs: int         = None,
    patience: int       = None,
    save_dir: str       = None,
    device              = None,
    config              = None,
) -> str:
    """
    Phase 3 : entraînement multi-propriétés + raisonneur.

    Returns:
        Chemin du meilleur checkpoint sauvegardé.
    """
    cfg      = config or PHASE3
    epochs   = epochs   or cfg["epochs"]
    patience = patience or cfg["patience"]
    save_dir = save_dir or os.path.join(CHECKPOINT_DIR, "phase3")
    device   = device   or DEVICE
    log      = setup_logger("phase3")

    os.makedirs(save_dir, exist_ok=True)

    # ── Modèle ────────────────────────────────────────────────────────
    from config import HIDDEN_DIM, DROPOUT
    encoder = build_encoder(arch=arch).to(device)

    # Charger depuis Phase 2 si disponible
    if os.path.exists(pretrained_tox_path):
        encoder = load_encoder_weights(encoder, pretrained_tox_path)
        log.info(f"Encodeur Phase 2 chargé : {pretrained_tox_path}")

    model = MultiPropertyPredictor(
        encoder, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
        freeze_encoder=True,
    ).to(device)

    reasoner = MolecularReasoner(
        mol_emb_dim=HIDDEN_DIM,
        d_model=cfg["reasoner_hidden_dim"],
        num_heads=cfg["reasoner_num_heads"],
        num_layers=cfg["reasoner_num_layers"],
        dropout=cfg["reasoner_dropout"],
    ).to(device)

    log.info(f"Paramètres MultiProp : {count_parameters(model):,}")
    log.info(f"Paramètres Reasoner  : {count_parameters(reasoner):,}")

    # ── Optimisation ──────────────────────────────────────────────────
    optimizer = AdamW([
        {"params": model.encoder.parameters(),  "lr": cfg["lr_encoder"]},
        {"params": [p for n, p in model.named_parameters() if "encoder" not in n],
         "lr": cfg["lr_heads"]},
        {"params": reasoner.parameters(),       "lr": cfg.get("lr_heads", 5e-4)},
    ], weight_decay=cfg["weight_decay"])

    scheduler  = WarmupCosineScheduler(
        optimizer, cfg["warmup_epochs"], epochs, cfg["lr_min"]
    )
    scaler     = GradScaler() if "cuda" in str(device) else None
    criterion  = MultiPropertyLoss()

    best_path  = os.path.join(save_dir, cfg["checkpoint_name"])
    early_stop = EarlyStopping(
        patience=patience, mode="min",
        checkpoint_path=best_path,
    )
    history = TrainingHistory()

    log.info("=" * 60)
    log.info("DÉMARRAGE PHASE 3 – Multi-Propriétés")
    log.info("=" * 60)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.gradual_unfreeze(epoch, cfg["freeze_encoder_epochs"])

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        reasoner.train()
        t_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"[P3 Train] E{epoch}/{epochs}", leave=False)

        for batch_data, label_dict in pbar:
            batch_data = batch_data.to(device)
            tgt_dict   = {k: v.to(device) for k, v in label_dict.items()}
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with autocast():
                    preds   = model(batch_data)
                    loss, _ = criterion(preds, tgt_dict)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reasoner.parameters()),
                    cfg["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                preds   = model(batch_data)
                loss, _ = criterion(preds, tgt_dict)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(reasoner.parameters()),
                    cfg["grad_clip"]
                )
                optimizer.step()

            t_loss += loss.item() * batch_data.num_graphs
            n      += batch_data.num_graphs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        reasoner.eval()
        v_loss, v_n = 0.0, 0

        with torch.no_grad():
            for batch_data, label_dict in val_loader:
                batch_data = batch_data.to(device)
                tgt_dict   = {k: v.to(device) for k, v in label_dict.items()}
                preds      = model(batch_data)
                loss, _    = criterion(preds, tgt_dict)
                v_loss += loss.item() * batch_data.num_graphs
                v_n    += batch_data.num_graphs

        train_l = t_loss / max(n, 1)
        val_l   = v_loss / max(v_n, 1)

        scheduler.step(epoch)
        lr_now = scheduler.get_last_lr()[0]
        history.record(epoch, train_l, val_l, lr=lr_now)

        elapsed = str(timedelta(seconds=int(time.time() - t_start)))
        log.info(
            f"E{epoch:3d}/{epochs} | "
            f"Train={train_l:.5f} | Val={val_l:.5f} | "
            f"LR={lr_now:.2e} | ⏱ {elapsed}"
        )

        of = history.check_overfitting()
        if of["severity"] != "none":
            log.warning(
                f"  ⚠ Surapprentissage [{of['severity']}] "
                f"divergence={of['loss_divergence']:.4f}"
            )

        if early_stop.step(val_l, model, epoch):
            log.info(f"  ↳ Early stopping à epoch {epoch}")
            break

    early_stop.load_best(model)

    # Sauvegarder modèle complet (encodeur + têtes + raisonneur)
    final_path = os.path.join(save_dir, cfg["checkpoint_name"])
    torch.save({
        "model_state"   : model.state_dict(),
        "reasoner_state": reasoner.state_dict(),
        "config"        : cfg,
        "epoch"         : early_stop.best_epoch,
        "best_val_loss" : early_stop.best_score,
        "timestamp"     : datetime.now().isoformat(),
    }, final_path)

    history.save(os.path.join(save_dir, "history_phase3.json"))
    history.plot(save_path=os.path.join(save_dir, "curves_phase3.png"))
    log.info(f"Phase 3 terminée. Modèle complet : {final_path}")
    return final_path
