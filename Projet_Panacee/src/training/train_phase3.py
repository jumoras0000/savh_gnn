"""
Phase 3 - Entraînement multi-propriétés + IA raisonnement.

Pipeline complet :
  1. Charger l'encodeur pré-entraîné Phase 2 (best_toxicity_model.pth)
  2. Construire le MultiPropertyPredictor (N têtes)
  3. Entraîner sur les datasets fusionnés (Tox21, ESOL, Lipo, BBBP, ClinTox, HIV)
  4. Entraîner le MolecularReasoner pour l'analyse combinatoire
  5. Sauvegarder le modèle complet : panacee_phase3_complete.pth
"""
import contextlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    EXTERNAL_DIR,
    HIDDEN_DIM,
    LOG_DIR,
    NUM_GNN_LAYERS,
    OUTPUT_DIM,
    PHASE3,
    loader_kwargs,
    lr_scale_for_batch,
)
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyLoss, MultiPropertyPredictor
from src.models.reasoner import MolecularReasoner, ReasonerLoss
from src.preprocessing.multi_property_loader import (
    MultiPropertyDataset,
    collate_multi_property,
    download_all_phase3_data,
    merge_datasets,
)
from src.utils.error_handler import (
    HealthMonitor,
    emergency_save,
    setup_logging,
)
from src.utils.gpu_manager import get_gpu_manager
from src.utils.live_logger import LiveLogger
from src.validation.clinical_metrics import clinical_score as clinical_score_fn
from src.validation.clinical_metrics import summarize as clinical_summarize

# ======================================================================
# Scheduler warmup + cosine
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-7):
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
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = self.lr_min + (blr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Métriques
# ======================================================================

def compute_phase3_metrics(predictions, targets):
    """
    Calcule les métriques pour toutes les propriétés.

    Returns:
        dict {property: {metric: value}}
    """
    metrics = {}

    for prop_name in predictions:
        if prop_name not in targets:
            continue

        pred = predictions[prop_name].detach().cpu()
        tgt = targets[prop_name].detach().cpu().numpy()
        valid = ~np.isnan(tgt)

        if valid.sum() == 0:
            continue

        prop_metrics = {}

        if prop_name in ("toxicity", "bioavailability", "metabolic_stability"):
            # Classification : AUC + F1
            probs = torch.sigmoid(pred).numpy()
            aucs = []
            f1s = []
            for t in range(tgt.shape[1]):
                mask = valid[:, t]
                if mask.sum() < 2:
                    continue
                y = tgt[mask, t]
                p = probs[mask, t]
                # AUC et F1 seulement si les DEUX classes sont présentes :
                # un 0 fictif (tâche mono-classe) fausserait la macro-moyenne.
                if len(np.unique(y)) > 1:
                    aucs.append(roc_auc_score(y, p))
                    f1s.append(f1_score(y, (p > 0.5).astype(int), zero_division=0))

            prop_metrics["roc_auc"] = float(np.mean(aucs)) if aucs else 0.0
            prop_metrics["f1"] = float(np.mean(f1s)) if f1s else 0.0

        elif prop_name in ("efficacy",):
            # Classification binaire
            probs = torch.sigmoid(pred).numpy()
            y = tgt[valid].flatten()
            p = probs[valid[:, 0] if valid.ndim > 1 else valid].flatten()
            if len(y) > 0 and len(np.unique(y)) > 1:
                prop_metrics["roc_auc"] = float(roc_auc_score(y, p))
                prop_metrics["f1"] = float(f1_score(y, (p > 0.5).astype(int), zero_division=0))
            else:
                prop_metrics["roc_auc"] = 0.0
                prop_metrics["f1"] = 0.0

        else:
            # Régression : RMSE + R²
            pred_np = pred.numpy()
            pred_valid = pred_np[valid]
            tgt_valid = tgt[valid]
            rmse = float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
            ss_res = float(np.sum((tgt_valid - pred_valid) ** 2))
            ss_tot = float(np.sum((tgt_valid - tgt_valid.mean()) ** 2))
            # R² indéfini si la cible est ~constante (ss_tot≈0) : on renvoie 0.0
            # (neutre) plutôt que ss_res/1e-8 qui produit un R² délirant (-1e8…).
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
            prop_metrics["rmse"] = rmse
            prop_metrics["r2"] = r2

        metrics[prop_name] = prop_metrics

    return metrics


# ======================================================================
# Boucles train / eval
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bioavailability", "metabolic_stability"]}
    all_targets = {k: [] for k in all_preds}

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_data, labels in pbar:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        predictions = model(batch_data)

        loss, _ = criterion(predictions, labels_device)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs
        n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach())
                all_targets[k].append(labels_device[k].detach())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Agréger les prédictions
    merged_preds = {}
    merged_targets = {}
    for k in all_preds:
        if all_preds[k]:
            merged_preds[k] = torch.cat(all_preds[k])
            merged_targets[k] = torch.cat(all_targets[k])

    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    metrics["_loss"] = total_loss / max(n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bioavailability", "metabolic_stability"]}
    all_targets = {k: [] for k in all_preds}

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        predictions = model(batch_data)
        loss, _ = criterion(predictions, labels_device)

        if not torch.isnan(loss):
            total_loss += loss.item() * batch_data.num_graphs
            n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach())
                all_targets[k].append(labels_device[k].detach())

    merged_preds = {}
    merged_targets = {}
    for k in all_preds:
        if all_preds[k]:
            merged_preds[k] = torch.cat(all_preds[k])
            merged_targets[k] = torch.cat(all_targets[k])

    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    metrics["_loss"] = total_loss / max(n, 1)
    return metrics, merged_preds, merged_targets


# ======================================================================
# Fonction principale
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 3 - Multi-propriétés + IA Raisonnement")
    p.add_argument("--download", action="store_true", help="Télécharge tous les datasets")
    p.add_argument("--data_dir", type=str, default=None, help="Dossier des données Phase 3")
    p.add_argument("--pretrained_model", type=str, default=None,
                   help="Checkpoint Phase 2 (best_toxicity_model.pth)")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase3"))
    p.add_argument("--epochs", type=int, default=PHASE3["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE3["batch_size"])
    p.add_argument("--patience", type=int, default=PHASE3["patience"])
    p.add_argument("--warm_start", type=int, default=1,
                   help="1=reprend le meilleur modele Phase 3 precedent (transfert)")
    p.add_argument("--save_epochs", type=int, default=1,
                   help="1=sauvegarde un checkpoint par epoch (gerable depuis le dashboard)")
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    # Initialisation GPU et logging
    gpu = get_gpu_manager(force_cpu=(args.device == "cpu"))
    device = gpu.device
    setup_logging(name="panacee.phase3")
    health = HealthMonitor(check_interval=50)

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase3"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Étape 0 : Vérifications ──────────────────────────────────────
    print("=" * 80)
    print("  PHASE 3 - MULTI-PROPRIÉTÉS + IA RAISONNEMENT")
    print("=" * 80)
    gpu.print_summary()

    # Chercher le modèle Phase 2
    pretrained = args.pretrained_model or str(CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth")
    if not os.path.exists(pretrained):
        print(f"❌ Modèle Phase 2 introuvable : {pretrained}")
        print("   Lancez d'abord Phase 2 : python run_phase2.py --download")
        sys.exit(1)
    print(f"✓ Modèle Phase 2 : {pretrained}")

    # ── Étape 1 : Télécharger / charger les données ──────────────────
    data_dir = args.data_dir or str(EXTERNAL_DIR / "phase3")

    if args.download:
        print("\n📥 Téléchargement de tous les datasets...")
        data_paths = download_all_phase3_data(data_dir)

        # Sauvegarder les chemins pour utilisation future
        paths_file = os.path.join(data_dir, "data_paths.json")
        serializable_paths = {}
        for k, v in data_paths.items():
            if v is not None:
                serializable_paths[k] = v
        with open(paths_file, "w") as f:
            json.dump(serializable_paths, f, indent=2)
        print(f"✓ Chemins sauvegardés : {paths_file}")
    else:
        paths_file = os.path.join(data_dir, "data_paths.json")
        if not os.path.exists(paths_file):
            print("❌ Pas de données. Utilisez --download pour télécharger")
            sys.exit(1)
        with open(paths_file, "r") as f:
            data_paths = json.load(f)
        print(f"✓ Données chargées depuis {paths_file}")

    # ── Étape 2 : Fusionner les datasets ─────────────────────────────
    print("\n🔀 Fusion des datasets...")
    try:
        train_df = merge_datasets(data_paths, "train")
        val_df = merge_datasets(data_paths, "val")
    except Exception as e:
        print(f"❌ Erreur fusion : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Étape 3 : Créer les datasets PyTorch ─────────────────────────
    print("\n📊 Construction des datasets...")
    try:
        train_ds = MultiPropertyDataset(train_df)
        val_ds = MultiPropertyDataset(val_df)
    except Exception as e:
        print(f"❌ Erreur dataset : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"  Train : {len(train_ds)} molécules")
    print(f"  Val   : {len(val_ds)} molécules")

    # Ajuster batch_size selon la VRAM disponible
    batch_size = gpu.optimize_batch_size(args.batch_size, model_size_mb=50)
    print(f"  Batch size : {batch_size} (demandé: {args.batch_size})")

    torch.backends.cudnn.benchmark = True  # kernels cuDNN les plus rapides
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_multi_property, drop_last=True, **loader_kwargs(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_multi_property, **loader_kwargs(),
    )

    # ── Étape 4 : Construire le modèle ───────────────────────────────
    print("\n🏗️ Construction du modèle Phase 3...")

    # Encodeur de base
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )

    # Charger les poids Phase 2
    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        if encoder_sd:
            encoder.load_state_dict(encoder_sd, strict=False)
            print("  ✓ Encodeur Phase 2 chargé")
        else:
            print("  ⚠ Pas de poids encoder dans le checkpoint Phase 2")
    else:
        print("  ⚠ Format checkpoint non reconnu")

    # Modèle multi-propriétés
    model = MultiPropertyPredictor(
        encoder=encoder, hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT, freeze_encoder=True,
    ).to(device)

    # Warm-start : reprend le MEILLEUR modèle Phase 3 précédent s'il existe
    # (poursuite de l'entraînement sur les phases à venir).
    prev_best = Path(args.save_dir) / PHASE3["checkpoint_name"]
    if getattr(args, "warm_start", 1) and prev_best.exists():
        try:
            prev = torch.load(prev_best, map_location="cpu", weights_only=False)
            if isinstance(prev, dict) and "model_state_dict" in prev:
                model.load_state_dict(prev["model_state_dict"], strict=False)
                print("  ✓ Warm-start : meilleur modèle Phase 3 précédent rechargé")
        except Exception as e:
            print(f"  ⚠ Warm-start Phase 3 ignoré ({e})")

    # Raisonneur IA (sera entraîné après le multi-propriétés)
    reasoner = MolecularReasoner(
        mol_dim=OUTPUT_DIM,
        hidden_dim=PHASE3["reasoner_hidden_dim"],
        num_heads=PHASE3["reasoner_num_heads"],
        num_layers=PHASE3["reasoner_num_layers"],
        max_molecules=PHASE3["max_molecules_combo"],
        num_dose_levels=len(PHASE3["dose_levels"]),
        dropout=PHASE3["reasoner_dropout"],
    ).to(device)

    n_params_model = sum(pp.numel() for pp in model.parameters())
    n_params_reasoner = sum(pp.numel() for pp in reasoner.parameters())
    n_train_model = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  MultiPropertyPredictor : {n_params_model:,} params ({n_train_model:,} entraînables)")
    print(f"  MolecularReasoner      : {n_params_reasoner:,} params")
    print(f"  Total                  : {n_params_model + n_params_reasoner:,} params")

    # ── Étape 5 : Loss et Optimiseur ─────────────────────────────────
    tox_pos_weight = train_ds.get_pos_weight("toxicity")
    if tox_pos_weight is not None:
        tox_pos_weight = tox_pos_weight.to(device)
    criterion = MultiPropertyLoss(tox_pos_weight=tox_pos_weight).to(device)

    # LR mis à l'échelle pour le batch EFFECTIF (après optimize_batch_size) via
    # la règle racine carrée : compense les mises à jour/epoch réduites quand le
    # batch grossit (anti-biais). NB : le raisonneur (plus bas) s'entraîne sur des
    # combinaisons, pas sur ce batch → son LR n'est PAS mis à l'échelle.
    lr_scale = lr_scale_for_batch(batch_size, PHASE3["batch_size"])
    lr_enc = PHASE3["lr_encoder"] * lr_scale
    lr_hd = PHASE3["lr_heads"] * lr_scale
    if lr_scale > 1.0:
        print(f"  LR x{lr_scale:.2f} pour batch {batch_size} (ref {PHASE3['batch_size']}) : "
              f"encodeur {PHASE3['lr_encoder']:.2e}->{lr_enc:.2e}, "
              f"tetes {PHASE3['lr_heads']:.2e}->{lr_hd:.2e}")
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": lr_enc},
        {"params": model.shared_layer.parameters(), "lr": lr_hd},
        {"params": model.toxicity_head.parameters(), "lr": lr_hd},
        {"params": model.efficacy_head.parameters(), "lr": lr_hd},
        {"params": model.solubility_head.parameters(), "lr": lr_hd},
        {"params": model.lipophilicity_head.parameters(), "lr": lr_hd},
        {"params": model.bioavailability_head.parameters(), "lr": lr_hd},
        {"params": model.metabolic_stability_head.parameters(), "lr": lr_hd},
    ], weight_decay=PHASE3["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer, PHASE3["warmup_epochs"], args.epochs, PHASE3["lr_min"],
    )

    # ── Étape 6 : Boucle d'entraînement ──────────────────────────────
    best_score = 0.0
    no_improve = 0
    history = []
    best_path = Path(args.save_dir) / PHASE3["checkpoint_name"]
    latest_path = Path(args.save_dir) / "checkpoint_latest.pth"
    freeze_epochs = PHASE3["freeze_encoder_epochs"]
    # Checkpoints par-epoch (gérables/supprimables depuis le dashboard) ;
    # nouveau run -> dossier propre, cohérent avec live_metrics.jsonl réinitialisé.
    epochs_dir = Path(args.save_dir) / "epochs"
    save_epochs = bool(getattr(args, "save_epochs", 1))
    if epochs_dir.exists():
        shutil.rmtree(epochs_dir, ignore_errors=True)
    if save_epochs:
        epochs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("▶️ ENTRAÎNEMENT MULTI-PROPRIÉTÉS")
    print(f"  {args.epochs} epochs, freeze_encoder={freeze_epochs}, patience={args.patience}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    # Logger temps reel (lu par le tableau de bord pendant l'entrainement)
    live = LiveLogger(Path(args.save_dir) / "live_metrics.jsonl",
                      meta={"phase": "phase3", "epochs_total": args.epochs,
                            "conv_type": CONV_TYPE})

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Dégel progressif de l'encodeur
        if epoch <= freeze_epochs:
            model._freeze_encoder()
        else:
            model.gradual_unfreeze(epoch, freeze_epochs)

        scheduler.step(epoch - 1)
        lrs = scheduler.get_last_lr()

        # Entraînement
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PHASE3["grad_clip"], epoch,
        )

        # Validation
        val_m, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

        # Score composite (moyenne pondérée des métriques)
        score = 0.0
        n_scores = 0
        for prop, prop_m in val_m.items():
            if prop.startswith("_"):
                continue
            if isinstance(prop_m, dict):
                if "roc_auc" in prop_m:
                    score += prop_m["roc_auc"]
                    n_scores += 1
                elif "r2" in prop_m:
                    score += max(prop_m["r2"], 0)
                    n_scores += 1
        composite_score = score / max(n_scores, 1)

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))

        # Affichage
        train_loss = train_m.get("_loss", 0)
        val_loss = val_m.get("_loss", 0)

        prop_summary = []
        for prop in ["toxicity", "efficacy", "solubility", "lipophilicity",
                      "bioavailability", "metabolic_stability"]:
            if prop in val_m and isinstance(val_m[prop], dict):
                m = val_m[prop]
                if "roc_auc" in m:
                    prop_summary.append(f"{prop[:4]}={m['roc_auc']:.3f}")
                elif "r2" in m:
                    prop_summary.append(f"{prop[:4]}={m['r2']:.3f}")

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"t_loss={train_loss:.4f} v_loss={val_loss:.4f} | "
            f"score={composite_score:.4f} | "
            f"{' '.join(prop_summary)} | "
            f"lr={lrs[0]:.1e} | ETA {eta}"
        )

        # Historique
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "composite_score": composite_score,
            "lr": lrs[0],
        }
        for prop in val_m:
            if not prop.startswith("_") and isinstance(val_m[prop], dict):
                history_entry[f"val_{prop}"] = val_m[prop]
        history.append(history_entry)

        # Point temps reel pour le tableau de bord (multi-propriétés + sécurité tox).
        # Construit ici, journalisé après les checkpoints (pour inclure is_best/ckpt).
        rec = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "composite_score": composite_score, "lr": lrs[0],
        }
        with contextlib.suppress(Exception):
            scalar_map = {"toxicity": ("tox_auc", "roc_auc"),
                          "efficacy": ("eff_auc", "roc_auc"),
                          "bioavailability": ("bio_auc", "roc_auc"),
                          "metabolic_stability": ("metab_auc", "roc_auc"),
                          "solubility": ("sol_r2", "r2"),
                          "lipophilicity": ("lipo_r2", "r2")}
            for prop, (out_key, m_key) in scalar_map.items():
                if isinstance(val_m.get(prop), dict) and m_key in val_m[prop]:
                    rec[out_key] = val_m[prop][m_key]
            # AUC « phare » = toxicité (cohérent avec les autres phases)
            if "tox_auc" in rec:
                rec["val_auc"] = rec["tox_auc"]
            # Sécurité clinique depuis la tête toxicité
            if "toxicity" in val_preds and "toxicity" in val_targets:
                clin = clinical_summarize(
                    torch.sigmoid(val_preds["toxicity"]).cpu().numpy(),
                    val_targets["toxicity"].cpu().numpy())
                agg = clin["aggregate"]
                rec.update({
                    "macro_sensitivity": agg["macro_sensitivity"],
                    "macro_specificity": agg["macro_specificity"],
                    "macro_fnr": agg["macro_fnr"],
                    "n_danger": agg["n_danger"], "n_warn": agg["n_warn"],
                    "per_task_auc": {t["task"]: t["roc_auc"] for t in clin["tasks"]},
                })
        rec["clinical_score"] = clinical_score_fn(
            rec.get("val_auc"), rec.get("macro_sensitivity"),
            rec.get("macro_fnr"), rec.get("n_danger"))

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {k: v for k, v in val_m.items() if not k.startswith("_")},
            "composite_score": composite_score,
            "clinical_score": rec["clinical_score"],
            "best_score": best_score,
            "history": history,
            "property_dims": train_ds.get_property_dims(),
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "num_gnn_layers": NUM_GNN_LAYERS,
                "output_dim": OUTPUT_DIM,
                "dropout": DROPOUT,
                "atom_feature_dim": ATOM_FEATURE_DIM,
                "bond_feature_dim": BOND_FEATURE_DIM,
                "conv_type": CONV_TYPE,
                "attention_heads": ATTENTION_HEADS,
            },
        }
        torch.save(ckpt_data, latest_path)

        # Checkpoint PAR EPOCH (allégé : sans historique ni optimiseur).
        ckpt_rel = None
        if save_epochs:
            ep_path = epochs_dir / f"epoch_{epoch:03d}.pth"
            slim = {k: v for k, v in ckpt_data.items()
                    if k not in ("history", "optimizer_state_dict")}
            torch.save(slim, ep_path)
            ckpt_rel = f"epochs/epoch_{epoch:03d}.pth"

        is_best = composite_score > best_score
        if is_best:
            best_score = composite_score
            no_improve = 0
            torch.save(ckpt_data, best_path)
            print(f"  → Nouvelle meilleure epoch (score={best_score:.4f}) sauvegardée !")
        else:
            no_improve += 1

        rec["is_best"] = is_best
        rec["ckpt"] = ckpt_rel
        with contextlib.suppress(Exception):
            live.log(rec)

        if no_improve >= args.patience:
            print(f"  Early stopping (patience={args.patience}) — "
                  f"meilleure epoch conservée (score={best_score:.4f}).")
            break

        # Monitoring de santé
        if not health.step(val_loss):
            print("  ALERTE SANTÉ : trop de NaN, sauvegarde d'urgence...")
            emergency_save(
                model, optimizer, epoch,
                str(Path(args.save_dir) / "emergency_checkpoint.pth"),
            )
            break

        gpu.clear_memory()

    # ── Étape 7 : Entraîner le Raisonneur IA ─────────────────────────
    print(f"\n{'='*80}")
    print("🧠 ENTRAÎNEMENT DU MODULE IA DE RAISONNEMENT")
    print(f"{'='*80}\n")

    # Charger le meilleur modèle multi-propriétés
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Pré-calculer les embeddings + une "qualité pharmacologique" par molécule.
    # qualité = (1 - toxicité moyenne prédite) pondérée par l'efficacité prédite.
    # → sert de signal d'auto-supervision plus pertinent que la pure diversité.
    print("  Calcul des embeddings + qualité prédite...")
    all_embeddings = []
    all_quality = []
    with torch.no_grad():
        for batch_data, _ in tqdm(train_loader, desc="  Embeddings"):
            batch_data = batch_data.to(device)
            emb = model.encode(batch_data)
            preds = model(batch_data)

            tox = torch.sigmoid(preds["toxicity"]).mean(dim=1)        # [B] tox moyenne
            quality = 1.0 - tox                                        # moins toxique = mieux
            if "efficacy" in preds:
                eff = torch.sigmoid(preds["efficacy"]).view(-1)        # [B]
                quality = quality * (0.5 + 0.5 * eff)                  # pondère par l'efficacité

            all_embeddings.append(emb.cpu())
            all_quality.append(quality.cpu())
    all_embeddings = torch.cat(all_embeddings)
    all_quality = torch.cat(all_quality).clamp(0.0, 1.0)
    print(f"  {all_embeddings.shape[0]} embeddings calculés, dim={all_embeddings.shape[1]}")

    # Entraîner le raisonneur sur des combinaisons aléatoires
    reasoner_optimizer = AdamW(reasoner.parameters(), lr=PHASE3["lr_reasoner"])
    reasoner_criterion = ReasonerLoss()

    max_combo = PHASE3["max_molecules_combo"]
    n_combos_per_epoch = min(len(all_embeddings) // max_combo, 500)

    print(f"  {n_combos_per_epoch} combinaisons par epoch, max {max_combo} molécules")

    reasoner_epochs = 30
    for r_epoch in range(1, reasoner_epochs + 1):
        reasoner.train()
        total_r_loss = 0.0

        for _ in range(n_combos_per_epoch):
            # Échantillonner N molécules aléatoires
            n_mols = torch.randint(2, max_combo + 1, (1,)).item()
            indices = torch.randperm(len(all_embeddings))[:n_mols]
            combo_emb = all_embeddings[indices].unsqueeze(0).to(device)  # [1, N, D]

            # Padding si nécessaire
            if n_mols < max_combo:
                pad = torch.zeros(1, max_combo - n_mols, combo_emb.shape[-1], device=device)
                combo_emb = torch.cat([combo_emb, pad], dim=1)
                mask = torch.zeros(1, max_combo, dtype=torch.bool, device=device)
                mask[0, n_mols:] = True
            else:
                mask = None

            reasoner_optimizer.zero_grad()
            output = reasoner(combo_emb, mask)

            # Auto-supervision (proxy de synergie, en attendant des données
            # réelles type DrugComb) : combine la QUALITÉ pharmacologique prédite
            # des molécules (faible toxicité / bonne efficacité) et leur DIVERSITÉ
            # structurelle. Une bonne combinaison = molécules de qualité + complémentaires.
            quality = all_quality[indices].mean().item()
            diversity = torch.pdist(combo_emb[0, :n_mols]).mean().item()
            success = 0.7 * quality + 0.3 * min(diversity / 5.0, 1.0)
            success_target = torch.tensor([[success]], device=device)

            targets = {"success_labels": success_target}
            loss, _ = reasoner_criterion(output, targets)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
                reasoner_optimizer.step()
                total_r_loss += loss.item()

        avg_loss = total_r_loss / max(n_combos_per_epoch, 1)
        print(f"  Reasoner Epoch {r_epoch:2d}/{reasoner_epochs} | loss={avg_loss:.4f}")

    # ── Étape 8 : Sauvegarder le modèle complet ──────────────────────
    print(f"\n{'='*80}")
    print("💾 SAUVEGARDE DU MODÈLE COMPLET")
    print(f"{'='*80}")

    # Recharger le meilleur checkpoint et ajouter le raisonneur
    best_ckpt["reasoner_state_dict"] = reasoner.state_dict()
    best_ckpt["reasoner_config"] = {
        "hidden_dim": PHASE3["reasoner_hidden_dim"],
        "num_heads": PHASE3["reasoner_num_heads"],
        "num_layers": PHASE3["reasoner_num_layers"],
        "max_molecules": PHASE3["max_molecules_combo"],
        "num_dose_levels": len(PHASE3["dose_levels"]),
        "dose_levels": PHASE3["dose_levels"],
        "dropout": PHASE3["reasoner_dropout"],
    }
    torch.save(best_ckpt, best_path)

    total_time = time.time() - t0
    print(f"\n✅ Phase 3 terminée en {timedelta(seconds=int(total_time))}")
    print(f"   Meilleur score composite = {best_score:.4f}")
    print(f"   Modèle sauvegardé : {best_path}")

    # Sauvegarder l'historique
    log_path = log_dir / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"   Log : {log_path}")


if __name__ == "__main__":
    main()
