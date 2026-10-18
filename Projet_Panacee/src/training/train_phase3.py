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
import hashlib
import json
import logging
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

logger = logging.getLogger("panacee.phase3")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed

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
try:
    from src.models.encoder import MolecularEncoder
    from src.models.multi_property_head import MultiPropertyLoss, MultiPropertyPredictor
except ModuleNotFoundError as exc:  # métriques utilisables sans PyG installé
    if exc.name != "torch_geometric":
        raise
    MolecularEncoder = None
    MultiPropertyLoss = None
    MultiPropertyPredictor = None
from src.models.reasoner import MolecularReasoner, ReasonerLoss
try:
    from src.preprocessing.multi_property_loader import (
        MultiPropertyDataset,
        TaskBalancedSampler,
        collate_multi_property,
        download_all_phase3_data,
        merge_datasets,
    )
except ModuleNotFoundError as exc:  # métriques importables sans pandas/DeepChem
    if exc.name not in {"pandas", "deepchem", "torch_geometric"}:
        raise
    MultiPropertyDataset = TaskBalancedSampler = None
    collate_multi_property = download_all_phase3_data = merge_datasets = None
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

def find_optimal_thresholds_multi(preds_dict, targets_dict):
    """Cherche seuils optimaux (max F1) pour chaque tache classification."""
    classification_tasks = ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity")
    thresholds = {}
    for name in classification_tasks:
        if name not in preds_dict or name not in targets_dict:
            continue
        probs = torch.sigmoid(preds_dict[name]).cpu().numpy()
        tgt = targets_dict[name].cpu().numpy()
        valid = np.isfinite(tgt) & np.isfinite(probs)
        task_thresh = []
        for t in range(tgt.shape[1]):
            mask = valid[:, t]
            if mask.sum() < 2:
                task_thresh.append(0.5)
                continue
            y = tgt[mask, t]
            p = probs[mask, t]
            if len(np.unique(y)) < 2:
                task_thresh.append(0.5)
                continue
            best_t, best_f1 = 0.5, 0.0
            for th in np.arange(0.2, 0.8, 0.05):
                f = f1_score(y, (p > th).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_t = float(th)
            task_thresh.append(best_t)
        thresholds[name] = task_thresh
    return thresholds


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
        pred_np = pred.numpy()
        valid = np.isfinite(tgt) & np.isfinite(pred_np)
        skipped = int((np.isfinite(tgt) & ~np.isfinite(pred_np)).sum())
        if skipped:
            logger.warning("%s: %d prédictions non finies exclues des métriques", prop_name, skipped)

        if valid.sum() == 0:
            continue

        prop_metrics = {}

        if prop_name in ("toxicity", "bbb_penetration", "clinical_toxicity"):
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

            prop_metrics["n_tasks_scored"] = len(aucs)
            if aucs:
                prop_metrics["roc_auc"] = float(np.mean(aucs))
                prop_metrics["f1"] = float(np.mean(f1s))
            else:
                logger.warning("%s: AUC/F1 non calculables (aucune tâche avec deux classes)", prop_name)

        elif prop_name in ("efficacy",):
            # Classification binaire
            probs = torch.sigmoid(pred).numpy()
            y = tgt[valid].flatten()
            p = probs[valid[:, 0] if valid.ndim > 1 else valid].flatten()
            if len(y) > 0 and len(np.unique(y)) > 1:
                prop_metrics["n_tasks_scored"] = 1
                prop_metrics["roc_auc"] = float(roc_auc_score(y, p))
                prop_metrics["f1"] = float(f1_score(y, (p > 0.5).astype(int), zero_division=0))
            else:
                prop_metrics["n_tasks_scored"] = 0
                logger.warning("%s: AUC/F1 non calculables (une seule classe)", prop_name)

        else:
            # Régression : RMSE + R²
            pred_valid = pred_np[valid]
            tgt_valid = tgt[valid]
            rmse = float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
            ss_res = float(np.sum((tgt_valid - pred_valid) ** 2))
            ss_tot = float(np.sum((tgt_valid - tgt_valid.mean()) ** 2))
            # R² indéfini sur une cible constante : ne pas fabriquer un zéro.
            prop_metrics["rmse"] = rmse
            if ss_tot > 1e-8:
                prop_metrics["r2"] = float(1 - ss_res / ss_tot)
            else:
                logger.warning("%s: R² non calculable (cible constante)", prop_name)

        metrics[prop_name] = prop_metrics

    return metrics


# ======================================================================
# Boucles train / eval
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch,
                    scaler=None, accum_steps=1):
    if accum_steps < 1:
        raise ValueError("accum_steps doit être >= 1")
    model.train()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bbb_penetration", "clinical_toxicity"]}
    all_targets = {k: [] for k in all_preds}
    use_amp = scaler is not None
    pending_backward = 0

    def apply_gradients():
        nonlocal pending_backward
        if pending_backward == 0:
            return
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        pending_backward = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for step, (batch_data, labels) in enumerate(pbar):
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(batch_data)
        # BCE/Huber en FP32 : la réduction en FP16 peut déborder sur les
        # batches clairsemés des différentes tâches.
        loss, _ = criterion({k: v.float() for k, v in predictions.items()}, labels_device)

        if not torch.isfinite(loss).all() or not loss.requires_grad:
            logger.warning("Loss non finie ou sans gradient, batch ignore (step=%s)", step)
            continue

        loss_scaled = loss / accum_steps
        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        pending_backward += 1

        if pending_backward == accum_steps:
            apply_gradients()

        total_loss += loss.item() * batch_data.num_graphs
        n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach().float())
                all_targets[k].append(labels_device[k].detach())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Un batch ignoré ne crée aucun gradient et ne doit pas déclencher GradScaler.step.
    apply_gradients()
    if n == 0:
        raise RuntimeError("Aucun batch valide pendant cette epoch ; vérifier les labels et la loss")

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
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bbb_penetration", "clinical_toxicity"]}
    all_targets = {k: [] for k in all_preds}

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(batch_data)
        loss, _ = criterion({k: v.float() for k, v in predictions.items()}, labels_device)

        if torch.isfinite(loss).all():
            total_loss += loss.item() * batch_data.num_graphs
            n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach().float())
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
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (varier pour Deep Ensemble : --seed 42, 123, 456, ...)")
    args = p.parse_args()

    # Reproductibilité
    set_seed(args.seed)

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
        os.makedirs(data_dir, exist_ok=True)
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

    # Les six jeux doivent partager UN découpage global : aucun composé ni
    # scaffold ne peut contaminer validation/test via un autre jeu.
    from src.preprocessing.multi_property_loader import verify_global_split_integrity
    split_audit = verify_global_split_integrity(data_paths)
    print(f"✓ Partitions Phase 3 sans fuite de scaffold : {split_audit['counts']}")
    manifest_path = Path(data_dir) / "split_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Manifeste de partitions Phase 3 absent ; relancer avec --download")
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = split_manifest.get("sha256_generated_splits", {})
    actual_hashes = {f"{name}_{split}": hashlib.sha256(Path(path).read_bytes()).hexdigest()
                     for name, paths in data_paths.items() for split, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise RuntimeError("Empreintes des partitions Phase 3 différentes du manifeste ; relancer avec --download")
    split_provenance = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "split_sha256": actual_hashes,
                        "audit": split_audit,
                        "quarantined_compounds_by_dataset": {
                            name: entry["quarantined_compounds"]
                            for name, entry in split_manifest.get("duplicate_resolution", {}).items()
                        }}

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
        val_ds = MultiPropertyDataset(val_df, regression_stats=train_ds.regression_stats)
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

    torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic
    train_sampler = TaskBalancedSampler(train_ds, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
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
                state = prev["model_state_dict"]
                if all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
                    model.load_state_dict(state, strict=False)
                    print("  ✓ Warm-start : meilleur modèle Phase 3 précédent rechargé")
                else:
                    print("  ⚠ Checkpoint Phase 3 non fini : reprise depuis la Phase 2")
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
    def _pw(group):
        pw = train_ds.get_pos_weight(group)
        return pw.to(device) if pw is not None else None

    criterion = MultiPropertyLoss(
        tox_pos_weight=_pw("toxicity"),
        efficacy_pos_weight=_pw("efficacy"),
        bbb_pos_weight=_pw("bbb_penetration"),
        clintox_pos_weight=_pw("clinical_toxicity"),
    ).to(device)

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
    # Separer params : weight_decay exclus des biais et LayerNorm
    wd = PHASE3["weight_decay"]
    def _split_params(module, lr):
        decay, no_decay = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() <= 1 or "bias" in name or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    # Layer-wise LR decay (LLRD) : couches profondes = LR plus bas
    llrd_factor = PHASE3.get("llrd_factor", 0.8)
    n_enc_layers = model.encoder.num_layers
    param_groups = []
    # Embedding (couche la plus basse) → LR minimal
    param_groups.extend(_split_params(model.encoder.atom_embedding, lr_enc * llrd_factor ** n_enc_layers))
    # Couches GNN : decay progressif (couche 0 = plus bas, couche N-1 = lr_enc)
    for i in range(n_enc_layers):
        layer_lr = lr_enc * llrd_factor ** (n_enc_layers - 1 - i)
        param_groups.extend(_split_params(model.encoder.convs[i], layer_lr))
        param_groups.extend(_split_params(model.encoder.norms[i], layer_lr))
    # Virtual Node (si actif) → LR encodeur plein
    if hasattr(model.encoder, "virtual_node"):
        param_groups.extend(_split_params(model.encoder.virtual_node, lr_enc))
    # Pooling + projection → LR encodeur plein
    param_groups.extend(_split_params(model.encoder.pool_gate, lr_enc))
    param_groups.extend(_split_params(model.encoder.projection, lr_enc))
    # Têtes : LR plus élevé
    for head_name in ["shared_layer", "toxicity_head", "efficacy_head",
                      "solubility_head", "lipophilicity_head",
                      "bbb_penetration_head", "clinical_toxicity_head"]:
        param_groups.extend(_split_params(getattr(model, head_name), lr_hd))
    # Uncertainty weighting params (Kendall et al.) — appris conjointement
    if hasattr(criterion, "log_vars"):
        param_groups.append({"params": list(criterion.log_vars.parameters()),
                             "lr": lr_hd, "weight_decay": 0.0})
    optimizer = AdamW(param_groups)

    scheduler = WarmupCosineScheduler(
        optimizer, PHASE3["warmup_epochs"], args.epochs, PHASE3["lr_min"],
    )

    # AMP (mixed precision) — 2x throughput, moitié VRAM
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    accum_steps = PHASE3.get("gradient_accumulation_steps", 2)

    # EMA pour stabiliser les poids (decay=0.999)
    ema_decay = PHASE3.get("ema_decay", 0.999)
    ema_model = None
    if ema_decay > 0:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
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
            scaler=scaler, accum_steps=accum_steps,
        )
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(
                "Poids Phase 3 non finis après entraînement ; aucun checkpoint corrompu sauvegardé"
            )

        # EMA update
        if ema_model is not None:
            ema_model.update_parameters(model)

        # Validation (EMA si disponible, sinon modele brut)
        eval_model = ema_model if ema_model is not None else model
        val_m, val_preds, val_targets = evaluate(eval_model, val_loader, criterion, device, use_amp)

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
        if n_scores == 0:
            raise RuntimeError(
                "Aucune métrique de validation calculable en Phase 3 ; "
                "vérifier les classes, les cibles et les prédictions avant de sauvegarder"
            )
        composite_score = score / n_scores

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))

        # Affichage
        train_loss = train_m.get("_loss", 0)
        val_loss = val_m.get("_loss", 0)

        prop_summary = []
        for prop in ["toxicity", "efficacy", "solubility", "lipophilicity",
                      "bbb_penetration", "clinical_toxicity"]:
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
                          "bbb_penetration": ("bbb_auc", "roc_auc"),
                          "clinical_toxicity": ("clintox_auc", "roc_auc"),
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
            "model_state_dict": ema_model.module.state_dict() if ema_model is not None else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {k: v for k, v in val_m.items() if not k.startswith("_")},
            "composite_score": composite_score,
            "clinical_score": rec["clinical_score"],
            "best_score": best_score,
            "history": history,
            "property_dims": train_ds.get_property_dims(),
            "regression_stats": train_ds.regression_stats,
            "split_provenance": split_provenance,
            "training_stage": "training",
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

        # Securite d'abord : 60% clinical_score + 40% composite_score
        selection_score = 0.6 * rec["clinical_score"] + 0.4 * composite_score
        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
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

    # ── Étape 6b : Seuils optimaux + Évaluation TEST SET ───────────────
    print(f"\n{'='*80}")
    print("📋 SEUILS OPTIMAUX + ÉVALUATION TEST SET")
    print(f"{'='*80}")

    # Charger meilleur modele
    best_ckpt_eval = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt_eval["model_state_dict"], strict=False)

    # Seuils optimaux sur VAL set (max F1 par tache)
    print("\n  Recherche seuils optimaux sur val set...")
    val_m_final, val_preds_final, val_targets_final = evaluate(
        model, val_loader, criterion, device, use_amp)
    optimal_thresholds = find_optimal_thresholds_multi(val_preds_final, val_targets_final)
    for task_name, thresh_list in optimal_thresholds.items():
        mean_t = np.mean(thresh_list)
        print(f"    {task_name}: mean_thresh={mean_t:.3f} ({len(thresh_list)} sub-tasks)")
    best_ckpt_eval["optimal_thresholds"] = optimal_thresholds

    # Temperature scaling (calibration post-hoc sur val set)
    print("\n  Temperature scaling (calibration)...")
    temperatures = {}
    for task_name in ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"):
        if task_name not in val_preds_final or task_name not in val_targets_final:
            continue
        logits = val_preds_final[task_name].to(device)
        tgt = val_targets_final[task_name].to(device)
        valid = ~torch.isnan(tgt)
        if valid.sum() < 10:
            continue
        temp = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
        temp_opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)
        def _closure():
            temp_opt.zero_grad()
            scaled = logits / temp.clamp(min=0.1)
            tgt_safe = torch.where(valid, tgt, torch.zeros_like(tgt))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled, tgt_safe, reduction="none")
            loss = (loss * valid.float()).sum() / valid.sum().clamp(min=1)
            loss.backward()
            return loss
        temp_opt.step(_closure)
        temperatures[task_name] = float(temp.item())
        print(f"    {task_name}: T={temp.item():.3f}")
    best_ckpt_eval["temperatures"] = temperatures

    # Évaluation TEST SET (hold-out)
    try:
        test_df = merge_datasets(data_paths, "test")
        test_ds = MultiPropertyDataset(test_df, regression_stats=train_ds.regression_stats)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_multi_property, **loader_kwargs(),
        )
        test_m, _, _ = evaluate(model, test_loader, criterion, device, use_amp)
        print(f"\n  Test loss: {test_m.get('_loss', 0):.4f}")
        for prop in ["toxicity", "efficacy", "solubility", "lipophilicity",
                      "bbb_penetration", "clinical_toxicity"]:
            if prop in test_m and isinstance(test_m[prop], dict):
                m = test_m[prop]
                if "roc_auc" in m:
                    print(f"    {prop}: AUC={m['roc_auc']:.4f} F1={m.get('f1', 0):.4f}")
                elif "r2" in m:
                    print(f"    {prop}: R²={m['r2']:.4f} RMSE={m.get('rmse', 0):.4f}")
        best_ckpt_eval["test_metrics"] = {k: v for k, v in test_m.items() if not k.startswith("_")}
    except Exception as e:
        raise RuntimeError(f"Évaluation du jeu test Phase 3 impossible : {e}") from e

    best_ckpt_eval["training_stage"] = "evaluated"
    torch.save(best_ckpt_eval, best_path)

    # ── Étape 7 : Entraîner le Raisonneur IA ─────────────────────────
    print(f"\n{'='*80}")
    print("🧠 ENTRAÎNEMENT DU MODULE IA DE RAISONNEMENT")
    print(f"{'='*80}\n")

    # Charger le meilleur modèle multi-propriétés
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Pré-calculer embeddings + profil pharmacologique par molécule.
    # Chaque molécule reçoit un vecteur de propriétés prédites pour le scoring
    # combinatoire via le modèle Bliss independence (pharmacologiquement fondé).
    print("  Calcul des embeddings + profils pharmacologiques...")
    all_embeddings = []
    all_safety = []      # 1 - mean(tox_probs)
    all_efficacy = []    # P(actif)
    all_bbb = []         # P(BBB permeable)
    all_druglike = []    # score drug-likeness composite [0,1]
    with torch.no_grad():
        for batch_data, _ in tqdm(train_loader, desc="  Embeddings"):
            batch_data = batch_data.to(device)
            emb = model.encode(batch_data)
            preds = model(batch_data)

            tox = torch.sigmoid(preds["toxicity"]).mean(dim=1)
            safety = (1.0 - tox).cpu()
            eff = torch.sigmoid(preds["efficacy"]).view(-1).cpu() if "efficacy" in preds else torch.full((emb.shape[0],), 0.5)
            bbb = torch.sigmoid(preds["bbb_penetration"]).view(-1).cpu() if "bbb_penetration" in preds else torch.full((emb.shape[0],), 0.5)
            druglike = (0.4 * safety + 0.3 * eff + 0.2 * bbb + 0.1 * torch.ones_like(safety)).clamp(0, 1)

            all_embeddings.append(emb.cpu())
            all_safety.append(safety)
            all_efficacy.append(eff)
            all_bbb.append(bbb)
            all_druglike.append(druglike)
    all_embeddings = torch.cat(all_embeddings)
    all_safety = torch.cat(all_safety).clamp(0.0, 1.0)
    all_efficacy = torch.cat(all_efficacy).clamp(0.0, 1.0)
    all_bbb = torch.cat(all_bbb).clamp(0.0, 1.0)
    all_druglike = torch.cat(all_druglike).clamp(0.0, 1.0)
    print(f"  {all_embeddings.shape[0]} embeddings calculés, dim={all_embeddings.shape[1]}")

    # Entraîner le raisonneur sur des combinaisons aléatoires
    reasoner_optimizer = AdamW(reasoner.parameters(), lr=PHASE3["lr_reasoner"])
    reasoner_criterion = ReasonerLoss()
    reasoner_epochs = 30
    reasoner_scheduler = WarmupCosineScheduler(
        reasoner_optimizer, warmup_epochs=3, total_epochs=reasoner_epochs, lr_min=1e-6,
    )

    max_combo = PHASE3["max_molecules_combo"]
    n_combos_per_epoch = min(len(all_embeddings) // max_combo, 500)

    print(f"  {n_combos_per_epoch} combinaisons par epoch, max {max_combo} molécules")
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

            # Proxy pharmacologique (Bliss independence + multi-critères).
            # Bliss : P(combo_safe) = Π P(mol_i safe)  — indépendance des toxicités
            # Eff Bliss : P(combo_eff) = 1 - Π(1 - P(mol_i eff)) — au moins 1 actif
            # + diversité structurelle (molécules complémentaires = meilleure couverture)
            combo_safety = all_safety[indices].prod().item()          # Bliss safety
            combo_eff = 1.0 - (1.0 - all_efficacy[indices]).prod().item()  # Bliss efficacy
            combo_bbb = all_bbb[indices].mean().item()
            diversity = torch.pdist(combo_emb[0, :n_mols]).mean().item()
            div_norm = min(diversity / 5.0, 1.0)
            success = (0.35 * combo_safety + 0.30 * combo_eff
                       + 0.15 * combo_bbb + 0.20 * div_norm)
            success_target = torch.tensor([[success]], device=device)

            targets = {"success_labels": success_target}
            loss, _ = reasoner_criterion(output, targets)

            if torch.isfinite(loss).all():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
                reasoner_optimizer.step()
                total_r_loss += loss.item()

        reasoner_scheduler.step(r_epoch - 1)
        avg_loss = total_r_loss / max(n_combos_per_epoch, 1)
        print(f"  Reasoner Epoch {r_epoch:2d}/{reasoner_epochs} | loss={avg_loss:.4f}"
              f" | lr={reasoner_scheduler.get_last_lr()[0]:.1e}")

    # ── Étape 8 : Sauvegarder le modèle complet ──────────────────────
    print(f"\n{'='*80}")
    print("💾 SAUVEGARDE DU MODÈLE COMPLET")
    print(f"{'='*80}")

    # Recharger le meilleur checkpoint et ajouter le raisonneur
    if not all(torch.isfinite(p).all() for p in reasoner.parameters()):
        raise RuntimeError("Poids du raisonneur non finis ; checkpoint Phase 3 non finalisé")
    best_ckpt["reasoner_state_dict"] = reasoner.state_dict()
    best_ckpt["training_stage"] = "complete"
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
