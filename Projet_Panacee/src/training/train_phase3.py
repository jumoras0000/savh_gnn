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
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("panacee.phase3")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed
from src.utils.prediction_contract import calibrated_logits

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

def phase3_parameter_groups(module, lr, weight_decay):
    """Include frozen parameters: they must remain optimizable after unfreezing."""
    decay, no_decay = [], []
    for name, parameter in module.named_parameters():
        if parameter.dim() <= 1 or "bias" in name or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": params, "lr": lr, "weight_decay": wd}
        for params, wd in ((decay, weight_decay), (no_decay, 0.0)) if params
    ]


def validate_warm_start(checkpoint, split_provenance, regression_stats,
                        expected_feature_schema=None):
    if checkpoint.get("split_provenance", {}).get("split_sha256") != split_provenance.get("split_sha256"):
        raise ValueError("Warm-start incompatible avec les partitions actuelles ; utiliser --warm_start 0")
    if checkpoint.get("regression_stats") != regression_stats:
        raise ValueError("Warm-start incompatible avec la normalisation actuelle ; utiliser --warm_start 0")
    if (expected_feature_schema is not None and
            checkpoint.get("config", {}).get("feature_schema") != expected_feature_schema):
        raise ValueError("Warm-start incompatible avec le schéma chimique actuel ; utiliser --warm_start 0")
    if not all(torch.isfinite(v).all() for v in checkpoint["model_state_dict"].values()
               if torch.is_tensor(v)):
        raise ValueError("Warm-start contenant des poids non finis")


def audit_phase2_exposure(checkpoint, data_paths):
    """Reject known supervised upstream overlap with Phase 3 validation/test."""
    exposure = checkpoint.get("phase2_exposure")
    if exposure is None:
        return {"status": "unverified", "reason": "legacy_checkpoint_without_exposure"}
    if (not isinstance(exposure, dict) or
            not isinstance(exposure.get("canonical_smiles"), list) or
            not isinstance(exposure.get("scaffolds"), list) or
            exposure.get("includes_validation_selection") is not True):
        raise ValueError("Provenance Phase 2 incomplète ou invalide")
    from src.preprocessing.multi_property_loader import _chemical_key

    phase2_molecules = set(exposure["canonical_smiles"])
    phase2_scaffolds = set(exposure["scaffolds"])
    overlaps = {}
    for split in ("val", "test"):
        molecules, scaffolds = set(), set()
        for paths in data_paths.values():
            frame = pd.read_csv(paths[split], usecols=["smiles"])
            for smiles in frame["smiles"]:
                canonical, scaffold = _chemical_key(smiles)
                molecules.add(canonical)
                scaffolds.add(scaffold)
        overlaps[split] = {"molecules": len(phase2_molecules & molecules),
                           "scaffolds": len(phase2_scaffolds & scaffolds)}
    if any(count for split in overlaps.values() for count in split.values()):
        raise ValueError(f"Fuite supervisée Phase 2 vers Phase 3 : {overlaps}. "
                         "Réentraîner la Phase 2 sur le train Phase 3 uniquement")
    return {"status": "verified_no_supervised_overlap", "overlaps": overlaps,
            "phase2_molecules": len(phase2_molecules)}


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
        logits = preds_dict[name].detach().float().cpu()
        probs = torch.sigmoid(logits).numpy()
        tgt = targets_dict[name].detach().cpu().numpy()
        valid = np.isfinite(tgt) & np.isfinite(logits.numpy())
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
            precision, recall, candidates = precision_recall_curve(y, p)
            scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            # Consumers use >, while sklearn's PR curve uses >=.
            best_t = float(np.nextafter(candidates[int(np.argmax(scores))], np.float32(-np.inf)))
            task_thresh.append(max(0.0, best_t))
        thresholds[name] = task_thresh
    return thresholds


def fit_temperature(logits, targets):
    """Fit a positive scalar on observed validation labels; never worsen NLL."""
    valid = torch.isfinite(targets) & torch.isfinite(logits)
    if valid.sum() < 10:
        return 1.0
    logits, targets = logits[valid].detach().float(), targets[valid].detach().float()
    log_temp = torch.nn.Parameter(torch.zeros((), device=logits.device))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_temp.clamp(-2.302585, 4.605170).exp(), targets)
        loss.backward()
        return loss
    baseline = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
    opt.step(closure)
    temperature = float(log_temp.detach().clamp(-2.302585, 4.605170).exp())
    final = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets).item()
    return temperature if np.isfinite(final) and final <= baseline else 1.0


def phase3_composite_score(metrics):
    """Score de sélection; conserve la pénalité d'un R² négatif."""
    values = []
    for name, result in metrics.items():
        if name.startswith("_") or not isinstance(result, dict):
            continue
        if "roc_auc" in result:
            if not np.isfinite(result["roc_auc"]):
                raise ValueError(f"{name}: AUC non finie")
            values.append(result["roc_auc"])
        elif "r2" in result:
            if not np.isfinite(result["r2"]):
                raise ValueError(f"{name}: R² non fini")
            values.append(max(-1.0, min(1.0, result["r2"])))
    if not values:
        raise RuntimeError("Aucune métrique de validation calculable en Phase 3")
    return sum(values) / len(values)


def phase3_selection_score(metrics, clinical_score, policy="multitask"):
    """Predeclared validation objective; never interpret it as clinical reliability."""
    composite = phase3_composite_score(metrics)
    if policy == "multitask":
        return composite
    if policy == "hiv_activity":
        value = metrics.get("efficacy", {}).get("average_precision")
        if value is None or not np.isfinite(value):
            raise RuntimeError("AUPRC VIH absente : sélection hiv_activity impossible")
        return float(value)
    if policy == "legacy_toxicity":
        return 0.6 * clinical_score + 0.4 * composite
    raise ValueError(f"Politique de sélection inconnue : {policy}")


def should_select_phase3_epoch(epoch, freeze_epochs, score, best_score):
    """A final Phase 3 checkpoint must include encoder adaptation."""
    return epoch > freeze_epochs and np.isfinite(score) and score > best_score


def compute_phase3_metrics(predictions, targets, thresholds=None, regression_stats=None):
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

        if prop_name in ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"):
            # Classification : AUC + F1
            probs = torch.sigmoid(pred).numpy()
            aucs = []
            f1s = []
            aps, precisions, recalls = [], [], []
            task_thresholds = (thresholds or {}).get(prop_name, [0.5] * tgt.shape[1])
            if len(task_thresholds) != tgt.shape[1]:
                raise ValueError(f"{prop_name}: nombre de seuils incorrect")
            for t in range(tgt.shape[1]):
                mask = valid[:, t]
                if mask.sum() < 2:
                    continue
                y = tgt[mask, t]
                p = probs[mask, t]
                # AUC et F1 seulement si les DEUX classes sont présentes :
                # un 0 fictif (tâche mono-classe) fausserait la macro-moyenne.
                if len(np.unique(y)) > 1:
                    decisions = (p > task_thresholds[t]).astype(int)
                    aucs.append(roc_auc_score(y, p))
                    f1s.append(f1_score(y, decisions, zero_division=0))
                    aps.append(average_precision_score(y, p))
                    precisions.append(precision_score(y, decisions, zero_division=0))
                    recalls.append(recall_score(y, decisions, zero_division=0))

            prop_metrics["n_tasks_scored"] = len(aucs)
            if aucs:
                prop_metrics["roc_auc"] = float(np.mean(aucs))
                prop_metrics["f1"] = float(np.mean(f1s))
                prop_metrics["average_precision"] = float(np.mean(aps))
                prop_metrics["precision"] = float(np.mean(precisions))
                prop_metrics["recall"] = float(np.mean(recalls))
                prop_metrics["positive_fraction"] = float(tgt[valid].mean())
            else:
                logger.warning("%s: AUC/F1 non calculables (aucune tâche avec deux classes)", prop_name)

        else:
            # Régression : RMSE + R²
            pred_valid = pred_np[valid]
            tgt_valid = tgt[valid]
            stats = (regression_stats or {}).get(prop_name)
            if stats is not None:
                pred_valid = pred_valid * stats["std"] + stats["mean"]
                tgt_valid = tgt_valid * stats["std"] + stats["mean"]
            rmse = float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
            ss_res = float(np.sum((tgt_valid - pred_valid) ** 2))
            ss_tot = float(np.sum((tgt_valid - tgt_valid.mean()) ** 2))
            # R² indéfini sur une cible constante : ne pas fabriquer un zéro.
            prop_metrics["rmse"] = rmse
            prop_metrics["units"] = "original_target" if stats is not None else "normalized_target"
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
                    scaler=None, accum_steps=1, use_amp=None, amp_dtype=torch.float16,
                    ema_model=None):
    if accum_steps < 1:
        raise ValueError("accum_steps doit être >= 1")
    model.train()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bbb_penetration", "clinical_toxicity"]}
    all_targets = {k: [] for k in all_preds}
    use_amp = scaler is not None if use_amp is None else use_amp
    pending_backward = 0
    bad_batches = 0
    optimizer_updates = 0
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]

    def finite_state():
        return (all(torch.isfinite(p).all() for p in optimizer_params) and
                all(torch.isfinite(b).all() for b in model.buffers() if b.is_floating_point()))

    def apply_gradients():
        nonlocal pending_backward, bad_batches, optimizer_updates
        if pending_backward == 0:
            return
        if scaler is not None:
            scaler.unscale_(optimizer)
        # The final accumulation window can contain fewer batches. Its
        # gradients must still be an average, not shrink by accum_steps.
        if pending_backward < accum_steps:
            for parameter in optimizer_params:
                if parameter.grad is not None:
                    parameter.grad.mul_(accum_steps / pending_backward)
        gradients_finite = all(torch.isfinite(p.grad).all() for p in optimizer_params
                               if p.grad is not None)
        if gradients_finite:
            torch.nn.utils.clip_grad_norm_(optimizer_params, grad_clip,
                                            error_if_nonfinite=True)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_updates += 1
            if not finite_state():
                raise RuntimeError("État non fini après optimizer.step ; entraînement arrêté")
            if ema_model is not None:
                ema_model.update_parameters(model)
        else:
            bad_batches += 1
            logger.warning("Gradients non finis ; mise à jour ignorée (epoch=%s)", epoch)
            if scaler is not None:
                # Le scaler a enregistré les inf/NaN lors de unscale_ ; adapter
                # son échelle sans appliquer de mise à jour corrompue.
                scaler.update()
        optimizer.zero_grad()
        pending_backward = 0
        if bad_batches >= 3:
            raise RuntimeError("Trop de batches/gradients non finis ; entraînement arrêté")

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for step, (batch_data, labels) in enumerate(pbar):
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        # BatchNorm modifie ses buffers pendant le forward, même si la loss est
        # ensuite rejetée. Les restaurer si ce batch les a rendus non finis.
        buffer_snapshot = [(b, b.detach().clone()) for b in model.buffers()
                           if b.is_floating_point()]
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            predictions = model(batch_data)
        nonfinite_heads = [name for name, output in predictions.items()
                           if not torch.isfinite(output).all()]
        nonfinite_buffers = any(not torch.isfinite(buffer).all()
                                for buffer, _ in buffer_snapshot)
        if nonfinite_heads or nonfinite_buffers:
            bad_batches += 1
            for buffer, previous in buffer_snapshot:
                buffer.copy_(previous)
            logger.warning("Forward non fini, batch ignoré (step=%s, têtes=%s, buffers=%s)",
                           step, nonfinite_heads, nonfinite_buffers)
            if not finite_state() or bad_batches >= 3:
                raise RuntimeError("Batches non finis répétés ou état non fini ; entraînement arrêté")
            continue
        # BCE/Huber en FP32 : la réduction en FP16 peut déborder sur les
        # batches clairsemés des différentes tâches.
        loss, _ = criterion({k: v.float() for k, v in predictions.items()}, labels_device)

        if not torch.isfinite(loss).all() or not loss.requires_grad:
            bad_batches += 1
            for buffer, previous in buffer_snapshot:
                buffer.copy_(previous)
            logger.warning("Loss non finie ou sans gradient, batch ignoré (step=%s)", step)
            if not finite_state() or bad_batches >= 3:
                raise RuntimeError("Batches non finis répétés ou état non fini ; entraînement arrêté")
            continue

        loss_scaled = loss / accum_steps
        if scaler is not None:
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
    if not finite_state():
        raise RuntimeError("Poids ou buffers non finis après entraînement ; aucun checkpoint sauvegardé")
    if n == 0:
        raise RuntimeError("Aucun batch valide pendant cette epoch ; vérifier les labels et la loss")
    if optimizer_updates == 0:
        raise RuntimeError("Aucune mise à jour de l'optimiseur pendant cette epoch")

    merged_preds = {}
    merged_targets = {}
    for k in all_preds:
        if all_preds[k]:
            merged_preds[k] = torch.cat(all_preds[k])
            merged_targets[k] = torch.cat(all_targets[k])

    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    metrics["_loss"] = total_loss / max(n, 1)
    metrics["_bad_batches"] = bad_batches
    metrics["_optimizer_updates"] = optimizer_updates
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bbb_penetration", "clinical_toxicity"]}
    all_targets = {k: [] for k in all_preds}

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            predictions = model(batch_data)
        if any(not torch.isfinite(value).all() for value in predictions.values()):
            raise RuntimeError("Prédictions non finies pendant l'évaluation")
        loss, _ = criterion({k: v.float() for k, v in predictions.items()}, labels_device)

        if not torch.isfinite(loss).all():
            raise RuntimeError("Loss non finie pendant l'évaluation")
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

    if n == 0:
        raise RuntimeError("Aucun exemple à évaluer")
    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    # Sparse multi-task labels: averaging batch losses weights tasks according
    # to which batches they appear in. Evaluate each task over all its labels.
    full_loss, _ = criterion(merged_preds, merged_targets)
    if not torch.isfinite(full_loss).all():
        raise RuntimeError("Loss globale de validation non finie")
    metrics["_loss"] = full_loss.item()
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
    p.add_argument("--selection_policy", choices=("multitask", "hiv_activity", "legacy_toxicity"),
                   default="multitask", help="Objectif de validation, défini avant entraînement")
    p.add_argument("--warm_start", type=int, default=1,
                   help="1=reprend le meilleur modele Phase 3 precedent (transfert)")
    p.add_argument("--save_epochs", type=int, default=1,
                   help="1=sauvegarde un checkpoint par epoch (gerable depuis le dashboard)")
    p.add_argument("--device", type=str, default=str(DEVICE))
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (varier pour Deep Ensemble : --seed 42, 123, 456, ...)")
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2:
        p.error("epochs/patience doivent être >= 1 et batch_size >= 2 (BatchNorm)")
    if args.epochs <= PHASE3["freeze_encoder_epochs"]:
        p.error("epochs doit dépasser freeze_encoder_epochs : aucun encodeur adapté sinon")

    # Reproductibilité
    set_seed(args.seed)

    # Initialisation GPU et logging
    gpu = get_gpu_manager(force_cpu=(args.device == "cpu"), device=args.device)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandé mais indisponible")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Phase 3 supporte cpu ou cuda[:index]")
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
    from src.preprocessing.multi_property_loader import verify_phase3_source_provenance
    source_audit = verify_phase3_source_provenance(split_manifest, data_dir)
    print(f"✓ Provenance des observations en quarantaine vérifiée : {source_audit}")
    split_provenance = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "split_sha256": actual_hashes,
                        "audit": split_audit,
                        "quarantine_source_rows_checked": source_audit,
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
    batch_size = max(2, gpu.optimize_batch_size(args.batch_size, model_size_mb=50))
    if len(train_ds) < 2:
        raise ValueError("Phase 3 nécessite au moins deux graphes d'entraînement (BatchNorm)")
    print(f"  Batch size : {batch_size} (demandé: {args.batch_size})")

    torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic
    train_sampler = TaskBalancedSampler(train_ds, batch_size=batch_size, shuffle=True)
    coverage = {name: len(indices) for name, indices in train_sampler.task_indices.items()}
    print(f"  Étiquettes observées par groupe : {coverage}")
    print("  Sampler : intercalage jusqu'à épuisement des petits groupes ; "
          "la fin d'époque peut être dominée par les grands jeux")
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
        use_chirality=PHASE3["use_chirality"],
    )

    # Charger les poids Phase 2
    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    pretrained_provenance = {
        "checkpoint_sha256": hashlib.sha256(Path(pretrained).read_bytes()).hexdigest(),
        "upstream_split_audit": audit_phase2_exposure(ckpt, data_paths),
    }
    if pretrained_provenance["upstream_split_audit"]["status"] == "unverified":
        logger.warning("Partitions Phase 3 vérifiées, mais exposition antérieure de l'encodeur Phase 2 aux données de validation/test non vérifiée")
    else:
        print("  ✓ Aucune fuite supervisée Phase 2 détectée vers val/test Phase 3")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        if encoder_sd:
            if not all(torch.isfinite(v).all() for v in encoder_sd.values() if torch.is_tensor(v)):
                raise ValueError("Poids de l'encodeur Phase 2 non finis")
            result = encoder.load_state_dict(encoder_sd, strict=False)
            allowed_missing = ({"chirality_embedding.weight"}
                               if PHASE3["use_chirality"] else set())
            if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
                raise ValueError("Encodeur Phase 2 incompatible avec le schéma Phase 3 : "
                                 f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
            print("  ✓ Encodeur Phase 2 chargé")
        else:
            raise ValueError("Pas de poids encoder dans le checkpoint Phase 2")
    else:
        raise ValueError("Format checkpoint Phase 2 non reconnu")

    # Modèle multi-propriétés
    model = MultiPropertyPredictor(
        encoder=encoder, hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT, freeze_encoder=True,
    ).to(device)

    # Warm-start : reprend le MEILLEUR modèle Phase 3 précédent s'il existe
    # (poursuite de l'entraînement sur les phases à venir).
    prev_best = Path(args.save_dir) / PHASE3["checkpoint_name"]
    feature_schema = ("atom9_bond6_cip_atom_v2" if PHASE3["use_chirality"]
                      else "atom9_bond6_legacy_v1")
    if getattr(args, "warm_start", 1) and prev_best.exists():
        prev = torch.load(prev_best, map_location="cpu", weights_only=False)
        validate_warm_start(prev, split_provenance, train_ds.regression_stats,
                            expected_feature_schema=feature_schema)
        model.load_state_dict(prev["model_state_dict"], strict=True)
        print("  ✓ Warm-start : partitions et normalisation vérifiées")

    # Raisonneur IA (sera entraîné après le multi-propriétés)
    reasoner = MolecularReasoner(
        mol_dim=HIDDEN_DIM,
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
        return phase3_parameter_groups(module, lr, wd)

    # Layer-wise LR decay (LLRD) : couches profondes = LR plus bas
    llrd_factor = PHASE3.get("llrd_factor", 0.8)
    n_enc_layers = model.encoder.num_layers
    param_groups = []
    # Embedding (couche la plus basse) → LR minimal
    param_groups.extend(_split_params(model.encoder.atom_embedding, lr_enc * llrd_factor ** n_enc_layers))
    if PHASE3["use_chirality"]:
        param_groups.extend(_split_params(model.encoder.chirality_embedding,
                                          lr_enc * llrd_factor ** n_enc_layers))
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
    optimized_ids = {id(p) for group in param_groups for p in group["params"]}
    missing = [name for name, p in model.named_parameters() if id(p) not in optimized_ids]
    if missing:
        raise RuntimeError(f"Paramètres absents de l'optimiseur : {missing}")

    scheduler = WarmupCosineScheduler(
        optimizer, PHASE3["warmup_epochs"], args.epochs, PHASE3["lr_min"],
    )

    # BF16 conserve l'étendue dynamique du FP32 et évite une classe de
    # débordements FP16 sur les graphes/molécules de tailles variables.
    use_amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    print(f"  Précision AMP : {amp_dtype if use_amp else 'FP32'}")
    accum_steps = PHASE3.get("gradient_accumulation_steps", 2)

    # EMA pour stabiliser les poids (decay=0.999)
    ema_decay = PHASE3.get("ema_decay", 0.999)
    ema_model = None
    if ema_decay > 0:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            use_buffers=True,
        )

    # ── Étape 6 : Boucle d'entraînement ──────────────────────────────
    best_score = -float("inf")
    no_improve = 0
    history = []
    best_path = Path(args.save_dir) / PHASE3["checkpoint_name"]
    latest_path = Path(args.save_dir) / "checkpoint_latest.pth"
    if best_path.exists():
        archived = best_path.with_name(
            f"panacee_phase3_previous_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pth")
        best_path.rename(archived)
        print(f"  Ancien checkpoint préservé : {archived}")
    freeze_epochs = PHASE3["freeze_encoder_epochs"]
    # Checkpoints par-epoch (gérables/supprimables depuis le dashboard) ;
    # nouveau run -> dossier propre, cohérent avec live_metrics.jsonl réinitialisé.
    epochs_dir = Path(args.save_dir) / "epochs"
    save_epochs = bool(getattr(args, "save_epochs", 1))
    if epochs_dir.exists() and any(epochs_dir.iterdir()):
        archive = epochs_dir.with_name("epochs_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        epochs_dir.rename(archive)
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
            use_amp=use_amp, amp_dtype=amp_dtype,
            ema_model=ema_model,
        )
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(
                "Poids Phase 3 non finis après entraînement ; aucun checkpoint corrompu sauvegardé"
            )

        # Validation (EMA si disponible, sinon modele brut)
        eval_model = ema_model if ema_model is not None else model
        val_m, val_preds, val_targets = evaluate(
            eval_model, val_loader, criterion, device, use_amp, amp_dtype)

        # Score composite (moyenne pondérée des métriques)
        composite_score = phase3_composite_score(val_m)

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
            f"bad_batches={train_m.get('_bad_batches', 0)} "
            f"updates={train_m.get('_optimizer_updates', 0)} | "
            f"{' '.join(prop_summary)} | "
            f"lr={lrs[0]:.1e} | ETA {eta}"
        )

        # Historique
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bad_batches": train_m.get("_bad_batches", 0),
            "optimizer_updates": train_m.get("_optimizer_updates", 0),
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
            "train_bad_batches": train_m.get("_bad_batches", 0),
            "optimizer_updates": train_m.get("_optimizer_updates", 0),
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

        selection_score = phase3_selection_score(val_m, rec["clinical_score"],
                                                 args.selection_policy)
        if not np.isfinite(selection_score):
            raise RuntimeError("Score de sélection non fini")
        eligible = epoch > freeze_epochs
        is_best = should_select_phase3_epoch(epoch, freeze_epochs,
                                             selection_score, best_score)
        if is_best:
            best_score = selection_score
        rec["selection_score"] = selection_score
        rec["selection_policy"] = args.selection_policy
        rec["selection_eligible"] = eligible
        history_entry["selection_score"] = selection_score
        history_entry["selection_eligible"] = eligible
        print(f"  sélection={selection_score:.4f} ({args.selection_policy}, "
              f"{'éligible' if eligible else 'encodeur gelé'}) ; "
              f"proxy toxicité={rec['clinical_score']:.4f}")

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": ema_model.module.state_dict() if ema_model is not None else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "raw_model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "selection_score": selection_score,
            "selection_policy": args.selection_policy,
            "selection_eligible": eligible,
            "val_metrics": {k: v for k, v in val_m.items() if not k.startswith("_")},
            "composite_score": composite_score,
            "clinical_score": rec["clinical_score"],
            "best_score": best_score,
            "history": history,
            "property_dims": train_ds.get_property_dims(),
            "regression_stats": train_ds.regression_stats,
            "split_provenance": split_provenance,
            "pretrained_provenance": pretrained_provenance,
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
                "use_chirality": PHASE3["use_chirality"],
                "feature_schema": feature_schema,
            },
        }
        torch.save(ckpt_data, latest_path)

        # Checkpoint PAR EPOCH (allégé : sans historique ni optimiseur).
        ckpt_rel = None
        if save_epochs:
            ep_path = epochs_dir / f"epoch_{epoch:03d}.pth"
            slim = {k: v for k, v in ckpt_data.items()
                    if k not in ("history", "optimizer_state_dict", "raw_model_state_dict")}
            torch.save(slim, ep_path)
            ckpt_rel = f"epochs/epoch_{epoch:03d}.pth"

        # Only a checkpoint trained after encoder unfreezing can be selected.
        if is_best:
            no_improve = 0
            torch.save(ckpt_data, best_path)
            print(f"  → Nouvelle meilleure epoch (score={best_score:.4f}) sauvegardée !")
        elif eligible:
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
            raise RuntimeError("Entraînement interrompu par le contrôle de santé")

        gpu.clear_memory()

    # ── Étape 6b : Seuils optimaux + Évaluation TEST SET ───────────────
    print(f"\n{'='*80}")
    print("📋 SEUILS OPTIMAUX + ÉVALUATION TEST SET")
    print(f"{'='*80}")

    # Charger meilleur modele
    best_ckpt_eval = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt_eval["model_state_dict"], strict=True)
    criterion.load_state_dict(best_ckpt_eval["criterion_state_dict"])

    # Seuils optimaux sur VAL set (max F1 par tache)
    print("\n  Recherche seuils optimaux sur val set...")
    val_m_final, val_preds_final, val_targets_final = evaluate(
        model, val_loader, criterion, device, use_amp, amp_dtype)
    # Temperature scaling (calibration post-hoc sur val set)
    print("\n  Temperature scaling (calibration)...")
    temperatures = {}
    for task_name in ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"):
        if task_name not in val_preds_final or task_name not in val_targets_final:
            continue
        logits = val_preds_final[task_name].to(device)
        tgt = val_targets_final[task_name].to(device)
        temperatures[task_name] = fit_temperature(logits, tgt)
        print(f"    {task_name}: T={temperatures[task_name]:.3f}")
    best_ckpt_eval["temperatures"] = temperatures
    optimal_thresholds = find_optimal_thresholds_multi(
        calibrated_logits(val_preds_final, temperatures), val_targets_final)
    best_ckpt_eval["optimal_thresholds"] = optimal_thresholds
    best_ckpt_eval["threshold_space"] = "calibrated_probability"

    # Évaluation TEST SET (hold-out)
    try:
        test_df = merge_datasets(data_paths, "test")
        test_ds = MultiPropertyDataset(test_df, regression_stats=train_ds.regression_stats)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_multi_property, **loader_kwargs(),
        )
        raw_test_m, test_preds, test_targets = evaluate(model, test_loader, criterion, device, use_amp, amp_dtype)
        test_m = compute_phase3_metrics(calibrated_logits(test_preds, temperatures), test_targets,
                                        thresholds=optimal_thresholds,
                                        regression_stats=train_ds.regression_stats)
        test_m["_loss"] = raw_test_m["_loss"]
        best_ckpt_eval["test_metric_convention"] = {
            "classification": "calibrated_probability_with_validation_thresholds",
            "regression": "original_target_units", "loss": "raw_logits_with_saved_criterion"}
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
    # combinatoire heuristique, sans mesure expérimentale de synergie.
    print("  Calcul des embeddings + profils pharmacologiques...")
    all_embeddings = []
    all_safety = []      # 1 - mean(tox_probs)
    all_efficacy = []    # P(actif)
    all_bbb = []         # P(BBB permeable)
    all_druglike = []    # score drug-likeness composite [0,1]
    with torch.no_grad():
        embedding_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_multi_property, **loader_kwargs())
        for batch_data, _ in tqdm(embedding_loader, desc="  Embeddings"):
            batch_data = batch_data.to(device)
            emb = model.encode(batch_data)
            preds = calibrated_logits(model(batch_data), best_ckpt.get("temperatures", {}))

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

    max_combo = min(PHASE3["max_molecules_combo"], len(all_embeddings))
    if max_combo < 2:
        raise RuntimeError("Au moins deux molécules requises pour entraîner le raisonneur")
    n_combos_per_epoch = min(max(1, len(all_embeddings) // max_combo), 500)

    print(f"  {n_combos_per_epoch} combinaisons par epoch, max {max_combo} molécules")
    for r_epoch in range(1, reasoner_epochs + 1):
        reasoner_scheduler.step(r_epoch - 1)
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

            # Proxy heuristique issu des prédictions individuelles. Les produits
            # supposent une indépendance, mais ne constituent PAS une mesure de
            # synergie Bliss : aucune courbe dose-réponse expérimentale ici.
            combo_safety = all_safety[indices].prod().item()
            combo_eff = 1.0 - (1.0 - all_efficacy[indices]).prod().item()
            combo_bbb = all_bbb[indices].mean().item()
            diversity = torch.pdist(combo_emb[0, :n_mols]).mean().item()
            div_norm = min(diversity / 5.0, 1.0)
            success = (0.35 * combo_safety + 0.30 * combo_eff
                       + 0.15 * combo_bbb + 0.20 * div_norm)
            success_target = torch.tensor([[success]], device=device)

            targets = {"success_labels": success_target}
            loss, _ = reasoner_criterion(output, targets)

            if not torch.isfinite(loss).all():
                raise RuntimeError("Loss du raisonneur non finie")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0, error_if_nonfinite=True)
            reasoner_optimizer.step()
            total_r_loss += loss.item()
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
    best_ckpt["reasoner_supervision"] = {
        "success": "model_derived_proxy", "synergy": False,
        "dose": False, "confidence": False,
    }
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
    print(f"   Meilleur score de sélection = {best_score:.4f}")
    print(f"   Modèle sauvegardé : {best_path}")

    # Sauvegarder l'historique
    log_path = log_dir / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"   Log : {log_path}")


if __name__ == "__main__":
    main()
