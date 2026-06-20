"""
metrics.py – Métriques et monitoring anti-surapprentissage
==========================================================
Métriques GNN complètes pour classification et régression moléculaire,
avec détection automatique du surapprentissage.

Classification :
  ROC-AUC, AUPRC (Average Precision), F1, Precision, Recall,
  MCC (Matthews Correlation Coefficient), Balanced Accuracy, Kappa

Régression :
  RMSE, MAE, R², Pearson r, Spearman ρ

Anti-surapprentissage :
  - Suivi train/val gap
  - Détection divergence train/val AUC
  - Early stopping avec restauration des meilleurs poids
  - Courbes d'apprentissage

Références GNN :
  - Wu et al. (2018) "MoleculeNet: A Benchmark for Molecular ML" (métriques standard)
  - Hu et al. (2020) "Open Graph Benchmark" (protocole d'évaluation)
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score,
)
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("panacee.metrics")


# ══════════════════════════════════════════════════════════════════════
#  STRUCTURE DE RÉSULTATS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationMetrics:
    roc_auc     : float = 0.0
    auprc       : float = 0.0
    f1          : float = 0.0
    precision   : float = 0.0
    recall      : float = 0.0
    mcc         : float = 0.0
    balanced_acc: float = 0.0
    kappa       : float = 0.0
    loss        : float = 0.0
    n_tasks     : int   = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"AUC={self.roc_auc:.4f} | AUPRC={self.auprc:.4f} | "
            f"F1={self.f1:.4f} | MCC={self.mcc:.4f} | "
            f"Loss={self.loss:.5f}"
        )


@dataclass
class RegressionMetrics:
    rmse    : float = 0.0
    mae     : float = 0.0
    r2      : float = 0.0
    pearson : float = 0.0
    spearman: float = 0.0
    loss    : float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"RMSE={self.rmse:.4f} | MAE={self.mae:.4f} | "
            f"R²={self.r2:.4f} | Pearson={self.pearson:.4f} | "
            f"Loss={self.loss:.5f}"
        )


# ══════════════════════════════════════════════════════════════════════
#  CALCUL DES MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════

def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: float = 0.0,
) -> ClassificationMetrics:
    """
    Calcule toutes les métriques de classification multi-tâches.
    Gère automatiquement les NaN et les tâches sans instances positives.

    Args:
        logits  : [B, T] logits bruts
        targets : [B, T] labels (peut contenir NaN)
        loss    : loss scalaire de l'epoch

    Returns:
        ClassificationMetrics (moyenne sur toutes les tâches valides)
    """
    probs  = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    valid  = ~np.isnan(y_true)

    aucs, auprcs, f1s, precs, recs, mccs, baccs, kappas = [], [], [], [], [], [], [], []
    num_tasks = y_true.shape[1] if y_true.ndim > 1 else 1

    for t in range(num_tasks):
        if y_true.ndim > 1:
            mask = valid[:, t]
            y    = y_true[mask, t]
            p    = probs[mask, t]
        else:
            mask = valid
            y    = y_true[mask]
            p    = probs[mask]

        if mask.sum() < 4:   # pas assez de données
            continue

        pred_binary = (p >= 0.5).astype(int)

        # AUC-ROC (requiert les 2 classes)
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))
            auprcs.append(average_precision_score(y, p))
        else:
            logger.debug(f"Tâche {t} : une seule classe, AUC ignoré")

        f1s.append(f1_score(y, pred_binary, zero_division=0))
        precs.append(precision_score(y, pred_binary, zero_division=0))
        recs.append(recall_score(y, pred_binary, zero_division=0))

        if len(np.unique(y)) > 1:
            mccs.append(matthews_corrcoef(y, pred_binary))
            baccs.append(balanced_accuracy_score(y, pred_binary))
            try:
                kappas.append(cohen_kappa_score(y, pred_binary))
            except Exception:
                pass

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return ClassificationMetrics(
        roc_auc      = _mean(aucs),
        auprc        = _mean(auprcs),
        f1           = _mean(f1s),
        precision    = _mean(precs),
        recall       = _mean(recs),
        mcc          = _mean(mccs),
        balanced_acc = _mean(baccs),
        kappa        = _mean(kappas),
        loss         = loss,
        n_tasks      = num_tasks,
    )


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss: float = 0.0,
) -> RegressionMetrics:
    """
    Calcule toutes les métriques de régression.
    Gère les NaN automatiquement.
    """
    pred_np = predictions.detach().cpu().numpy().flatten()
    tgt_np  = targets.detach().cpu().numpy().flatten()
    valid   = ~np.isnan(tgt_np)

    if valid.sum() < 4:
        return RegressionMetrics(loss=loss)

    p = pred_np[valid]
    t = tgt_np[valid]

    rmse     = float(np.sqrt(np.mean((p - t) ** 2)))
    mae      = float(np.mean(np.abs(p - t)))
    ss_res   = np.sum((t - p) ** 2)
    ss_tot   = np.sum((t - t.mean()) ** 2)
    r2       = float(1 - ss_res / max(ss_tot, 1e-8))

    try:
        pearson  = float(pearsonr(p, t)[0])
        spearman = float(spearmanr(p, t)[0])
    except Exception:
        pearson  = 0.0
        spearman = 0.0

    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2,
                              pearson=pearson, spearman=spearman, loss=loss)


def find_optimal_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: np.ndarray = np.arange(0.20, 0.81, 0.05),
) -> List[float]:
    """
    Cherche le seuil optimal par tâche (maximise le F1-score).

    Returns:
        list[float]  seuil optimal par tâche
    """
    probs  = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy()
    valid  = ~np.isnan(y_true)
    num_tasks = y_true.shape[1] if y_true.ndim > 1 else 1
    best_thresholds = []

    for t in range(num_tasks):
        if y_true.ndim > 1:
            mask = valid[:, t]
            y    = y_true[mask, t]
            p    = probs[mask, t]
        else:
            mask = valid
            y    = y_true[mask]
            p    = probs[mask]

        if mask.sum() == 0:
            best_thresholds.append(0.5)
            continue

        best_th, best_f1 = 0.5, 0.0
        for th in thresholds:
            f1 = f1_score(y, (p >= th).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)

        best_thresholds.append(best_th)

    return best_thresholds


# ══════════════════════════════════════════════════════════════════════
#  EARLY STOPPING (avec restauration des meilleurs poids)
# ══════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Early stopping avec restauration automatique des meilleurs poids.

    Surveille une métrique et arrête si elle ne s'améliore pas
    pendant `patience` epochs consécutives.

    Anti-surapprentissage : sauvegarde les poids du meilleur epoch
    et les restaure à la fin de l'entraînement.
    """

    def __init__(
        self,
        patience: int   = 20,
        min_delta: float = 1e-4,
        mode: str        = "max",   # 'max' pour AUC, 'min' pour loss
        checkpoint_path: str = "best_model.pth",
    ):
        assert mode in ("max", "min")
        self.patience         = patience
        self.min_delta        = min_delta
        self.mode             = mode
        self.checkpoint_path  = checkpoint_path
        self.best_score       = -float("inf") if mode == "max" else float("inf")
        self.counter          = 0
        self.best_epoch       = 0
        self.triggered        = False

    def _is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Appeler à chaque epoch avec la métrique de validation.

        Returns:
            True si l'entraînement doit s'arrêter.
        """
        if self._is_better(score):
            self.best_score = score
            self.counter    = 0
            self.best_epoch = epoch
            # Sauvegarder les meilleurs poids
            os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(
                f"  [EarlyStopping] Nouveau meilleur à epoch {epoch}: {score:.5f} → sauvegardé"
            )
        else:
            self.counter += 1
            logger.debug(
                f"  [EarlyStopping] Patience {self.counter}/{self.patience} "
                f"(best={self.best_score:.5f} @ epoch {self.best_epoch})"
            )

        if self.counter >= self.patience:
            self.triggered = True
            logger.info(
                f"  [EarlyStopping] Déclenchement à epoch {epoch}. "
                f"Meilleur epoch : {self.best_epoch} ({self.best_score:.5f})"
            )
            return True

        return False

    def load_best(self, model: nn.Module):
        """Restaure les meilleurs poids enregistrés."""
        if os.path.exists(self.checkpoint_path):
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
            logger.info(
                f"  [EarlyStopping] Meilleurs poids restaurés "
                f"(epoch {self.best_epoch}, score {self.best_score:.5f})"
            )


# ══════════════════════════════════════════════════════════════════════
#  MONITORING ANTI-SURAPPRENTISSAGE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingHistory:
    """Historique complet d'entraînement pour analyse du surapprentissage."""
    train_losses  : List[float] = field(default_factory=list)
    val_losses    : List[float] = field(default_factory=list)
    train_aucs    : List[float] = field(default_factory=list)
    val_aucs      : List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epochs        : List[int]   = field(default_factory=list)

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_auc: float = 0.0,
        val_auc: float   = 0.0,
        lr: float        = 0.0,
    ):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_aucs.append(train_auc)
        self.val_aucs.append(val_auc)
        self.learning_rates.append(lr)

    # ─────────────────────────────────────────────────────────────────
    def check_overfitting(self, window: int = 5) -> dict:
        """
        Détecte des signes de surapprentissage :
          1. Gap train/val AUC > 0.10 sur les `window` dernières epochs
          2. Val AUC en baisse tandis que Train AUC monte
          3. Val loss en hausse tandis que Train loss baisse

        Returns:
            dict {indicator: bool, severity: str}
        """
        n = len(self.epochs)
        if n < window:
            return {"overfitting_detected": False, "severity": "none"}

        t_auc  = np.array(self.train_aucs[-window:])
        v_auc  = np.array(self.val_aucs[-window:])
        t_loss = np.array(self.train_losses[-window:])
        v_loss = np.array(self.val_losses[-window:])

        gap         = float(np.mean(t_auc - v_auc))
        val_trend   = float(np.polyfit(range(window), v_auc, 1)[0])   # pente val AUC
        train_trend = float(np.polyfit(range(window), t_auc, 1)[0])   # pente train AUC
        loss_div    = float(np.mean(v_loss - t_loss))

        # Indicateurs
        gap_high   = gap > 0.10
        diverging  = (val_trend < -0.005) and (train_trend > 0.0)
        loss_div_h = loss_div > 0.20

        severity = "none"
        if gap_high or diverging:
            severity = "moderate"
        if gap_high and (diverging or loss_div_h):
            severity = "severe"

        return {
            "overfitting_detected": gap_high or diverging,
            "auc_gap"             : round(gap, 4),
            "val_auc_trend"       : round(val_trend, 5),
            "train_auc_trend"     : round(train_trend, 5),
            "loss_divergence"     : round(loss_div, 4),
            "severity"            : severity,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "epochs"       : self.epochs,
                "train_losses" : self.train_losses,
                "val_losses"   : self.val_losses,
                "train_aucs"   : self.train_aucs,
                "val_aucs"     : self.val_aucs,
                "learning_rates": self.learning_rates,
            }, f, indent=2)

    def plot(self, save_path: str = None, show: bool = False):
        """Affiche les courbes d'apprentissage."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib non disponible — graphiques désactivés")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.epochs, self.train_losses, label="Train", linewidth=2)
        axes[0].plot(self.epochs, self.val_losses,   label="Val",   linewidth=2, linestyle="--")
        axes[0].set_title("Loss d'entraînement")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # AUC
        axes[1].plot(self.epochs, self.train_aucs, label="Train AUC", linewidth=2)
        axes[1].plot(self.epochs, self.val_aucs,   label="Val AUC",   linewidth=2, linestyle="--")
        axes[1].set_title("ROC-AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
