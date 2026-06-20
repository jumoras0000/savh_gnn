"""
losses.py – Fonctions de perte
==============================
  • MultiTaskBCELoss        – BCE multi-tâches avec gestion NaN + pos_weight
  • MultiPropertyLoss       – Perte Phase 3 (BCE + MSE par groupe)
  • ReasonerLoss            – Perte pour le Raisonneur moléculaire
  • FocalLoss               – Focal loss (Lin 2017) pour données très déséquilibrées
  • LabelSmoothingBCELoss   – BCE avec lissage de labels (anti-surconfiance)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
#  1. MultiTaskBCELoss
# ══════════════════════════════════════════════════════════════════════

class MultiTaskBCELoss(nn.Module):
    """
    BCE multi-tâches avec :
      - masquage automatique des labels NaN
      - pos_weight par tâche (corrige le déséquilibre de classes)
      - lissage de labels optionnel

    Usage :
        criterion = MultiTaskBCELoss(pos_weight=dataset.get_pos_weight())
        loss = criterion(logits, targets)
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.register_buffer("_pw", pos_weight)
        else:
            self._pw = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, T]
        targets : [B, T]  (peut contenir NaN pour labels manquants)
        """
        valid = ~torch.isnan(targets)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Remplacer NaN par 0 avant le calcul (masqué ensuite)
        t_safe = targets.clone()
        t_safe[~valid] = 0.0

        # Lissage de labels (anti-surapprentissage)
        if self.label_smoothing > 0:
            t_safe = t_safe * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Calcul BCE élément par élément
        pw = self._pw.to(logits.device) if self._pw is not None else None
        loss_all = F.binary_cross_entropy_with_logits(
            logits, t_safe,
            pos_weight=pw,
            reduction="none",
        )   # [B, T]

        # Masquer les NaN et moyenner
        loss_valid = loss_all[valid]
        return loss_valid.mean()


# ══════════════════════════════════════════════════════════════════════
#  2. FocalLoss (Lin et al. 2017 – RetinaNet)
# ══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss pour classification très déséquilibrée.
    FL(p) = -α · (1-p_t)^γ · log(p_t)

    Référence : Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV.

    γ=2, α=0.25 sont les valeurs recommandées pour données déséquilibrées.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(targets)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_v  = logits[valid]
        targets_v = targets.clone()[valid]

        bce  = F.binary_cross_entropy_with_logits(logits_v, targets_v, reduction="none")
        probs = torch.sigmoid(logits_v)
        pt   = torch.where(targets_v == 1, probs, 1 - probs)
        fl   = self.alpha * (1 - pt) ** self.gamma * bce
        return fl.mean()


# ══════════════════════════════════════════════════════════════════════
#  3. LabelSmoothingBCELoss
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingBCELoss(nn.Module):
    """
    BCE avec lissage de labels : empêche le modèle de devenir trop confiant.
    y_smooth = y * (1 - ε) + ε/2
    """

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_s = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_s, reduction="mean")


# ══════════════════════════════════════════════════════════════════════
#  4. MultiPropertyLoss (Phase 3)
# ══════════════════════════════════════════════════════════════════════

# Propriétés de classification (BCE) vs régression (MSE)
_CLASSIFICATION_PROPS = {"toxicity", "efficacy", "bioavailability", "metabolic_stability"}
_REGRESSION_PROPS     = {"solubility", "lipophilicity"}


class MultiPropertyLoss(nn.Module):
    """
    Loss combinée pour les 6 groupes de propriétés Phase 3.

    Chaque groupe a :
      - un type de loss (BCE ou MSE)
      - un poids configurable
      - un masquage automatique des NaN

    Poids par défaut équilibrés pour éviter que la toxicité
    (12 tâches) domine la loss totale.
    """

    _DEFAULT_WEIGHTS = {
        "toxicity"          : 0.30,
        "efficacy"          : 0.15,
        "solubility"        : 0.15,
        "lipophilicity"     : 0.15,
        "bioavailability"   : 0.15,
        "metabolic_stability": 0.10,
    }

    def __init__(
        self,
        weights: Optional[dict] = None,
        pos_weights: Optional[dict] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.weights         = weights or self._DEFAULT_WEIGHTS
        self.label_smoothing = label_smoothing

        # Poids de classe pour déséquilibre
        self.pos_weights: dict = pos_weights or {}

    def forward(self, predictions: dict, targets: dict) -> tuple:
        """
        Returns:
            total_loss  : scalar
            loss_dict   : dict {prop: scalar} pour monitoring
        """
        total   = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        details = {}

        for prop, pred in predictions.items():
            if prop not in targets:
                continue

            tgt   = targets[prop].to(pred.device)
            valid = ~torch.isnan(tgt)
            if valid.sum() == 0:
                continue

            if prop in _REGRESSION_PROPS:
                # MSE sur les valeurs valides (régression)
                p_v = pred[valid[:, 0] if valid.ndim > 1 else valid].squeeze(-1)
                t_v = tgt[valid[:, 0] if valid.ndim > 1 else valid].squeeze(-1)
                l   = F.mse_loss(p_v, t_v)

            elif prop in _CLASSIFICATION_PROPS:
                # BCE avec gestion NaN + lissage de labels
                t_safe = tgt.clone()
                t_safe[~valid] = 0.0
                if self.label_smoothing > 0:
                    t_safe[valid] = t_safe[valid] * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

                pw = self.pos_weights.get(prop)
                if pw is not None:
                    pw = pw.to(pred.device)

                loss_all = F.binary_cross_entropy_with_logits(
                    pred, t_safe, pos_weight=pw, reduction="none"
                )
                l = loss_all[valid].mean()
            else:
                continue

            w          = self.weights.get(prop, 1.0)
            total      = total + w * l
            details[prop] = l.item()

        return total, details


# ══════════════════════════════════════════════════════════════════════
#  5. ReasonerLoss (Phase 3 – module Raisonneur)
# ══════════════════════════════════════════════════════════════════════

class ReasonerLoss(nn.Module):
    """
    Perte pour entraîner le MolecularReasoner.

    Composantes :
      - combo_loss    : score de combinaison (BCE si labels disponibles)
      - dose_entropy  : entropie minimale sur les doses (encourage la décision)
      - confidence    : calibration de la confiance
    """

    def __init__(
        self,
        combo_weight: float = 1.0,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.combo_w   = combo_weight
        self.entropy_w = entropy_weight

    def forward(
        self,
        outputs: dict,
        combo_labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        device = outputs["combo_score"].device
        total  = torch.tensor(0.0, device=device)
        details = {}

        # Score de combinaison
        if combo_labels is not None:
            combo_loss = F.binary_cross_entropy(
                outputs["combo_score"].squeeze(-1),
                combo_labels.float().to(device),
            )
            total  = total + self.combo_w * combo_loss
            details["combo"] = combo_loss.item()

        # Pénalité d'entropie sur les doses (encourage des distributions + piquées)
        doses   = outputs["doses"]                            # [B, N, D]
        entropy = -(doses * torch.log(doses + 1e-8)).sum(-1)  # [B, N]
        ent_loss = entropy.mean()
        total    = total + self.entropy_w * ent_loss
        details["dose_entropy"] = ent_loss.item()

        return total, details
