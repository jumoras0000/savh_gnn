"""
Améliorations modèle pour fixer les 7 problèmes diagnostiqués :

Problème observé → Technique appliquée :
  1. Class imbalance HIV 2.2% → Focal Loss (Lin et al. 2017) + Weighted Sampler
  2. 48% OOD rate → SMILES augmentation (Bjerrum 2017, Kimber 2021)
  3. ROC-AUC 0.66 sous SOTA → Test-Time Augmentation (TTA)
  4. 66% ambiguous conformal → Snapshot Ensemble (Huang 2017) via cosine restart
  5. Overfit val/test → Label smoothing + Mixup embedding
  6. DrugBank FAIL 44% → Pretrain multi-virus avant Phase 3
  7. Variance non quantifiée → N-seed ensembling natif

Modules réutilisables importés dans run_phase3.py et finetune_hiv_chembl.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS — class imbalance HIV 2.2% (Lin et al. 2017)
# ═══════════════════════════════════════════════════════════════════════

class FocalBCELoss(nn.Module):
    """
    Binary Focal Loss pour class imbalance sévère.

    FL(p) = -alpha_t * (1 - p_t)^gamma * log(p_t)
      p_t = p si y=1 sinon 1-p
      alpha_t = alpha si y=1 sinon 1-alpha

    Hyperparams recommandés HIV (2.2% actifs) :
      - alpha = 0.75 (poids élevé sur actifs minoritaires)
      - gamma = 2.0 (focalisation sur exemples durs)

    Référence : Lin et al. 2017 "Focal Loss for Dense Object Detection"
    arXiv:1708.02002, utilisé en virtual screening (Walters 2021).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("alpha ∈ (0, 1)")
        if gamma < 0:
            raise ValueError("gamma >= 0")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask: Optional[torch.Tensor] = None):
        """
        logits : [B] ou [B, 1]
        targets : [B] ou [B, 1] float 0/1
        mask : [B] bool, True pour labels valides (NaN masking)
        """
        logits = logits.squeeze(-1) if logits.ndim > 1 else logits
        targets = targets.squeeze(-1).float() if targets.ndim > 1 else targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce

        if mask is not None:
            loss = loss * mask.float()
            if self.reduction == "mean":
                n = max(1, int(mask.sum().item()))
                return loss.sum() / n
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════
# 2. SMILES AUGMENTATION — anti-overfit, Bjerrum 2017
# ═══════════════════════════════════════════════════════════════════════

def randomize_smiles(smiles: str, n_variants: int = 5, seed: int = 42) -> list[str]:
    """
    Génère N SMILES randomisés via RDKit RenumAtoms (invariants de molécule).

    Chaque variante représente la MÊME molécule mais avec un ordre d'atomes
    différent → augmente la diversité des graphes vus par le GNN.

    Référence : Bjerrum 2017 arXiv:1703.07076, Kimber 2021 chem-rxiv.
    Gain typique : +0.5-2% AUC sur petits datasets.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return [smiles] * n_variants

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [smiles] * n_variants

    rng = np.random.default_rng(seed)
    variants = []
    n_atoms = mol.GetNumAtoms()
    for _ in range(n_variants):
        new_order = list(range(n_atoms))
        rng.shuffle(new_order)
        try:
            renumbered = Chem.RenumberAtoms(mol, new_order)
            variant = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=True)
            variants.append(variant)
        except Exception:
            variants.append(smiles)
    return variants


# ═══════════════════════════════════════════════════════════════════════
# 3. TEST-TIME AUGMENTATION — gain gratuit en inference
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tta_predict(model, smiles: str, graph_builder_fn, device,
                n_augmentations: int = 10, head_key: str = "efficacy") -> dict:
    """
    Test-Time Augmentation : prédit sur N randomisations SMILES puis moyenne.

    Réduit la variance de prédiction et améliore la robustesse.
    Coût : N forward passes par molécule, mais parallélisable en batch.
    """
    from torch_geometric.data import Batch

    variants = randomize_smiles(smiles, n_variants=n_augmentations)
    graphs = []
    for v in variants:
        try:
            g = graph_builder_fn(v)
            if g is not None:
                graphs.append(g)
        except Exception:
            continue
    if not graphs:
        return {"mean": None, "std": None, "n_valid": 0}

    batch = Batch.from_data_list(graphs).to(device)
    model.eval()
    preds = model(batch)
    scores = preds[head_key].squeeze(-1).cpu().numpy()
    if head_key in ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"):
        scores = 1.0 / (1.0 + np.exp(-scores))  # sigmoid

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "n_valid": len(scores),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. WEIGHTED SAMPLER pour oversampling classes minoritaires
# ═══════════════════════════════════════════════════════════════════════

def make_weighted_sampler(labels, oversample_ratio: float = 2.0) -> torch.utils.data.WeightedRandomSampler:
    """
    Oversampling des actifs HIV (classe minoritaire 2.2%).

    oversample_ratio : facteur sur la fréquence inverse (1.0 = pondération exacte,
    2.0 = double la proba d'un actif).

    Attention : à utiliser avec drop_last=False et pos_weight=1.0 (sinon double-counting).
    """
    labels_arr = np.asarray(labels)
    valid_mask = ~np.isnan(labels_arr)
    pos_count = int((labels_arr[valid_mask] == 1).sum())
    neg_count = int((labels_arr[valid_mask] == 0).sum())
    if pos_count == 0 or neg_count == 0:
        # Fallback : poids uniformes
        return torch.utils.data.WeightedRandomSampler(
            weights=torch.ones(len(labels_arr)), num_samples=len(labels_arr),
            replacement=True,
        )
    inv_pos = oversample_ratio * (1.0 / pos_count)
    inv_neg = 1.0 / neg_count
    weights = np.where(labels_arr == 1, inv_pos, inv_neg)
    weights[~valid_mask] = inv_neg  # NaN → poids inactif
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(labels_arr),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. SNAPSHOT ENSEMBLE — Huang et al. 2017 (free ensemble via cosine restart)
# ═══════════════════════════════════════════════════════════════════════

class CosineRestartScheduler:
    """
    Cosine Annealing with Warm Restarts → produit N "snapshots" du modèle
    à chaque minimum du cycle. Chaque snapshot est un modèle différent qu'on
    peut ensembler sans coût training supplémentaire.

    Référence : Huang et al. 2017 arXiv:1704.00109 "Snapshot Ensembles".
    """

    def __init__(self, optimizer, cycle_epochs: int, n_cycles: int,
                 lr_max: float, lr_min: float = 1e-6):
        self.optimizer = optimizer
        self.cycle_epochs = cycle_epochs
        self.n_cycles = n_cycles
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_epochs = cycle_epochs * n_cycles
        self._snapshot_epochs = [cycle_epochs * (i + 1) - 1
                                  for i in range(n_cycles)]

    def step(self, epoch: int):
        cycle_pos = epoch % self.cycle_epochs
        progress = cycle_pos / self.cycle_epochs
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
             (1.0 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def is_snapshot_epoch(self, epoch: int) -> bool:
        """True quand on doit sauvegarder un snapshot."""
        return epoch in self._snapshot_epochs


# ═══════════════════════════════════════════════════════════════════════
# 6. LABEL SMOOTHING pour BCE
# ═══════════════════════════════════════════════════════════════════════

class BCEWithLabelSmoothing(nn.Module):
    """
    BCE avec label smoothing — réduit l'overconfidence et améliore calibration.

    Pour label y ∈ {0, 1} : y_smooth = y * (1 - 2*eps) + eps
    → labels 0 deviennent eps, labels 1 deviennent 1-eps.

    eps typique : 0.05-0.1. Référence : Szegedy et al. 2016 (CVPR).
    """

    def __init__(self, epsilon: float = 0.05,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if not 0 <= epsilon < 0.5:
            raise ValueError("epsilon ∈ [0, 0.5)")
        self.eps = epsilon
        self.pos_weight = pos_weight

    def forward(self, logits, targets, mask: Optional[torch.Tensor] = None):
        y_smooth = targets.float() * (1 - 2 * self.eps) + self.eps
        loss = F.binary_cross_entropy_with_logits(
            logits, y_smooth,
            pos_weight=self.pos_weight, reduction="none",
        )
        if mask is not None:
            loss = loss * mask.float()
            if int(mask.sum().item()) > 0:
                return loss.sum() / int(mask.sum().item())
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════
# 7. MIXUP embedding — Zhang et al. 2017
# ═══════════════════════════════════════════════════════════════════════

def mixup_embeddings(embeddings, labels, alpha: float = 0.2):
    """
    Mixup au niveau des embeddings moléculaires (post-encoder).
    Génère un batch mixé : (λ x_i + (1-λ) x_j, λ y_i + (1-λ) y_j).

    Référence : Zhang et al. 2017 arXiv:1710.09412.
    """
    if alpha <= 0:
        return embeddings, labels, 1.0

    batch_size = embeddings.size(0)
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # garantit lam >= 0.5 (préserve majorité)

    idx = torch.randperm(batch_size, device=embeddings.device)
    mixed_emb = lam * embeddings + (1 - lam) * embeddings[idx]
    mixed_lbl = lam * labels + (1 - lam) * labels[idx]
    return mixed_emb, mixed_lbl, lam
