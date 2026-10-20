"""Utilitaires de reproductibilité et scheduling partagés."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, warn_only: bool = True):
    """
    Fixe toutes les seeds pour reproductibilité.

    Fix H10 (audit 2026-10-20) : ajout de PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG
    et torch.use_deterministic_algorithms — nécessaires pour reproductibilité
    bit-à-bit sur CUDA. `warn_only=True` par défaut car certaines ops GNN
    (scatter_add) n'ont pas d'implé déterministe → warning plutôt qu'erreur.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Requis par CUDA >= 10.2 pour cuBLAS déterministe
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except (RuntimeError, TypeError):
            # torch < 1.11 : signature différente ; warn_only non supporté
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def seed_worker(worker_id: int) -> None:
    """Seed per-worker du DataLoader. À passer en `worker_init_fn=seed_worker`."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WarmupCosineScheduler:
    """Warmup linéaire + cosine annealing. Utilisé par les 3 phases."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        if warmup_epochs < 0 or total_epochs <= 0:
            raise ValueError("warmup_epochs >= 0 et total_epochs > 0 requis")
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if self.warmup > 0 and epoch < self.warmup:
            alpha = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            alpha = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = self.lr_min + (blr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]
