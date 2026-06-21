# -*- coding: utf-8 -*-
"""
Quantification de l'incertitude des prédictions.

Deux sources d'incertitude ÉPISTÉMIQUE (incertitude du modèle, réductible avec
plus de données) :

  1) MC-Dropout (Gal & Ghahramani, 2016) — on garde le dropout ACTIF à
     l'inférence et on échantillonne plusieurs passes : la dispersion des
     probabilités mesure la confiance du modèle.
  2) Ensemble — on agrège les prédictions de plusieurs modèles entraînés
     indépendamment ; le désaccord mesure l'incertitude.

En sécurité médicale, une prédiction « non toxique » mais TRÈS incertaine doit
être traitée avec prudence : l'incertitude est aussi importante que la valeur.

Le cœur (résumé statistique) ne dépend que de numpy → testable partout. La partie
torch (activation du dropout) est importée paresseusement.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

# Seuils de confiance sur l'écart-type des PROBABILITÉS [0, 1]
STD_OK = 0.05    # < 5 pts : haute confiance
STD_WARN = 0.12  # < 12 pts : confiance modérée ; au-delà : faible confiance


def confidence_label(std: float) -> str:
    """Niveau de confiance (OK/WARN/DANGER) à partir de l'écart-type des probas."""
    if std < STD_OK:
        return "OK"
    if std < STD_WARN:
        return "WARN"
    return "DANGER"


def summarize(samples) -> dict:
    """
    Résume des échantillons de prédictions.

    Args:
        samples: tableau [n_samples] ou [n_samples, n_sorties].

    Returns:
        {mean, std, ci_low, ci_high, confidence} (listes si multi-sorties,
        scalaires sinon). IC à 95 % par percentiles.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        mean = float(arr.mean())
        std = float(arr.std())
        return {
            "mean": mean, "std": std,
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "confidence": confidence_label(std),
            "n_samples": int(arr.shape[0]),
        }
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "mean": mean.tolist(), "std": std.tolist(),
        "ci_low": np.percentile(arr, 2.5, axis=0).tolist(),
        "ci_high": np.percentile(arr, 97.5, axis=0).tolist(),
        "confidence": [confidence_label(s) for s in std],
        "n_samples": int(arr.shape[0]),
    }


def mc_dropout_samples(forward_fn: Callable[[], np.ndarray],
                       n_samples: int = 30) -> np.ndarray:
    """
    Échantillonne `n_samples` passes avant (dropout supposé déjà ACTIF).

    `forward_fn` doit renvoyer un tableau de probabilités à chaque appel.
    Renvoie un tableau empilé [n_samples, ...].
    """
    if n_samples < 1:
        raise ValueError("n_samples doit être ≥ 1")
    return np.stack([np.asarray(forward_fn(), dtype=float)
                     for _ in range(n_samples)], axis=0)


def ensemble_samples(forward_fns: Sequence[Callable[[], np.ndarray]]) -> np.ndarray:
    """Empile une prédiction par modèle d'un ensemble → [n_modeles, ...]."""
    fns = list(forward_fns)
    if not fns:
        raise ValueError("ensemble vide")
    return np.stack([np.asarray(fn(), dtype=float) for fn in fns], axis=0)


def enable_mc_dropout(model) -> int:
    """
    Active UNIQUEMENT les couches de dropout d'un modèle PyTorch (le reste —
    LayerNorm/BatchNorm — reste en mode évaluation, donc déterministe).

    Couvre `nn.Dropout` ET les couches qui appliquent `F.dropout(training=self.training)`
    (repérées par un attribut `dropout` numérique > 0, ex. la convolution
    d'attention de l'encodeur).

    Returns:
        Nombre de modules passés en mode dropout (0 → MC-Dropout sans effet).
    """
    import torch.nn as nn

    model.eval()
    n = 0
    for m in model.modules():
        is_dropout = isinstance(m, nn.Dropout)
        d = getattr(m, "dropout", None)
        has_fdropout = isinstance(d, (int, float)) and d > 0
        if is_dropout or has_fdropout:
            m.train()
            n += 1
    return n
