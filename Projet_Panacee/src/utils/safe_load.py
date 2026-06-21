# -*- coding: utf-8 -*-
"""
Chargement SÉCURISÉ de checkpoints PyTorch.

Pourquoi ? `torch.load(..., weights_only=False)` désérialise du pickle arbitraire :
un fichier .pth malveillant (importé via l'UI) peut exécuter du code à l'ouverture.
On charge donc par défaut en mode `weights_only=True` (seuls tenseurs + types de
base sont autorisés), après avoir mis en liste blanche les types numpy courants
que produisent nos métriques — sinon nos propres checkpoints ne se chargeraient pas.

- Fichier de CONFIANCE (produit localement par le projet) : si le mode sécurisé
  échoue (ancien format picklé), on retombe sur le mode permissif AVEC un warning.
- Fichier NON DE CONFIANCE (sous un dossier `uploads/`) : on REFUSE le mode
  permissif. Un .pth importé doit être chargeable sans exécuter de pickle.
"""
from __future__ import annotations

import warnings
from pathlib import Path

_SAFE_GLOBALS_REGISTERED = False


def _register_numpy_safe_globals() -> None:
    """Autorise les reconstructeurs numpy sous weights_only=True (idempotent)."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    try:
        import importlib

        import numpy as np
        import torch.serialization as ts

        allow = []
        # `numpy.core` (ancien) ou `numpy._core` (numpy ≥ 2) selon la version
        for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
            try:
                m = importlib.import_module(modname)
            except Exception:
                continue
            for name in ("scalar", "_reconstruct"):
                obj = getattr(m, name, None)
                if obj is not None:
                    allow.append(obj)
        # ndarray, dtype et scalaires numpy (np.float64, np.int64, …) via sklearn
        for name in ("ndarray", "dtype", "float64", "float32",
                     "int64", "int32", "bool_"):
            obj = getattr(np, name, None)
            if obj is not None:
                allow.append(obj)
        # classes de dtype numpy ≥ 2 (numpy.dtypes.Float64DType, Int64DType, …)
        try:
            import numpy.dtypes as npd
            for name in dir(npd):
                if name.endswith("DType"):
                    obj = getattr(npd, name, None)
                    if isinstance(obj, type):
                        allow.append(obj)
        except Exception:
            pass
        if allow:
            ts.add_safe_globals(allow)
    except Exception:
        pass  # numpy absent ou API torch trop ancienne : on tentera quand même
    _SAFE_GLOBALS_REGISTERED = True


def _is_trusted(path: str | Path) -> bool:
    """Un checkpoint est « de confiance » s'il ne vient pas d'un dossier d'upload."""
    parts = {p.lower() for p in Path(path).parts}
    return "uploads" not in parts


def safe_load_checkpoint(path: str | Path, map_location: str = "cpu",
                         trusted: bool | None = None):
    """
    Charge un checkpoint de façon sûre.

    Args:
        path:         chemin du .pth
        map_location: cible torch (cpu par défaut)
        trusted:      force le statut de confiance ; sinon déduit du chemin
                      (un fichier sous `uploads/` est considéré non sûr).

    Raises:
        ValueError si un fichier non sûr ne peut être chargé en mode sécurisé.
    """
    import torch

    _register_numpy_safe_globals()
    if trusted is None:
        trusted = _is_trusted(path)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e_secure:
        if not trusted:
            raise ValueError(
                "Checkpoint importé refusé : impossible de le charger en mode "
                "sécurisé (weights_only). Un .pth de source externe ne doit pas "
                "nécessiter la désérialisation d'objets arbitraires. "
                f"Détail : {e_secure}"
            ) from e_secure
        warnings.warn(
            f"Chargement permissif (weights_only=False) d'un checkpoint de "
            f"confiance : {path}. Mode sécurisé indisponible ({e_secure}).",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)
