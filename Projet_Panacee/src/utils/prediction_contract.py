"""Shared probability/threshold convention for training and inference."""
import math

import torch

CLASSIFICATION_TASKS = ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity")


def calibrated_logits(predictions, temperatures):
    result = dict(predictions)
    for name, temperature in temperatures.items():
        if name not in CLASSIFICATION_TASKS:
            raise ValueError(f"Température pour une tâche inconnue : {name}")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"{name}: température invalide")
        if name in result:
            result[name] = result[name] / temperature
    return result


def checkpoint_thresholds(checkpoint):
    """Migrate legacy raw-probability thresholds into calibrated space."""
    space = checkpoint.get("threshold_space", "raw_probability")
    if space not in ("raw_probability", "calibrated_probability"):
        raise ValueError(f"Espace des seuils inconnu : {space}")
    result = {}
    for name, values in checkpoint.get("optimal_thresholds", {}).items():
        thresholds = torch.as_tensor(values, dtype=torch.float64)
        if thresholds.ndim != 1 or not torch.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
            raise ValueError(f"{name}: seuils invalides")
        if space == "raw_probability":
            logits = torch.logit(thresholds)
            logits = calibrated_logits({name: logits}, {name: checkpoint.get("temperatures", {}).get(name, 1.0)})[name]
            thresholds = logits.sigmoid()
        result[name] = thresholds.tolist()
    return result
