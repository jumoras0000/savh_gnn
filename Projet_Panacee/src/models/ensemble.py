"""
Deep Ensembles pour estimation d'incertitude robuste.

Littérature : Lakshminarayanan et al. 2017, confirmé sur GNN moléculaires
(OGB benchmarks) comme gold standard > MC Dropout > Evidential.

Usage :
    ensemble = DeepEnsemble.from_checkpoints(
        paths=["ckpt_seed0.pth", "ckpt_seed1.pth", ...],
        model_factory=lambda: MultiPropertyPredictor(encoder, hidden_dim=256),
    )
    mean_preds, std_preds = ensemble.predict(batch)
"""
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional


class DeepEnsemble(nn.Module):
    """Ensemble de N modèles indépendants → mean + std par prédiction."""

    def __init__(self, models: List[nn.Module]):
        super().__init__()
        if len(models) < 2:
            raise ValueError("Ensemble nécessite >= 2 modèles")
        self.models = nn.ModuleList(models)

    @classmethod
    def from_checkpoints(
        cls,
        paths: List[str],
        model_factory: Callable[[], nn.Module],
        device: str = "cpu",
        key: str = "model_state_dict",
    ) -> "DeepEnsemble":
        models = []
        for p in paths:
            m = model_factory()
            from src.utils.safe_load import safe_load_checkpoint
            ckpt = safe_load_checkpoint(p)
            m.load_state_dict(ckpt[key], strict=True)
            m.to(device)
            m.eval()
            models.append(m)
        return cls(models)

    # Têtes de classification (seuils "low/medium/high" calibrés sur probabilités)
    _CLASSIF_HEADS = frozenset({
        "toxicity", "efficacy", "bbb_penetration", "clinical_toxicity",
    })

    @torch.no_grad()
    def predict(
        self, batch, transform=None, apply_sigmoid_to_classif: bool = True,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Retourne (mean_preds, std_preds) par task group.

        Fix B4 (audit 2026-10-20) : les têtes de classification sortent des
        LOGITS bruts. Comparer leur std à des seuils calibrés pour PROBABILITÉS
        (0.05/0.15 dans predict_with_disagreement) donne un désaccord illisible.
        Par défaut, applique sigmoid aux têtes classification AVANT stacking
        pour que la std soit homogène sur l'échelle [0, 1].
        Les têtes régression (solubility, lipophilicity) restent sur leur
        échelle normalisée (z-score pendant training, dénormalisée hors ensemble).
        """
        all_preds: Dict[str, List[torch.Tensor]] = {}

        for model in self.models:
            model.eval()
            preds = model(batch)
            if transform is not None:
                preds = transform(preds)
            if apply_sigmoid_to_classif:
                preds = {
                    k: (torch.sigmoid(v) if k in self._CLASSIF_HEADS else v)
                    for k, v in preds.items()
                }
            if any(not torch.isfinite(value).all() for value in preds.values()):
                raise ValueError("Prédictions non finies dans l'ensemble")
            if all_preds and set(preds) != set(all_preds):
                raise ValueError("Les modèles de l'ensemble ne prédisent pas les mêmes tâches")
            for k, v in preds.items():
                all_preds.setdefault(k, []).append(v.float())

        means = {}
        stds = {}
        for k, tensors in all_preds.items():
            stacked = torch.stack(tensors, dim=0)  # [N_models, B, D]
            means[k] = stacked.mean(dim=0)
            stds[k] = stacked.std(dim=0, unbiased=True)  # ddof=1 (fix H6)

        return means, stds

    @torch.no_grad()
    def predict_with_disagreement(self, batch) -> Dict:
        """
        Prédiction + métrique de désaccord inter-modèles.

        Les seuils 0.05 / 0.15 sont calibrés pour la variance SUR PROBABILITÉS
        des têtes classification (fix B4) — n'inclut PAS les têtes régression
        dont l'échelle n'est pas comparable.
        """
        means, stds = self.predict(batch, apply_sigmoid_to_classif=True)

        classif_stds = [s.mean() for k, s in stds.items() if k in self._CLASSIF_HEADS]
        if classif_stds:
            avg_std = torch.stack(classif_stds).mean().item()
        else:
            avg_std = torch.stack([s.mean() for s in stds.values()]).mean().item()
        disagreement = "low" if avg_std < 0.05 else "medium" if avg_std < 0.15 else "high"

        return {
            "predictions": means,
            "uncertainties": stds,
            "avg_disagreement": avg_std,
            "confidence": disagreement,
            "n_models": len(self.models),
            "note": "std des têtes classif sur [0,1] (après sigmoid); régression sur échelle normalisée",
        }
