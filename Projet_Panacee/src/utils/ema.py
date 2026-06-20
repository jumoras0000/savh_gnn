"""
Exponential Moving Average (EMA) des poids du modèle.

Maintient une copie "lissée" des paramètres (et buffers flottants comme les
stats BatchNorm). Évaluer / sauvegarder avec les poids EMA stabilise le
fine-tuning et améliore souvent la généralisation (cf. Mean Teacher, SWA).

Usage :
    ema = ModelEMA(model, decay=0.999)
    ...
    optimizer.step()
    ema.update(model)           # après chaque step
    ...
    ema.store(model)            # sauve les poids courants
    ema.copy_to(model)          # charge les poids EMA dans le modèle
    metrics = evaluate(model)   # éval avec EMA
    ema.restore(model)          # restaure les poids courants
"""
import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        # On lisse tous les tenseurs flottants du state_dict (params + buffers BN).
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def store(self, model):
        """Sauvegarde les poids courants (pour restauration après éval EMA)."""
        self._backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model):
        """Charge les poids EMA dans le modèle (in place)."""
        msd = model.state_dict()
        for k in self.shadow:
            msd[k].copy_(self.shadow[k].to(msd[k].dtype))

    @torch.no_grad()
    def restore(self, model):
        """Restaure les poids sauvegardés par store()."""
        if not self._backup:
            return
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}

    def state_dict(self):
        return self.shadow
