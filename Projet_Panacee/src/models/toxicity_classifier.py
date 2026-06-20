"""
Classificateur de toxicité multi-tâches – v2.

Corrections :
  1. MultiTaskBCELoss gère NaN par tâche + class weights dynamiques.
  2. Classifier head avec BatchNorm pour stabiliser le fine-tuning.
  3. Méthode unfreeze progressive (gradual unfreezing).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MolecularEncoder


class ToxicityClassifier(nn.Module):
    """
    Encodeur GNN pré-entraîné + tête de classification multi-tâches.
    """

    def __init__(
        self,
        encoder: MolecularEncoder,
        num_tasks: int = 12,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_tasks = num_tasks

        if freeze_encoder:
            self._freeze_encoder()

        # Tête de classification plus robuste
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_tasks),
        )
        self._init_classifier()

    # ──────────────────────────────────────────────────────────────────
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def gradual_unfreeze(self, epoch: int, total_freeze_epochs: int = 10):
        """
        Dégèle progressivement l'encodeur couche par couche.
        Epoch 0→freeze_epochs : tête seule.
        Epoch freeze_epochs+i : dégèle couche -(i+1).
        """
        n_layers = self.encoder.num_layers
        if epoch < total_freeze_epochs:
            self._freeze_encoder()
        else:
            layers_to_unfreeze = min(epoch - total_freeze_epochs + 1, n_layers)
            # D'abord tout geler
            self._freeze_encoder()
            # Puis dégeler les N dernières couches (les plus proches de la sortie)
            for i in range(n_layers - layers_to_unfreeze, n_layers):
                for p in self.encoder.convs[i].parameters():
                    p.requires_grad = True
                for p in self.encoder.norms[i].parameters():
                    p.requires_grad = True
            # Toujours dégeler la projection finale
            for p in self.encoder.projection.parameters():
                p.requires_grad = True
            # Au dégel complet, dégeler aussi l'embedding d'entrée et le gating
            # du pooling (sinon ces sous-modules ne s'entraînent JAMAIS).
            if layers_to_unfreeze >= n_layers:
                for p in self.encoder.atom_embedding.parameters():
                    p.requires_grad = True
                for p in self.encoder.pool_gate.parameters():
                    p.requires_grad = True

    # ──────────────────────────────────────────────────────────────────
    def forward(self, batch):
        mol_emb = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        return self.classifier(mol_emb)


# ══════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════

class MultiTaskBCELoss(nn.Module):
    """
    BCE pré-tâche avec :
      - masquage automatique des NaN (labels manquants),
      - pos_weight par tâche pour corriger l'imbalance de classes.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer(
            "_pos_weight",
            pos_weight if pos_weight is not None else None,
        )

    @property
    def pos_weight(self):
        return self._pos_weight

    @pos_weight.setter
    def pos_weight(self, value):
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float)
        self._pos_weight = value

    def forward(self, logits, targets):
        """
        Args:
            logits  : [B, T]
            targets : [B, T]  (peut contenir NaN)
        Returns:
            scalar loss
        """
        valid = ~torch.isnan(targets)          # [B, T]
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        targets_safe = targets.clone()
        targets_safe[~valid] = 0.0

        # BCE par élément
        if self._pos_weight is not None:
            pw = self._pos_weight.to(logits.device)
            loss_all = F.binary_cross_entropy_with_logits(
                logits, targets_safe, pos_weight=pw, reduction="none"
            )
        else:
            loss_all = F.binary_cross_entropy_with_logits(
                logits, targets_safe, reduction="none"
            )

        # Masquer et moyenner
        loss_all = loss_all * valid.float()
        return loss_all.sum() / valid.sum()
