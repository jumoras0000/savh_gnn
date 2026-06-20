"""
Phase 3 - Tête multi-propriétés étendue.

Prédit simultanément :
  - Toxicité (12 tâches Tox21)
  - Efficacité (score prédit)
  - Solubilité (LogS)
  - Lipophilicité (LogP)
  - Biodisponibilité orale
  - Stabilité métabolique
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MolecularEncoder


class PropertyHead(nn.Module):
    """Tête de prédiction pour un groupe de propriétés."""

    def __init__(self, input_dim: int, num_tasks: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_tasks),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


class MultiPropertyPredictor(nn.Module):
    """
    Modèle multi-propriétés : encodeur GNN + N têtes spécialisées.

    Chaque tête prédit un groupe de propriétés :
      - toxicity_head  → 12 tâches (classification, sigmoid)
      - efficacy_head  → 1 score (régression ou classification)
      - solubility_head → 1 (régression)
      - lipophilicity_head → 1 (régression)
      - bioavailability_head → 1 (classification)
      - metabolic_stability_head → 1 (classification)
    """

    def __init__(
        self,
        encoder: MolecularEncoder,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        if freeze_encoder:
            self._freeze_encoder()

        # Couche partagée pour enrichir la représentation
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Têtes spécialisées
        self.toxicity_head = PropertyHead(hidden_dim, 12, dropout)
        self.efficacy_head = PropertyHead(hidden_dim, 1, dropout)
        self.solubility_head = PropertyHead(hidden_dim, 1, dropout)
        self.lipophilicity_head = PropertyHead(hidden_dim, 1, dropout)
        self.bioavailability_head = PropertyHead(hidden_dim, 1, dropout)
        self.metabolic_stability_head = PropertyHead(hidden_dim, 1, dropout)

        self.head_names = [
            "toxicity", "efficacy", "solubility",
            "lipophilicity", "bioavailability", "metabolic_stability",
        ]

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def gradual_unfreeze(self, epoch: int, total_freeze_epochs: int = 5):
        n_layers = self.encoder.num_layers
        if epoch < total_freeze_epochs:
            self._freeze_encoder()
        else:
            layers_to_unfreeze = min(epoch - total_freeze_epochs + 1, n_layers)
            self._freeze_encoder()
            for i in range(n_layers - layers_to_unfreeze, n_layers):
                for p in self.encoder.convs[i].parameters():
                    p.requires_grad = True
                for p in self.encoder.norms[i].parameters():
                    p.requires_grad = True
            for p in self.encoder.projection.parameters():
                p.requires_grad = True

    def encode(self, batch):
        """Retourne l'embedding moléculaire."""
        mol_emb = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        return self.shared_layer(mol_emb)

    def forward(self, batch):
        """Retourne un dict de prédictions par propriété."""
        emb = self.encode(batch)
        return {
            "toxicity": self.toxicity_head(emb),         # [B, 12]
            "efficacy": self.efficacy_head(emb),          # [B, 1]
            "solubility": self.solubility_head(emb),      # [B, 1]
            "lipophilicity": self.lipophilicity_head(emb),# [B, 1]
            "bioavailability": self.bioavailability_head(emb),    # [B, 1]
            "metabolic_stability": self.metabolic_stability_head(emb),  # [B, 1]
        }


class MultiPropertyLoss(nn.Module):
    """
    Loss combinée multi-propriétés.

    - Classification (toxicité, biodisponibilité, stabilité) : BCE
    - Régression (efficacité, solubilité, lipophilicité) : MSE/Huber
    - Masquage NaN pour labels manquants
    """

    def __init__(self, tox_pos_weight=None, task_weights=None):
        super().__init__()
        self.register_buffer("tox_pos_weight", tox_pos_weight)

        # Poids relatif de chaque groupe de propriétés
        default_weights = {
            "toxicity": 1.0,
            "efficacy": 2.0,       # Plus important
            "solubility": 0.5,
            "lipophilicity": 0.5,
            "bioavailability": 0.8,
            "metabolic_stability": 0.8,
        }
        self.task_weights = task_weights or default_weights

    def forward(self, predictions, targets):
        """
        predictions: dict {name: tensor}
        targets: dict {name: tensor} (peut contenir des NaN)
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_details = {}

        for name, pred in predictions.items():
            if name not in targets:
                continue

            target = targets[name]
            weight = self.task_weights.get(name, 1.0)

            # Aligner les dimensions defensivement (robustesse si data != tete)
            if pred.shape != target.shape and pred.dim() == target.dim() == 2:
                m = min(pred.shape[1], target.shape[1])
                pred = pred[:, :m]
                target = target[:, :m]

            # Masquer les NaN SANS aplatir (preserve le broadcast de pos_weight)
            valid_mask = ~torch.isnan(target)
            if valid_mask.sum() == 0:
                continue
            target_safe = torch.where(valid_mask, target, torch.zeros_like(target))

            if name in ("toxicity", "bioavailability", "metabolic_stability"):
                # Classification : BCE par element (reduction='none' -> masque ensuite)
                if name == "toxicity" and self.tox_pos_weight is not None:
                    per = F.binary_cross_entropy_with_logits(
                        pred, target_safe, pos_weight=self.tox_pos_weight, reduction="none",
                    )
                else:
                    per = F.binary_cross_entropy_with_logits(
                        pred, target_safe, reduction="none",
                    )
            else:
                # Régression : Huber loss (robuste aux outliers)
                per = F.huber_loss(pred, target_safe, delta=1.0, reduction="none")

            loss = (per * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

            loss_details[name] = loss.item()
            total_loss = total_loss + weight * loss

        loss_details["total"] = total_loss.item()
        return total_loss, loss_details
