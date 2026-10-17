"""
Phase 3 - Tête multi-propriétés étendue.

Prédit simultanément :
  - Toxicité (12 tâches Tox21)
  - Efficacité anti-VIH (HIV active, classification)
  - Solubilité aqueuse (LogS, régression)
  - Lipophilicité (LogP, régression)
  - Perméabilité BBB (Blood-Brain Barrier, classification)
  - Toxicité clinique (ClinTox CT_TOX, classification)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder import MolecularEncoder
except ModuleNotFoundError as exc:  # la loss reste testable sans PyG
    if exc.name != "torch_geometric":
        raise
    MolecularEncoder = None


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
      - toxicity_head  → 12 tâches Tox21 (classification)
      - efficacy_head  → 1 HIV active (classification)
      - solubility_head → 1 LogS (régression)
      - lipophilicity_head → 1 LogP (régression)
      - bbb_penetration_head → 1 BBB perméabilité (classification)
      - clinical_toxicity_head → 1 ClinTox CT_TOX (classification)
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
        self.bbb_penetration_head = PropertyHead(hidden_dim, 1, dropout)
        self.clinical_toxicity_head = PropertyHead(hidden_dim, 1, dropout)

        # Backward compat : anciens checkpoints ont bioavailability_head / metabolic_stability_head
        self.bioavailability_head = self.bbb_penetration_head
        self.metabolic_stability_head = self.clinical_toxicity_head

        self.head_names = [
            "toxicity", "efficacy", "solubility",
            "lipophilicity", "bbb_penetration", "clinical_toxicity",
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
            # Au dégel complet, dégeler aussi l'embedding d'entrée et le gating
            # du pooling (sinon ces sous-modules ne s'entraînent JAMAIS).
            if layers_to_unfreeze >= n_layers:
                for p in self.encoder.atom_embedding.parameters():
                    p.requires_grad = True
                for p in self.encoder.pool_gate.parameters():
                    p.requires_grad = True
                if hasattr(self.encoder, "virtual_node"):
                    for p in self.encoder.virtual_node.parameters():
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
            "bbb_penetration": self.bbb_penetration_head(emb),        # [B, 1]
            "clinical_toxicity": self.clinical_toxicity_head(emb),     # [B, 1]
        }


class MultiPropertyLoss(nn.Module):
    """
    Loss combinée multi-propriétés.

    - Classification (toxicité, efficacité, BBB, toxicité clinique) : BCE
    - Régression (solubilité, lipophilicité) : Huber
    - Masquage NaN pour labels manquants
    """

    def __init__(self, tox_pos_weight=None, task_weights=None,
                 efficacy_pos_weight=None, bbb_pos_weight=None, clintox_pos_weight=None,
                 uncertainty_weighting=True):
        super().__init__()
        self.register_buffer("tox_pos_weight", tox_pos_weight)
        self.register_buffer("efficacy_pos_weight", efficacy_pos_weight)
        self.register_buffer("bbb_pos_weight", bbb_pos_weight)
        self.register_buffer("clintox_pos_weight", clintox_pos_weight)

        default_weights = {
            "toxicity": 1.0,
            "efficacy": 2.0,
            "solubility": 0.5,
            "lipophilicity": 0.5,
            "bbb_penetration": 0.8,
            "clinical_toxicity": 0.8,
        }
        self.task_weights = task_weights or default_weights
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting:
            # Kendall et al. 2018: log(sigma^2) per task, learned jointly
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in self.task_weights
            })

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
            # Les têtes non supervisées sur ce batch peuvent produire NaN/Inf.
            # Les masquer AVANT BCE/Huber évite 0 * NaN dans la réduction.
            pred_safe = torch.where(valid_mask, pred, torch.zeros_like(pred))

            # efficacy = hiv_HIV_active : label BINAIRE (0/1) → classification (BCE),
            # cohérent avec l'évaluation (sigmoid + ROC-AUC). Auparavant en Huber
            # (régression) par erreur → loss incohérente avec la métrique.
            if name in ("toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"):
                pw_map = {
                    "toxicity": self.tox_pos_weight,
                    "efficacy": self.efficacy_pos_weight,
                    "bbb_penetration": self.bbb_pos_weight,
                    "clinical_toxicity": self.clintox_pos_weight,
                }
                pw = pw_map.get(name)
                if pw is not None:
                    per = F.binary_cross_entropy_with_logits(
                        pred_safe, target_safe, pos_weight=pw, reduction="none",
                    )
                else:
                    per = F.binary_cross_entropy_with_logits(
                        pred_safe, target_safe, reduction="none",
                    )
            else:
                # Régression : Huber loss (robuste aux outliers)
                per = F.huber_loss(pred_safe, target_safe, delta=1.0, reduction="none")

            loss = (per * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

            loss_details[name] = loss.item()
            if self.uncertainty_weighting and name in self.log_vars:
                precision = torch.exp(-self.log_vars[name])
                total_loss = total_loss + precision * loss + 0.5 * self.log_vars[name]
            else:
                total_loss = total_loss + weight * loss

        loss_details["total"] = total_loss.item()
        return total_loss, loss_details
