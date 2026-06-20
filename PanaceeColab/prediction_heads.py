"""
prediction_heads.py – Têtes de prédiction et module Raisonneur
==============================================================
  • MGMHead             – Reconstruction d'atomes masqués (Phase 1)
  • MaskedGraphModel    – Encodeur + MGMHead (Phase 1 complet)
  • ToxicityClassifier  – Classification multi-tâches (Phase 2)
  • PropertyHead        – Tête générique mono-groupe
  • MultiPropertyPredictor – 6 groupes de propriétés (Phase 3)
  • MolecularReasoner   – Transformer de raisonnement (Phase 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  1. MGMHead – Masked Graph Modeling (Phase 1)
# ══════════════════════════════════════════════════════════════════════

class MGMHead(nn.Module):
    """
    Prédit les features atomiques originales à partir des
    embeddings *au niveau nœud* (avant le pooling global).

    Référence : Hu et al. (2020) "Strategies for Pre-training Graph
    Neural Networks", ICLR 2020.
    """

    def __init__(self, hidden_dim: int = 256, atom_dim: int = 9):
        super().__init__()
        self.atom_dim  = atom_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, atom_dim),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,     # [N_total, hidden_dim]
        masked_indices_per_graph: list,    # list[list[int]]
        batch_vector: torch.Tensor,        # [N_total]
    ) -> torch.Tensor:
        """
        Retourne les prédictions pour tous les atomes masqués.
        Forme : [M, atom_dim]   M = total atomes masqués dans le batch.
        """
        device      = node_embeddings.device
        predictions = []
        num_graphs  = int(batch_vector.max().item()) + 1

        for g in range(num_graphs):
            mask_g  = (batch_vector == g)
            nodes_g = mask_g.nonzero(as_tuple=True)[0]
            if nodes_g.numel() == 0:
                continue

            offset        = nodes_g[0].item()
            local_indices = masked_indices_per_graph[g]
            if len(local_indices) == 0:
                continue

            global_idx = torch.tensor(
                [offset + li for li in local_indices],
                dtype=torch.long, device=device,
            )
            predictions.append(self.predictor(node_embeddings[global_idx]))

        if predictions:
            return torch.cat(predictions, dim=0)
        return torch.empty(0, self.atom_dim, device=device)


class MaskedGraphModel(nn.Module):
    """Phase 1 complet : Encodeur + MGMHead."""

    def __init__(self, encoder: nn.Module, mgm_head: MGMHead):
        super().__init__()
        self.encoder  = encoder
        self.mgm_head = mgm_head

    def forward(self, x, edge_index, edge_attr, batch, masked_atom_indices: list):
        # Embeddings au niveau nœud (avant pooling)
        node_emb = self.encoder.encode_nodes(x, edge_index, edge_attr)
        return self.mgm_head(node_emb, masked_atom_indices, batch)

    def get_encoder_state(self) -> dict:
        return self.encoder.state_dict()


# ══════════════════════════════════════════════════════════════════════
#  2. ToxicityClassifier – Phase 2
# ══════════════════════════════════════════════════════════════════════

class ToxicityClassifier(nn.Module):
    """
    Encodeur GNN pré-entraîné + tête de classification multi-tâches.

    Fonctionnalités anti-surapprentissage :
      - BatchNorm dans la tête (stabilise le fine-tuning)
      - Dégel progressif de l'encodeur couche par couche
      - Dropout adaptatif
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_tasks: int    = 12,
        hidden_dim: int   = 256,
        dropout: float    = 0.20,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder   = encoder
        self.num_tasks = num_tasks

        if freeze_encoder:
            self._freeze_encoder()

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
        self._init_weights()

    def _init_weights(self):
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
        """Dégel progressif : 1 couche par epoch après freeze_epochs."""
        n = getattr(self.encoder, "num_layers", 6)
        if epoch < total_freeze_epochs:
            self._freeze_encoder()
        else:
            k = min(epoch - total_freeze_epochs + 1, n)
            self._freeze_encoder()
            # Dégeler les k dernières couches (proches de la sortie)
            convs = getattr(self.encoder, "convs", getattr(self.encoder, "atom_convs", None))
            norms = getattr(self.encoder, "norms", None)
            if convs is not None:
                for i in range(n - k, n):
                    for p in convs[i].parameters():
                        p.requires_grad = True
                    if norms is not None:
                        for p in norms[i].parameters():
                            p.requires_grad = True
            for p in self.encoder.proj.parameters():
                p.requires_grad = True

    def forward(self, batch):
        mol_emb = self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        return self.classifier(mol_emb)


# ══════════════════════════════════════════════════════════════════════
#  3. MultiPropertyPredictor – Phase 3
# ══════════════════════════════════════════════════════════════════════

class PropertyHead(nn.Module):
    """Tête de prédiction pour un groupe de propriétés."""

    def __init__(self, input_dim: int, num_tasks: int, dropout: float = 0.20):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_tasks),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


class MultiPropertyPredictor(nn.Module):
    """
    Encodeur GNN + 6 têtes spécialisées :
      toxicity (12)       – classification BCE
      efficacy (1)        – classification BCE
      solubility (1)      – régression MSE
      lipophilicity (1)   – régression MSE
      bioavailability (1) – classification BCE
      metabolic_stability (1) – classification BCE

    Total : 17 tâches simultanées.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        dropout: float  = 0.20,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder    = encoder
        self.hidden_dim = hidden_dim

        if freeze_encoder:
            self._freeze_encoder()

        # Couche partagée entre toutes les têtes
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Têtes spécialisées
        self.toxicity_head           = PropertyHead(hidden_dim, 12, dropout)
        self.efficacy_head           = PropertyHead(hidden_dim,  1, dropout)
        self.solubility_head         = PropertyHead(hidden_dim,  1, dropout)
        self.lipophilicity_head      = PropertyHead(hidden_dim,  1, dropout)
        self.bioavailability_head    = PropertyHead(hidden_dim,  1, dropout)
        self.metabolic_head          = PropertyHead(hidden_dim,  1, dropout)

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def gradual_unfreeze(self, epoch: int, total_freeze_epochs: int = 5):
        n = getattr(self.encoder, "num_layers", 6)
        if epoch < total_freeze_epochs:
            self._freeze_encoder()
        else:
            k = min(epoch - total_freeze_epochs + 1, n)
            self._freeze_encoder()
            convs = getattr(self.encoder, "convs", getattr(self.encoder, "atom_convs", None))
            if convs is not None:
                for i in range(n - k, n):
                    for p in convs[i].parameters():
                        p.requires_grad = True
            for p in self.encoder.proj.parameters():
                p.requires_grad = True

    def encode(self, batch) -> torch.Tensor:
        mol = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.shared(mol)

    def forward(self, batch) -> dict:
        z = self.encode(batch)
        return {
            "toxicity"           : self.toxicity_head(z),
            "efficacy"           : self.efficacy_head(z),
            "solubility"         : self.solubility_head(z),
            "lipophilicity"      : self.lipophilicity_head(z),
            "bioavailability"    : self.bioavailability_head(z),
            "metabolic_stability": self.metabolic_head(z),
        }


# ══════════════════════════════════════════════════════════════════════
#  4. MolecularReasoner – Phase 3 (Transformer de raisonnement)
# ══════════════════════════════════════════════════════════════════════

class ReasonerBlock(nn.Module):
    """Bloc Transformer : self-attention + FFN + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, attn_w = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_w


class MolecularReasoner(nn.Module):
    """
    Raisonneur moléculaire.
    Prend N embeddings de molécules → prédit synergie + score global.

    Architecture inspirée de :
      - Vaswani et al. (2017) "Attention Is All You Need"
      - Ryu et al. (2018) "Deep Learning Improves Prediction of Drug-Drug
        and Drug-Food Interactions"
      - Preuer et al. (2018) "DeepSynergy: Predicting Anti-Cancer Drug
        Synergy with Deep Learning"
    """

    def __init__(
        self,
        mol_emb_dim: int  = 256,
        d_model: int      = 512,
        num_heads: int    = 8,
        num_layers: int   = 4,
        dropout: float    = 0.15,
        num_dose_levels: int = 7,
    ):
        super().__init__()
        self.mol_proj = nn.Linear(mol_emb_dim, d_model)

        self.blocks = nn.ModuleList([
            ReasonerBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

        # Prédictions de sortie
        self.synergy_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.dose_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_dose_levels),
            nn.Softmax(dim=-1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.combo_score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, mol_embeddings: torch.Tensor, mask=None):
        """
        Args:
            mol_embeddings : [B, N, mol_emb_dim]   B=batch, N=nb molécules
            mask           : [B, N] True = position ignorée (padding)

        Returns:
            dict avec synergy, doses, confidence, combo_score, attn_weights
        """
        x = self.mol_proj(mol_embeddings)   # [B, N, d_model]

        attn_weights = []
        for block in self.blocks:
            x, aw = block(x, mask)
            attn_weights.append(aw)

        # CLS token simulé : pooling moyen sur les molécules non-maskées
        if mask is not None:
            valid_mask  = ~mask.unsqueeze(-1).float()   # [B, N, 1]
            n_valid     = valid_mask.sum(dim=1).clamp(min=1)
            cls_token   = (x * valid_mask).sum(dim=1) / n_valid
        else:
            cls_token = x.mean(dim=1)                   # [B, d_model]

        # Scores de synergie par paires
        N     = x.size(1)
        synergy_matrix = torch.zeros(x.size(0), N, N, device=x.device)
        for i in range(N):
            for j in range(i + 1, N):
                pair = torch.cat([x[:, i, :], x[:, j, :]], dim=-1)
                s    = self.synergy_head(pair).squeeze(-1)
                synergy_matrix[:, i, j] = s
                synergy_matrix[:, j, i] = s    # symétrique

        return {
            "combo_score"   : self.combo_score_head(cls_token),   # [B, 1]
            "doses"         : self.dose_head(x),                  # [B, N, num_dose_levels]
            "confidence"    : self.confidence_head(cls_token),    # [B, 1]
            "synergy_matrix": synergy_matrix,                     # [B, N, N]
            "attn_weights"  : attn_weights,
        }
