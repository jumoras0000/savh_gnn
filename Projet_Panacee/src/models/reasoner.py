"""
Module IA de Raisonnement Moléculaire – Phase 3.

Ce module implémente un réseau de type Transformer qui :
  1. Prend en entrée les embeddings de N molécules candidates
  2. Effectue de l'attention croisée pour trouver des interactions/synergies
  3. Analyse les combinaisons de molécules (superposition)
  4. Prédit un score de synergie et des doses optimales
  5. Génère un score de confiance pour chaque prédiction

Architecture :
  MoleculeEmbeddings → SelfAttention → CrossAttention → SynergyMLP → Outputs

Inspiré de :
  - Attention Is All You Need (Vaswani 2017)
  - Drug-Drug Interaction prediction (Ryu 2018)
  - Drug combination synergy prediction (Preuer 2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularReasonerBlock(nn.Module):
    """
    Bloc Transformer pour raisonner sur les molécules.
    Self-attention + FFN avec résidus et normalisation.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.15):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: [B, N, D] - N molécules, D dimensions
        mask: [B, N] - True = masqué (padding)
        """
        key_padding_mask = mask if mask is not None else None

        # Self-attention avec résidu
        attn_out, attn_weights = self.self_attention(
            x, x, x, key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + attn_out)

        # Feed-forward avec résidu
        x = self.norm2(x + self.ffn(x))

        return x, attn_weights


class SynergyAnalyzer(nn.Module):
    """
    Analyse les interactions par paires entre molécules.
    Calcule un score de synergie pour chaque combinaison.
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.interaction_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, mol_embeddings, mask=None):
        """
        mol_embeddings: [B, N, D]
        Retourne : synergy_scores [B, N, N] - matrice de synergie par paires
        """
        B, N, D = mol_embeddings.shape
        synergy_matrix = torch.zeros(B, N, N, device=mol_embeddings.device)
        # valid[:, i] = 1 si la molécule i est réelle (pas du padding)
        valid = (~mask).float() if mask is not None else None

        for i in range(N):
            for j in range(i + 1, N):
                # Combiner : mol_i, mol_j, et leur produit élément par élément
                combined = torch.cat([
                    mol_embeddings[:, i, :],
                    mol_embeddings[:, j, :],
                    mol_embeddings[:, i, :] * mol_embeddings[:, j, :],
                ], dim=-1)  # [B, 3D]

                score = torch.sigmoid(self.interaction_mlp(combined).squeeze(-1))
                if valid is not None:
                    # Annuler la synergie si l'une des deux molécules est du padding
                    score = score * valid[:, i] * valid[:, j]
                synergy_matrix[:, i, j] = score
                synergy_matrix[:, j, i] = score  # symétrique

        return synergy_matrix


class DosePredictor(nn.Module):
    """
    Prédit les doses optimales pour chaque molécule dans une combinaison.
    """

    def __init__(self, d_model: int, num_dose_levels: int = 7, dropout: float = 0.15):
        super().__init__()
        self.num_dose_levels = num_dose_levels
        self.dose_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_dose_levels),
        )

    def forward(self, mol_embedding):
        """
        mol_embedding: [B, D] ou [B, N, D]
        Retourne : distribution de probabilité sur les niveaux de dose
        """
        logits = self.dose_mlp(mol_embedding)
        return F.softmax(logits, dim=-1)


class ConfidenceEstimator(nn.Module):
    """
    Estime la confiance du modèle dans ses prédictions.
    """

    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.confidence_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.confidence_mlp(x)


class MolecularReasoner(nn.Module):
    """
    Moteur d'IA de Raisonnement Moléculaire.

    Pipeline complet :
      1. Projeter les embeddings moléculaires dans l'espace du raisonneur
      2. Ajouter des positional embeddings pour distinguer les molécules
      3. Appliquer N couches de Transformer (self-attention)
      4. Analyser les synergies par paires
      5. Prédire les doses optimales
      6. Estimer la confiance
      7. Calculer un score global de réussite

    Entrée : embeddings de N molécules [B, N, mol_dim]
    Sorties :
      - reasoned_embeddings: [B, N, D] embeddings enrichis
      - synergy_matrix: [B, N, N] scores de synergie
      - dose_distributions: [B, N, num_doses] probabilités par dose
      - confidence: [B, 1] confiance globale
      - success_score: [B, 1] score de réussite prédit
      - attention_weights: list de [B, N, N] attention par couche
    """

    def __init__(
        self,
        mol_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_molecules: int = 5,
        num_dose_levels: int = 7,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.mol_dim = mol_dim
        self.hidden_dim = hidden_dim
        self.max_molecules = max_molecules
        self.num_dose_levels = num_dose_levels

        # Projection des embeddings moléculaires → espace raisonneur
        self.input_projection = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Positional embeddings (pour distinguer molécule 1, 2, 3...)
        self.position_embedding = nn.Embedding(max_molecules, hidden_dim)

        # Couches Transformer
        self.layers = nn.ModuleList([
            MolecularReasonerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Modules spécialisés
        self.synergy_analyzer = SynergyAnalyzer(hidden_dim, dropout)
        self.dose_predictor = DosePredictor(hidden_dim, num_dose_levels, dropout)
        self.confidence_estimator = ConfidenceEstimator(hidden_dim, dropout)

        # Score de réussite global
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, mol_embeddings, mask=None):
        """
        Args:
            mol_embeddings: [B, N, mol_dim] embeddings de N molécules
            mask: [B, N] True = position de padding (à ignorer)

        Returns:
            dict avec toutes les prédictions
        """
        B, N, _ = mol_embeddings.shape

        # 1. Projeter dans l'espace du raisonneur
        x = self.input_projection(mol_embeddings)  # [B, N, hidden_dim]

        # 2. Ajouter positional embeddings (clamp si N > max_molecules -> pas d'IndexError)
        positions = torch.arange(N, device=x.device).clamp(max=self.max_molecules - 1)
        positions = positions.unsqueeze(0).expand(B, -1)
        x = x + self.position_embedding(positions)

        # 3. Passer dans les couches Transformer
        all_attn_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            all_attn_weights.append(attn_w)

        # 4. Analyser les synergies
        synergy_matrix = self.synergy_analyzer(x, mask)

        # 5. Prédire les doses
        dose_distributions = self.dose_predictor(x)  # [B, N, num_doses]

        # 6. Calculer la représentation globale (pool sur les molécules non masquées)
        if mask is not None:
            # Masquer les positions de padding
            x_masked = x.clone()
            x_masked[mask] = 0
            # Mean pooling sur les positions valides
            valid_counts = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1)
            global_repr = x_masked.sum(dim=1) / valid_counts  # [B, hidden_dim]
        else:
            global_repr = x.mean(dim=1)  # [B, hidden_dim]

        # 7. Confiance et score de réussite
        confidence = self.confidence_estimator(global_repr)  # [B, 1]
        success_score = self.success_predictor(global_repr)  # [B, 1]

        return {
            "reasoned_embeddings": x,         # [B, N, D]
            "synergy_matrix": synergy_matrix,  # [B, N, N]
            "dose_distributions": dose_distributions,  # [B, N, num_doses]
            "confidence": confidence,          # [B, 1]
            "success_score": success_score,    # [B, 1]
            "attention_weights": all_attn_weights,
            "global_representation": global_repr,
        }


class ReasonerLoss(nn.Module):
    """
    Loss pour entraîner le module de raisonnement.

    Combine :
      - Synergy loss (si labels de synergie disponibles)
      - Success prediction loss
      - Confidence calibration loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        predictions: dict du MolecularReasoner
        targets: dict avec :
          - synergy_labels: [B, N, N] (optionnel)
          - success_labels: [B, 1] (optionnel)
        """
        total_loss = torch.tensor(0.0, device=predictions["confidence"].device)
        details = {}

        # Loss synergie
        if "synergy_labels" in targets:
            syn_pred = predictions["synergy_matrix"]
            syn_true = targets["synergy_labels"]
            valid = ~torch.isnan(syn_true)
            if valid.sum() > 0:
                syn_loss = F.mse_loss(syn_pred[valid], syn_true[valid])
                total_loss = total_loss + syn_loss
                details["synergy"] = syn_loss.item()

        # Loss score de réussite
        if "success_labels" in targets:
            suc_pred = predictions["success_score"]
            suc_true = targets["success_labels"]
            valid = ~torch.isnan(suc_true)
            if valid.sum() > 0:
                suc_loss = F.binary_cross_entropy(suc_pred[valid], suc_true[valid])
                total_loss = total_loss + 2.0 * suc_loss
                details["success"] = suc_loss.item()

        # Régularisation : confiance doit être calibrée
        # Si le modèle a tort, la confiance doit être basse
        conf = predictions["confidence"]
        conf_reg = -torch.mean(conf * torch.log(conf + 1e-8) +
                               (1 - conf) * torch.log(1 - conf + 1e-8))
        total_loss = total_loss + 0.1 * conf_reg
        details["confidence_entropy"] = conf_reg.item()

        details["total"] = total_loss.item()
        return total_loss, details
