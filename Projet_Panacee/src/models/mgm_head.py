"""
Tête de prédiction Masked Graph Modeling (MGM) – v2.

Correction : la prédiction utilise les embeddings *au niveau nœud*
(pas la représentation moléculaire globale). Cela donne au modèle
l'information locale nécessaire pour reconstruire les features masquées.
"""
import torch
import torch.nn as nn


class MGMHead(nn.Module):
    """
    Prédit le TYPE d'atome masqué (classification) à partir des embeddings
    au niveau nœud (sortie des couches GNN *avant* pooling global).

    La sortie est un vecteur de logits sur le vocabulaire de types d'atomes
    (cross-entropy), pas une régression du vecteur de features : l'identité de
    l'élément ne se déduit pas de la topologie, l'objectif est donc difficile.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 13, atom_dim: int | None = None):
        super().__init__()
        # `atom_dim` conservé pour compatibilité d'appel ; la sortie est num_classes.
        self.num_classes = num_classes
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, node_embeddings, masked_indices_per_graph, batch_vector):
        """
        Args:
            node_embeddings       : [N_total, hidden_dim]  embeddings de TOUS les nœuds
            masked_indices_per_graph : list[list[int]]     indices locaux masqués par graphe
            batch_vector          : [N_total]              mapping nœud → graphe
        Returns:
            predictions : [M, atom_dim]  M = nb total d'atomes masqués
        """
        # Calculer l'offset de nœuds pour chaque graphe dans le batch
        device = node_embeddings.device
        predictions = []

        # Offset cumulatif : nombre de nœuds avant le graphe i
        num_graphs = batch_vector.max().item() + 1
        for g in range(num_graphs):
            mask_g = (batch_vector == g)
            offset = mask_g.nonzero(as_tuple=True)[0][0].item()  # premier nœud du graphe g
            local_indices = masked_indices_per_graph[g]
            if len(local_indices) == 0:
                continue
            global_indices = [offset + li for li in local_indices]
            global_indices_t = torch.tensor(global_indices, dtype=torch.long, device=device)
            selected = node_embeddings[global_indices_t]   # [K, hidden_dim]
            pred = self.predictor(selected)                # [K, atom_dim]
            predictions.append(pred)

        if predictions:
            return torch.cat(predictions, dim=0)
        return torch.empty(0, self.predictor[-1].out_features, device=device)


class MaskedGraphModel(nn.Module):
    """
    Modèle complet Phase 1 : Encodeur + Tête MGM.

    Le forward expose les embeddings de nœuds intermédiaires
    (avant le pooling global) pour la tête MGM.
    """

    def __init__(self, encoder, mgm_head):
        super().__init__()
        self.encoder = encoder
        self.mgm_head = mgm_head

    def _encode_nodes(self, x, edge_index, edge_attr, batch):
        """
        Embeddings par nœud (AVANT pooling global), via l'encodeur.
        Délègue à MolecularEncoder.encode_nodes (logique unique, conv-agnostique).
        """
        return self.encoder.encode_nodes(x, edge_index, edge_attr)  # [N_total, hidden_dim]

    def forward(self, x, edge_index, edge_attr, batch, masked_atom_indices):
        # 1. Embeddings par nœud (avant pooling)
        node_emb = self._encode_nodes(x, edge_index, edge_attr, batch)

        # 2. Prédiction des features masquées
        predictions = self.mgm_head(node_emb, masked_atom_indices, batch)
        return predictions

    def get_encoder_checkpoint(self):
        """Retourne le state_dict de l'encodeur seul (pour Phase 2)."""
        return self.encoder.state_dict()
