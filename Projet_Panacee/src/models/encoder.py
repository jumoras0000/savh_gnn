"""
Encodeur GNN moléculaire – v2 (corrigé).

Corrections par rapport à v1 :
  1. Message passing utilise BOTH x_i ET x_j (+ edge_attr).
  2. Update function explicite après agrégation.
  3. Dropout appliqué dans chaque couche (pas seulement embedding/pooling).
  4. Triple pooling (mean + sum + max) avec gating appris.
  5. Features d'entrée attendues normalisées [0,1] (9-dim).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool


class GraphConvolution(MessagePassing):
    """
    Couche de message-passing complète (Gilmer et al. 2017).
    m_ij = MLP([x_i ‖ x_j ‖ e_ij])   — message
    h_i'  = MLP([h_i ‖ Aggr(m_ij)])   — update
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim

        # Message MLP : prend x_i, x_j, edge_attr
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update MLP : combine nœud original + messages agrégés
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Utilise les deux nœuds + attributs d'arête."""
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inp)

    def update(self, aggr_out, x):
        """Fusionne messages agrégés avec features originales du nœud."""
        inp = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(inp)


class MolecularEncoder(nn.Module):
    """
    Encodeur GNN pour molécules.

    Architecture :
        Embedding  →  N × (Conv + Norm + SiLU + Dropout + Residual)  →  TriplePool  →  Projection
    """

    def __init__(
        self,
        atom_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 6,
        edge_dim: int = 6,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # ── Embedding initial ─────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Couches de convolution ────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GraphConvolution(hidden_dim, edge_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.drops.append(nn.Dropout(dropout))

        # ── Pooling global (mean + sum + max) avec gating appris ──────
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # ── Projection finale ─────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x          : [N, atom_dim]
            edge_index : [2, E]
            edge_attr  : [E, edge_dim]
            batch      : [N]  (indices de graphe)
        Returns:
            [B, output_dim]  représentation moléculaire
        """
        # 1. Embedding
        h = self.atom_embedding(x)

        # 2. Message-passing avec skip connections
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.silu(h)
            h = drop(h)
            h = h + h_res  # résidu

        # 3. Pooling global (batch par défaut = un seul graphe)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h_mean = global_mean_pool(h, batch)   # [B, D]
        h_sum  = global_add_pool(h, batch)    # [B, D]
        h_max  = global_max_pool(h, batch)    # [B, D]

        # Gating : pondération apprise des 3 stratégies
        h_cat = torch.cat([h_mean, h_sum, h_max], dim=-1)   # [B, 3D]
        gates = self.pool_gate(h_cat)                        # [B, 3]

        h_pooled = (
            gates[:, 0:1] * h_mean
            + gates[:, 1:2] * h_sum
            + gates[:, 2:3] * h_max
        )  # [B, D]

        # 4. Projection
        return self.projection(h_pooled)
