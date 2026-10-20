"""
Encodeur GNN moléculaire – v3.

Nouveautés v3 :
  1. Conv sélectionnable : "mpnn" (Gilmer MLP) ou "attention" (GATv2 edge-aware).
  2. Attention multi-têtes tenant compte des features d'arête (edge_attr).
  3. Sum-pooling NORMALISÉ par sqrt(n_atomes) → stable quelle que soit la taille.
  4. Triple pooling (mean + sum_norm + max) avec gating appris (conservé).

Conserve v2 :
  - Message passing utilise x_i, x_j et edge_attr.
  - Résidus + LayerNorm + Dropout par couche.
  - Features d'entrée normalisées [0,1] (9-dim).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax


class GraphConvolution(MessagePassing):
    """
    Couche de message-passing complète (Gilmer et al. 2017).
    m_ij = MLP([x_i ‖ x_j ‖ e_ij])   — message
    h_i'  = MLP([h_i ‖ Aggr(m_ij)])   — update
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        # Fix H1 (audit 2026-10-20) : dropout stocké en nn.Dropout pour qu'il
        # soit activé par enable_mc_dropout(). Avant → paramètre ignoré,
        # MC-Dropout totalement inactif sous CONV_TYPE="mpnn".
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inp)

    def update(self, aggr_out, x):
        inp = torch.cat([x, aggr_out], dim=-1)
        return self.dropout_layer(self.update_mlp(inp))


class EdgeAttentionConv(MessagePassing):
    """
    Message-passing à attention multi-têtes, edge-aware (style GATv2).

    Pour chaque arête (j → i) :
        e_ij = LeakyReLU(W_i x_i + W_j x_j + W_e edge_ij)
        α_ij = softmax_i( aᵀ e_ij )            (par tête, normalisé sur les voisins de i)
        m_ij = α_ij · (W_m x_j)
        h_i' = Σ_j m_ij

    L'attention apprend QUELS voisins (et via quelles liaisons) comptent le plus,
    ce qui dépasse l'agrégation uniforme du MPNN classique.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr="add")
        assert hidden_dim % heads == 0, "hidden_dim doit être divisible par heads"
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout

        self.lin_i = nn.Linear(hidden_dim, hidden_dim)
        self.lin_j = nn.Linear(hidden_dim, hidden_dim)
        self.lin_e = nn.Linear(edge_dim, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)
        self.att = nn.Parameter(torch.empty(1, heads, self.head_dim))

        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        H, D = self.heads, self.head_dim
        xi = self.lin_i(x_i).view(-1, H, D)
        xj = self.lin_j(x_j).view(-1, H, D)
        ee = self.lin_e(edge_attr).view(-1, H, D)

        # GATv2 : non-linéarité AVANT le produit avec le vecteur d'attention
        a = F.leaky_relu(xi + xj + ee, negative_slope=0.2)
        alpha = (a * self.att).sum(dim=-1)              # [E, heads]
        alpha = softmax(alpha, index, ptr, size_i)      # normalisé sur les voisins de i
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = self.lin_msg(x_j).view(-1, H, D)          # [E, heads, head_dim]
        out = msg * alpha.unsqueeze(-1)
        return out.reshape(-1, self.hidden_dim)


def _make_conv(conv_type: str, hidden_dim: int, edge_dim: int, heads: int, dropout: float):
    conv_type = (conv_type or "attention").lower()
    if conv_type == "mpnn":
        return GraphConvolution(hidden_dim, edge_dim, dropout)
    if conv_type == "attention":
        return EdgeAttentionConv(hidden_dim, edge_dim, heads=heads, dropout=dropout)
    raise ValueError(f"conv_type inconnu: {conv_type} (attendu: 'mpnn' ou 'attention')")


class VirtualNode(nn.Module):
    """Virtual Node : broadcast global → tous les nœuds à chaque couche.

    Ajoute +1-3% sur OGB-mol benchmarks en permettant un message passing
    global en O(n) sans full connectivity.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.vn_embedding = nn.Embedding(1, hidden_dim)
        self.mlp_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.vn_embedding.weight)

    def pool_and_broadcast(self, x, batch, vn_feat):
        """Agrège nœuds → VN, met à jour VN, broadcast retour."""
        pooled = global_mean_pool(x, batch)  # [B, D]
        vn_feat = vn_feat + self.mlp_update(pooled + vn_feat)  # residual
        x = x + vn_feat[batch]
        return x, vn_feat


class MolecularEncoder(nn.Module):
    """
    Encodeur GNN pour molécules.

    Architecture :
        Embedding → N × (Conv + VN + Norm + SiLU + Dropout + Residual) → TriplePool → Projection
    """

    def __init__(
        self,
        atom_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 6,
        edge_dim: int = 6,
        output_dim: int = 256,
        dropout: float = 0.2,
        conv_type: str = "attention",
        attention_heads: int = 4,
        use_virtual_node: bool = True,
        use_chirality: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.conv_type = conv_type
        self.use_virtual_node = use_virtual_node
        self.use_chirality = use_chirality

        # ── Embedding initial ─────────────────────────────────────────
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if use_chirality:
            self.chirality_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
            nn.init.zeros_(self.chirality_embedding.weight[0])

        # ── Virtual Node ──────────────────────────────────────────────
        if use_virtual_node:
            self.virtual_node = VirtualNode(hidden_dim, dropout)

        # ── Couches de convolution ────────────────────────────────────
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(_make_conv(conv_type, hidden_dim, edge_dim, attention_heads, dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.drops.append(nn.Dropout(dropout))

        # ── Pooling global (mean + sum_norm + max) avec gating appris ──
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

    def encode_nodes(self, x, edge_index, edge_attr, batch=None, chirality=None):
        """Embeddings par nœud (avant pooling) — réutilisé par la tête MGM (Phase 1)."""
        h = self.atom_embedding(x)
        if self.use_chirality:
            if chirality is None or chirality.shape != (x.size(0),):
                raise ValueError("Chiralité par atome requise pour cet encodeur")
            h = h + self.chirality_embedding(chirality.long())

        vn_feat = None
        if self.use_virtual_node:
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            n_graphs = int(batch.max().item()) + 1 if h.size(0) > 0 else 1
            vn_feat = self.virtual_node.vn_embedding.weight.expand(n_graphs, -1).clone()

        for conv, norm, drop in zip(self.convs, self.norms, self.drops, strict=False):
            h_res = h
            if self.use_virtual_node:
                h, vn_feat = self.virtual_node.pool_and_broadcast(h, batch, vn_feat)
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.silu(h)
            h = drop(h)
            h = h + h_res
        return h

    def forward(self, x, edge_index, edge_attr, batch=None, chirality=None):
        """
        Args:
            x          : [N, atom_dim]
            edge_index : [2, E]
            edge_attr  : [E, edge_dim]
            batch      : [N]
        Returns:
            [B, output_dim]
        """
        # 1-2. Embedding + message-passing avec skip connections (+ Virtual Node)
        h = self.encode_nodes(x, edge_index, edge_attr, batch, chirality)

        # 3. Pooling global
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h_mean = global_mean_pool(h, batch)   # [B, D]
        h_max = global_max_pool(h, batch)     # [B, D]

        # Sum-pool NORMALISÉ par sqrt(n_atomes) → magnitude stable
        ones = torch.ones(h.size(0), 1, device=h.device)
        counts = global_add_pool(ones, batch).clamp(min=1.0)   # [B, 1]
        h_sum = global_add_pool(h, batch) / counts.sqrt()       # [B, D]

        # Gating : pondération apprise des 3 stratégies
        h_cat = torch.cat([h_mean, h_sum, h_max], dim=-1)       # [B, 3D]
        gates = self.pool_gate(h_cat)                            # [B, 3]

        h_pooled = (
            gates[:, 0:1] * h_mean
            + gates[:, 1:2] * h_sum
            + gates[:, 2:3] * h_max
        )

        # 4. Projection
        return self.projection(h_pooled)
