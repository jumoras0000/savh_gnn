"""
gnn_models.py – Architectures GNN état de l'art pour molécules
==============================================================
5 architectures implémentées, sélectionnables via `build_encoder()` :

  1. MPNN   – Message Passing NN (Gilmer et al. 2017, J. Chem. Phys.)
              Baseline robuste avec attention d'arête et triple pooling.

  2. AttFP  – Attentive Fingerprints (Xiong et al. 2020, JACS)
              "Pushing the Boundaries of Molecular Representation for
               Drug Discovery with the Graph Attention Mechanism"
              → Meilleur sur Tox21 / SIDER dans les benchmarks MoleculeNet.

  3. GIN    – Graph Isomorphism Network (Xu et al. 2019, ICLR)
              "How Powerful are Graph Neural Networks?"
              + Intégration features arêtes (Hu et al. 2020, ICLR
                "Strategies for Pre-training Graph Neural Networks")
              → Maximal en expressivité WL.

  4. PNA    – Principal Neighbourhood Aggregation (Corso et al. 2020, NeurIPS)
              "Principal Neighbourhood Aggregation for Graph Nets"
              Agrégateurs multiples (mean, max, min, std)
              × Scalers (identité, amplification, atténuation).

  5. GPS    – General Powerful Scalable Graph Transformer
              (Rampásek et al. 2022, NeurIPS)
              "Recipe for a General, Powerful, Scalable Graph Transformer"
              MPNN local + attention globale multi-tête par couche.
              → État de l'art sur OGB / MoleculeNet 2022-2024.

Toutes les architectures exposent la même interface :
    encoder = build_encoder(arch='attfp', ...)
    mol_emb = encoder(x, edge_index, edge_attr, batch)  # → [B, output_dim]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool, global_add_pool, global_max_pool,
)
from torch_geometric.utils import softmax as pyg_softmax

from config import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT


# ══════════════════════════════════════════════════════════════════════
#  COMPOSANTS PARTAGÉS
# ══════════════════════════════════════════════════════════════════════

class GatedPooling(nn.Module):
    """
    Pooling global pondéré appris : gates sur (mean ‖ sum ‖ max).
    Réduit le biais d'une seule stratégie de pooling.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_mean = global_mean_pool(h, batch)   # [B, D]
        h_sum  = global_add_pool(h, batch)    # [B, D]
        h_max  = global_max_pool(h, batch)    # [B, D]

        h_cat  = torch.cat([h_mean, h_sum, h_max], dim=-1)  # [B, 3D]
        g      = self.gate(h_cat)                           # [B, 3]

        return g[:, 0:1] * h_mean + g[:, 1:2] * h_sum + g[:, 2:3] * h_max


class WarmupCosineScheduler:
    """Scheduler : warmup linéaire puis cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, lr_min: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr_min        = lr_min
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            alpha = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            alpha    = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.lr_min + (base_lr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════════════
#  1. MPNN – Message Passing Neural Network (Gilmer 2017)
# ══════════════════════════════════════════════════════════════════════

class MPNNConv(MessagePassing):
    """
    Couche MPNN complète.
    m_ij = MLP([x_i ‖ x_j ‖ e_ij])   ← message
    h_i' = MLP([h_i ‖ Σ m_ij])         ← update (résidu)
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=-1))


class MPNNEncoder(nn.Module):
    """
    Encodeur MPNN complet :
      Embedding → N × (MPNNConv + LayerNorm + SiLU + Dropout + Résidu) → GatedPool → Projection
    """
    def __init__(
        self,
        atom_dim: int   = ATOM_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        edge_dim: int   = BOND_FEATURE_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([MPNNConv(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.pool = GatedPooling(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        """Node-level embeddings (avant pooling) — utilisé par MGMHead."""
        h = self.embedding(x)
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h = drop(F.silu(norm(conv(h, edge_index, edge_attr)))) + h
        return h

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.encode_nodes(x, edge_index, edge_attr)
        return self.proj(self.pool(h, batch))


# ══════════════════════════════════════════════════════════════════════
#  2. ATTFP – Attentive Fingerprints (Xiong et al. 2020, JACS)
# ══════════════════════════════════════════════════════════════════════

class AttFPAtomConv(MessagePassing):
    """
    Atom-level attention (éq. 5-8, Xiong 2020).
    a_ij = softmax_j( LeakyReLU( W_a · [h_i ‖ e_ij ‖ h_j] ) )
    h_i' = ELU( Σ_j a_ij · W_n h_j )   + h_i  (skip)
    """
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__(aggr="add", node_dim=0)
        self.attn_linear = nn.Linear(2 * hidden_dim + edge_dim, 1)
        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout     = nn.Dropout(dropout)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        alpha = self.attn_linear(torch.cat([x_i, edge_attr, x_j], dim=-1))
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = pyg_softmax(alpha, index)          # softmax sur voisinage
        alpha = self.dropout(alpha)
        return alpha * self.node_linear(x_j)

    def update(self, aggr_out, x):
        return self.norm(F.elu(aggr_out) + x)      # ELU + résidu


class AttFPMolConv(nn.Module):
    """
    Molecule-level attention (éq. 9-12, Xiong 2020).
    Readout : un « super-nœud » (contexte) attends sur tous les atomes.
    """
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.attn_linear = nn.Linear(2 * hidden_dim, 1)
        self.context_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.dropout     = nn.Dropout(dropout)
        self.norm        = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Context vecteur = moyenne des nœuds par graphe
        context = global_mean_pool(h, batch)                # [B, D]
        ctx_exp = context[batch]                            # [N, D] — broadcast par graphe

        # Attention scores
        alpha = self.attn_linear(torch.cat([h, ctx_exp], dim=-1))  # [N, 1]
        alpha = F.leaky_relu(alpha, 0.2)

        # Softmax par graphe
        alpha = pyg_softmax(alpha, batch)                   # [N, 1]
        alpha = self.dropout(alpha)

        # Weighted sum → [B, D]
        h_weighted = global_add_pool(alpha * self.context_mlp(h), batch)
        return self.norm(F.elu(h_weighted) + context)


class AttFPEncoder(nn.Module):
    """
    Encodeur AttentiveFP complet.
    Atom conv × num_layers → Molecule conv × mol_layers → Projection
    """
    def __init__(
        self,
        atom_dim: int   = ATOM_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        mol_layers: int = 3,
        edge_dim: int   = BOND_FEATURE_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        # Edge embedding pour aligner la dimension
        self.edge_emb  = nn.Linear(edge_dim, edge_dim)

        self.atom_convs = nn.ModuleList([
            AttFPAtomConv(hidden_dim, edge_dim, dropout) for _ in range(num_layers)
        ])
        self.mol_convs = nn.ModuleList([
            AttFPMolConv(hidden_dim, dropout) for _ in range(mol_layers)
        ])

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        h = self.embedding(x)
        ea = self.edge_emb(edge_attr)
        for conv in self.atom_convs:
            h = conv(h, edge_index, ea)
        return h

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.encode_nodes(x, edge_index, edge_attr)
        mol = global_mean_pool(h, batch)          # init super-nœud
        for conv in self.mol_convs:
            mol = conv(h, batch)
        return self.proj(mol)


# ══════════════════════════════════════════════════════════════════════
#  3. GIN – Graph Isomorphism Network (Xu 2019 + Hu 2020 edge features)
# ══════════════════════════════════════════════════════════════════════

class GINConv(MessagePassing):
    """
    GIN avec features d'arêtes (Hu et al. 2020).
    h_i^k = MLP^k( (1+ε) · h_i^{k-1} + Σ_{j∈N(i)} (h_j^{k-1} + e_ij) )
    """
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, train_eps: bool = True):
        super().__init__(aggr="add")
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else 0.0
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        edge_feat = self.edge_proj(edge_attr)
        return self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_feat=edge_feat)
        )

    def message(self, x_j, edge_feat):
        return F.relu(x_j + edge_feat)


class GINEncoder(nn.Module):
    """
    Encodeur GIN complet.
    Suit l'architecture de Hu et al. (2020) pour la pré-entraînement GNN.
    """
    def __init__(
        self,
        atom_dim: int   = ATOM_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        edge_dim: int   = BOND_FEATURE_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float  = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.convs  = nn.ModuleList([GINConv(hidden_dim, edge_dim, dropout) for _ in range(num_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drops  = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.pool = GatedPooling(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        h = self.embedding(x)
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h = drop(norm(conv(h, edge_index, edge_attr))) + h
        return h

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.encode_nodes(x, edge_index, edge_attr)
        return self.proj(self.pool(h, batch))


# ══════════════════════════════════════════════════════════════════════
#  4. PNA – Principal Neighbourhood Aggregation (Corso 2020, NeurIPS)
# ══════════════════════════════════════════════════════════════════════

class PNAConv(MessagePassing):
    """
    Couche PNA : multiple aggregators × multiple scalers.

    Aggregators : mean, max, min, std
    Scalers     : identity, amplification (d/δ), attenuation (δ/d)
    → 4 agrégateurs × 3 scalers = 12 représentations concaténées.

    Référence : Corso et al. (2020) "Principal Neighbourhood Aggregation
    for Graph Nets", NeurIPS 2020.
    """
    AGGREGATORS = ("mean", "max", "min", "std")
    SCALERS     = ("identity", "amplification", "attenuation")

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        delta: float = 2.5,   # degré moyen du dataset (log-scaled)
    ):
        super().__init__(aggr=None)   # aggr custom
        self.hidden_dim = hidden_dim
        self.delta      = delta

        n_feats = len(self.AGGREGATORS) * len(self.SCALERS) + hidden_dim  # input + context
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr, deg=None):
        edge_feat = self.edge_mlp(edge_attr)
        out = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        return self.norm(self.mlp(torch.cat([x, out], dim=-1)) + x)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """Agrégation multi-stratégies."""
        from torch_geometric.utils import scatter
        n = dim_size if dim_size is not None else int(index.max()) + 1

        aggs = []
        for agg in self.AGGREGATORS:
            if agg == "mean":
                aggs.append(scatter(inputs, index, dim=0, dim_size=n, reduce="mean"))
            elif agg == "max":
                aggs.append(scatter(inputs, index, dim=0, dim_size=n, reduce="max"))
            elif agg == "min":
                aggs.append(scatter(inputs, index, dim=0, dim_size=n, reduce="min"))
            elif agg == "std":
                mean_ = scatter(inputs, index, dim=0, dim_size=n, reduce="mean")
                sq_   = scatter(inputs ** 2, index, dim=0, dim_size=n, reduce="mean")
                aggs.append(torch.sqrt(torch.clamp(sq_ - mean_ ** 2, min=0) + 1e-8))

        h_agg = torch.cat(aggs, dim=-1)   # [N, 4·D]

        # Degree scaler
        deg = scatter(torch.ones(index.size(0), device=inputs.device),
                      index, dim=0, dim_size=n, reduce="sum").unsqueeze(-1)
        log_deg = torch.log(deg.clamp(min=1.0)) / math.log(self.delta + 1e-8)

        scaled = []
        for i, agg in enumerate(aggs):
            col = agg
            scaled.append(col)                                # identity
            scaled.append(col * log_deg)                     # amplification
            scaled.append(col / (log_deg + 1e-8))            # attenuation

        return torch.cat(scaled, dim=-1)   # [N, 12·D]

    def message(self, x_j, edge_feat):
        return F.silu(x_j + edge_feat)


class PNAEncoder(nn.Module):
    """Encodeur PNA complet."""
    def __init__(
        self,
        atom_dim: int   = ATOM_FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        edge_dim: int   = BOND_FEATURE_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float  = DROPOUT,
        delta: float    = 2.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding  = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.convs  = nn.ModuleList([
            PNAConv(hidden_dim, edge_dim, dropout, delta) for _ in range(num_layers)
        ])
        self.pool = GatedPooling(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )

    def encode_nodes(self, x, edge_index, edge_attr):
        h = self.embedding(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        return h

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.encode_nodes(x, edge_index, edge_attr)
        return self.proj(self.pool(h, batch))


# ══════════════════════════════════════════════════════════════════════
#  5. GPS – General Powerful Scalable Graph Transformer (Rampásek 2022)
# ══════════════════════════════════════════════════════════════════════

class GPSLayer(nn.Module):
    """
    Couche GPS = MPNN local + attention globale (multi-head).

    Formule :
      h' = h + MPNN(h) + MultiHeadAttn(h)   (+ gating appris)

    Référence : Rampásek et al. (2022) "Recipe for a General, Powerful,
    Scalable Graph Transformer", NeurIPS 2022.
    """
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int  = 8,
        dropout: float  = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float    = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        # ── MPNN local ──────────────────────────────────────────────
        self.local_conv = MPNNConv(hidden_dim, edge_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_drop = nn.Dropout(dropout)

        # ── Attention globale ────────────────────────────────────────
        self.attn      = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # ── Feed-forward ─────────────────────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ── Gating (contrôle l'influence locale vs globale) ──────────
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, 2), nn.Softmax(dim=-1))

        # ── Stochastic depth (drop path) ─────────────────────────────
        self.drop_path_prob = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_prob == 0.0:
            return x
        keep = torch.rand(x.size(0), 1, device=x.device) > self.drop_path_prob
        return x * keep.float()

    def forward(self, x, edge_index, edge_attr, batch):
        # ── Partie locale (MPNN) ────────────────────────────────────
        h_local = self.local_norm(self.local_drop(self.local_conv(x, edge_index, edge_attr)))

        # ── Partie globale (attention) ───────────────────────────────
        # Regrouper les nœuds de chaque graphe pour l'attention
        # (padding nécessaire car les graphes ont des tailles différentes)
        B = int(batch.max().item()) + 1
        max_n = int((batch == torch.arange(B, device=batch.device).unsqueeze(-1)).sum(-1).max())

        h_pad   = torch.zeros(B, max_n, x.size(-1), device=x.device)
        attn_mask = torch.ones(B, max_n, dtype=torch.bool, device=x.device)

        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n   = idx.size(0)
            h_pad[g, :n]    = x[idx]
            attn_mask[g, :n] = False   # False = pas masqué

        attn_out, _ = self.attn(h_pad, h_pad, h_pad, key_padding_mask=attn_mask)

        # Reconstruire le format [N, D]
        h_global = torch.zeros_like(x)
        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=True)[0]
            n   = idx.size(0)
            h_global[idx] = attn_out[g, :n]

        h_global = self.attn_norm(x + self.attn_drop(h_global))

        # ── Gating local vs global ───────────────────────────────────
        g_weights = self.gate(torch.cat([h_local, h_global], dim=-1))  # [N, 2]
        h = g_weights[:, 0:1] * h_local + g_weights[:, 1:2] * h_global

        # ── FFN + drop path ──────────────────────────────────────────
        h = self.ffn_norm(h + self._drop_path(self.ffn(h)))
        return h


class GPSEncoder(nn.Module):
    """Encodeur GPS complet."""
    def __init__(
        self,
        atom_dim: int       = ATOM_FEATURE_DIM,
        hidden_dim: int     = HIDDEN_DIM,
        num_layers: int     = NUM_LAYERS,
        edge_dim: int       = BOND_FEATURE_DIM,
        output_dim: int     = OUTPUT_DIM,
        num_heads: int      = 8,
        dropout: float      = DROPOUT,
        drop_path: float    = 0.10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            GPSLayer(hidden_dim, edge_dim, num_heads, dropout, drop_path=drop_path)
            for _ in range(num_layers)
        ])
        self.pool = GatedPooling(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
            nn.SiLU(), nn.Dropout(dropout),
        )

    def encode_nodes(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)
        return h

    def forward(self, x, edge_index, edge_attr, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.encode_nodes(x, edge_index, edge_attr, batch)
        return self.proj(self.pool(h, batch))


# ══════════════════════════════════════════════════════════════════════
#  FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════════════

def build_encoder(
    arch: str       = "attfp",
    atom_dim: int   = ATOM_FEATURE_DIM,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    edge_dim: int   = BOND_FEATURE_DIM,
    output_dim: int = OUTPUT_DIM,
    dropout: float  = DROPOUT,
    **kwargs,
) -> nn.Module:
    """
    Factory : construit l'encodeur GNN choisi.

    Args:
        arch: 'mpnn' | 'attfp' | 'gin' | 'pna' | 'gps'

    Returns:
        Encodeur avec interface : encoder(x, edge_index, edge_attr, batch) → [B, output_dim]
    """
    common = dict(
        atom_dim=atom_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, edge_dim=edge_dim,
        output_dim=output_dim, dropout=dropout,
    )
    arch = arch.lower()

    if arch == "mpnn":
        return MPNNEncoder(**common)
    elif arch in ("attfp", "attentivefp"):
        return AttFPEncoder(**common, mol_layers=kwargs.get("mol_layers", 3))
    elif arch == "gin":
        return GINEncoder(**common)
    elif arch == "pna":
        return PNAEncoder(**common, delta=kwargs.get("delta", 2.5))
    elif arch == "gps":
        return GPSEncoder(
            **common,
            num_heads=kwargs.get("num_heads", 8),
            drop_path=kwargs.get("drop_path", 0.10),
        )
    else:
        raise ValueError(
            f"Architecture inconnue : '{arch}'. Choisir parmi : mpnn, attfp, gin, pna, gps"
        )


def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
