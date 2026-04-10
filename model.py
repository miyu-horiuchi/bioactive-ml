"""
Meal Shield GNN — Model Architecture
Multi-task Graph Attention Network for predicting peptide/molecule
bioactivity across meal-shield targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, global_max_pool, global_add_pool


class MealShieldGNN(nn.Module):
    """
    Multi-task GNN for predicting bioactivity against meal-shield targets.

    Architecture:
    - Graph Attention Network (GAT) layers for message passing
    - Mean + Max pooling for graph-level readout
    - Shared trunk for multi-task learning
    - Task-specific prediction heads

    Handles both small molecules (11 atom features) and peptides (8 residue features)
    via an input projection layer.
    """

    def __init__(
        self,
        node_feature_dim=11,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        target_names=None,
    ):
        super().__init__()

        if target_names is None:
            target_names = [
                "alpha_glucosidase",
                "lipase",
                "bile_acid_receptor",
                "sodium_hydrogen_exchanger",
                "ace_inhibitor",
                "dpp4_inhibitor",
            ]

        self.target_names = target_names

        # Input projection (handles different feature dimensions)
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Graph-level readout dimension (mean + max pooling)
        graph_dim = hidden_dim * 2

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(graph_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict()
        for name in target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project input features to hidden dimension
        x = self.input_proj(x)
        x = F.relu(x)

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + residual  # Residual connection

        # Graph-level readout
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Shared representation
        shared = self.shared(x)

        # Task-specific predictions
        predictions = {}
        for name, head in self.heads.items():
            predictions[name] = head(shared).squeeze(-1)

        return predictions

    def get_graph_embedding(self, data):
        """Get the graph-level embedding (for visualization, clustering)."""
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_proj(x)
        x = F.relu(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + residual

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.shared(x)


class MealShieldGNN_TDA(nn.Module):
    """
    GNN + Persistent Homology (TDA) combined model.

    Architecture:
        ┌─────────────────┐     ┌──────────────────┐
        │   GNN Branch    │     │   TDA Branch     │
        │  (GAT layers)   │     │  (MLP on topo    │
        │                 │     │   features)      │
        └────────┬────────┘     └────────┬─────────┘
                 │                       │
                 └───────────┬───────────┘
                             │
                       CONCATENATE
                             │
                      Shared trunk
                             │
                   Task-specific heads

    The GNN captures local chemistry (atoms, bonds, neighborhoods).
    The TDA branch captures global topology (rings, cavities, voids).
    Together they provide a richer molecular representation.
    """

    def __init__(
        self,
        node_feature_dim=11,
        tda_feature_dim=30,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        target_names=None,
    ):
        super().__init__()

        if target_names is None:
            target_names = [
                "alpha_glucosidase",
                "lipase",
                "bile_acid_receptor",
                "sodium_hydrogen_exchanger",
                "ace_inhibitor",
                "dpp4_inhibitor",
            ]

        self.target_names = target_names
        self.tda_feature_dim = tda_feature_dim

        # --- GNN Branch ---
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        gnn_out_dim = hidden_dim * 2  # mean + max pooling

        # --- TDA Branch ---
        self.tda_branch = nn.Sequential(
            nn.Linear(tda_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        tda_out_dim = 64

        # --- Fusion ---
        combined_dim = gnn_out_dim + tda_out_dim

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict()
        for name in target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )

        # --- GNN branch ---
        x = self.input_proj(x)
        x = F.relu(x)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + residual

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        gnn_out = torch.cat([x_mean, x_max], dim=1)

        # --- TDA branch ---
        tda_features = data.tda  # Shape: [batch_size, tda_feature_dim]
        tda_out = self.tda_branch(tda_features)

        # --- Fusion ---
        combined = torch.cat([gnn_out, tda_out], dim=1)
        shared = self.shared(combined)

        # Task-specific predictions
        predictions = {}
        for name, head in self.heads.items():
            predictions[name] = head(shared).squeeze(-1)

        return predictions


class MealShieldGIN(nn.Module):
    """
    Alternative architecture using Graph Isomorphism Network (GIN).
    GIN is theoretically more powerful than GAT for distinguishing
    different graph structures (Weisfeiler-Leman test).
    """

    def __init__(
        self,
        node_feature_dim=11,
        hidden_dim=128,
        num_layers=4,
        dropout=0.2,
        target_names=None,
    ):
        super().__init__()

        if target_names is None:
            target_names = [
                "alpha_glucosidase",
                "lipase",
                "bile_acid_receptor",
                "sodium_hydrogen_exchanger",
                "ace_inhibitor",
                "dpp4_inhibitor",
            ]

        self.target_names = target_names
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # JK (Jumping Knowledge) — concatenate all layer outputs
        graph_dim = hidden_dim * num_layers

        self.shared = nn.Sequential(
            nn.Linear(graph_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict()
        for name in target_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_proj(x)
        x = F.relu(x)

        # Collect outputs from all layers (Jumping Knowledge)
        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            layer_outputs.append(global_add_pool(x, batch))

        # Concatenate all layer readouts
        x = torch.cat(layer_outputs, dim=1)

        shared = self.shared(x)

        predictions = {}
        for name, head in self.heads.items():
            predictions[name] = head(shared).squeeze(-1)

        return predictions
