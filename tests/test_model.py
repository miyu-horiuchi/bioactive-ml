"""Tests for model architecture."""

import sys
import torch
import pytest

from model import MealShieldGNN, MealShieldGNN_TDA, MealShieldGIN
from torch_geometric.data import Data, Batch

# GAT scatter_reduce segfaults in pytest on macOS + Python 3.9.
# Forward-pass tests are safe on Linux (CI).
_skip_forward = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="torch-geometric GAT segfaults in pytest on macOS"
)


def _make_graph(n_nodes, feat_dim, **extra):
    """Helper to build a simple test graph."""
    edges = []
    for i in range(n_nodes - 1):
        edges.extend([[i, i + 1], [i + 1, i]])
    if not edges:
        edges = [[0, 0]]
    return Data(
        x=torch.randn(n_nodes, feat_dim),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        **extra,
    )


class TestMealShieldGNN:
    def test_default_targets(self):
        model = MealShieldGNN()
        assert len(model.target_names) == 6
        assert "ace_inhibitor" in model.target_names
        assert "dpp4_inhibitor" in model.target_names

    @_skip_forward
    def test_forward_11dim(self):
        """11-dim node features (atom-level small molecule graph)."""
        model = MealShieldGNN(node_feature_dim=11, hidden_dim=64)
        model.eval()
        batch = Batch.from_data_list([_make_graph(10, 11)])
        with torch.no_grad():
            preds = model(batch)
        assert isinstance(preds, dict)
        for t in model.target_names:
            assert t in preds
            assert preds[t].shape == (1,)

    @_skip_forward
    def test_forward_8dim(self):
        """8-dim node features (residue-level, no ESM)."""
        model = MealShieldGNN(node_feature_dim=8, hidden_dim=64)
        model.eval()
        batch = Batch.from_data_list([_make_graph(5, 8)])
        with torch.no_grad():
            preds = model(batch)
        assert all(t in preds for t in model.target_names)

    @_skip_forward
    def test_forward_328dim(self):
        """328-dim node features (residue-level + ESM-2)."""
        model = MealShieldGNN(node_feature_dim=328, hidden_dim=64)
        model.eval()
        batch = Batch.from_data_list([_make_graph(5, 328)])
        with torch.no_grad():
            preds = model(batch)
        assert all(t in preds for t in model.target_names)

    @_skip_forward
    def test_batch_of_two(self):
        """Model handles batched graphs."""
        model = MealShieldGNN(node_feature_dim=11, hidden_dim=64)
        model.eval()
        batch = Batch.from_data_list([_make_graph(6, 11), _make_graph(8, 11)])
        with torch.no_grad():
            preds = model(batch)
        for t in model.target_names:
            assert preds[t].shape == (2,)

    def test_custom_targets(self):
        targets = ["target_a", "target_b"]
        model = MealShieldGNN(target_names=targets)
        assert model.target_names == targets

    def test_parameter_count_reasonable(self):
        model = MealShieldGNN(node_feature_dim=11, hidden_dim=128)
        n_params = sum(p.numel() for p in model.parameters())
        assert 10_000 < n_params < 5_000_000


class TestMealShieldGNNTDA:
    @_skip_forward
    def test_forward_with_tda(self):
        model = MealShieldGNN_TDA(
            node_feature_dim=11, tda_feature_dim=42, hidden_dim=64
        )
        model.eval()
        g = _make_graph(10, 11, tda=torch.randn(1, 42))
        batch = Batch.from_data_list([g])
        with torch.no_grad():
            preds = model(batch)
        assert isinstance(preds, dict)
        assert len(preds) == 6


class TestMealShieldGIN:
    @_skip_forward
    def test_forward(self):
        model = MealShieldGIN(node_feature_dim=11, hidden_dim=64)
        model.eval()
        batch = Batch.from_data_list([_make_graph(8, 11)])
        with torch.no_grad():
            preds = model(batch)
        assert isinstance(preds, dict)
