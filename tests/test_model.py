"""Tests for model architecture."""

import torch
import pytest

from model import MealShieldGNN, MealShieldGNN_TDA, MealShieldGIN


class TestMealShieldGNN:
    def test_default_targets(self):
        model = MealShieldGNN()
        assert len(model.target_names) == 6
        assert "ace_inhibitor" in model.target_names
        assert "dpp4_inhibitor" in model.target_names

    def test_forward_small_molecule(self):
        """11-dim node features (atom-level graph)."""
        model = MealShieldGNN(node_feature_dim=11, hidden_dim=64)
        from torch_geometric.data import Data, Batch

        g = Data(
            x=torch.randn(10, 11),
            edge_index=torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long),
            y=torch.tensor([5.0]),
        )
        batch = Batch.from_data_list([g])
        preds = model(batch)
        assert isinstance(preds, dict)
        for target in model.target_names:
            assert target in preds
            assert preds[target].shape == (1,)

    def test_forward_peptide_8dim(self):
        """8-dim node features (residue-level, no ESM)."""
        model = MealShieldGNN(node_feature_dim=8, hidden_dim=64)
        from torch_geometric.data import Data, Batch

        g = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0,1,2,3,1], [1,2,3,4,3]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g])
        preds = model(batch)
        assert all(t in preds for t in model.target_names)

    def test_forward_peptide_328dim(self):
        """328-dim node features (residue-level + ESM-2)."""
        model = MealShieldGNN(node_feature_dim=328, hidden_dim=64)
        from torch_geometric.data import Data, Batch

        g = Data(
            x=torch.randn(5, 328),
            edge_index=torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g])
        preds = model(batch)
        assert all(t in preds for t in model.target_names)

    def test_custom_targets(self):
        targets = ["target_a", "target_b"]
        model = MealShieldGNN(target_names=targets)
        assert model.target_names == targets

    def test_parameter_count_reasonable(self):
        model = MealShieldGNN(node_feature_dim=11, hidden_dim=128)
        n_params = sum(p.numel() for p in model.parameters())
        assert 10_000 < n_params < 5_000_000


class TestMealShieldGNNTDA:
    def test_forward_with_tda(self):
        model = MealShieldGNN_TDA(
            node_feature_dim=11, tda_feature_dim=42, hidden_dim=64
        )
        from torch_geometric.data import Data, Batch

        g = Data(
            x=torch.randn(10, 11),
            edge_index=torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long),
            tda=torch.randn(1, 42),
        )
        batch = Batch.from_data_list([g])
        preds = model(batch)
        assert isinstance(preds, dict)
        assert len(preds) == 6


class TestMealShieldGIN:
    def test_forward(self):
        model = MealShieldGIN(node_feature_dim=11, hidden_dim=64)
        from torch_geometric.data import Data, Batch

        g = Data(
            x=torch.randn(8, 11),
            edge_index=torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g])
        preds = model(batch)
        assert isinstance(preds, dict)
