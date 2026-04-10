"""Tests for training utilities."""

import torch
import numpy as np
import pytest

from train import (
    pad_features,
    collate_mixed,
    compute_metrics,
)
from torch_geometric.data import Data


class TestPadFeatures:
    def test_pad_smaller(self):
        g = Data(x=torch.randn(3, 8))
        g = pad_features(g, 11)
        assert g.x.shape == (3, 11)

    def test_truncate_larger(self):
        g = Data(x=torch.randn(3, 328))
        g = pad_features(g, 11)
        assert g.x.shape == (3, 11)

    def test_exact_match(self):
        g = Data(x=torch.randn(3, 11))
        g = pad_features(g, 11)
        assert g.x.shape == (3, 11)

    def test_target_idx_becomes_tensor(self):
        g = Data(x=torch.randn(3, 8))
        g.target_idx = 2
        g = pad_features(g, 8)
        assert isinstance(g.target_idx, torch.Tensor)
        assert g.target_idx.item() == 2


class TestCollateMixed:
    def test_pads_all_graphs(self):
        graphs = [
            Data(x=torch.randn(3, 8)),
            Data(x=torch.randn(5, 11)),
        ]
        result = collate_mixed(graphs, 11)
        assert all(g.x.shape[1] == 11 for g in result)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        m = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m["R2"] == 1.0
        assert m["RMSE"] == 0.0

    def test_bad_prediction(self):
        m = compute_metrics([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert m["R2"] < 0.5

    def test_too_few_samples(self):
        m = compute_metrics([1.0], [1.0])
        assert m["R2"] is None
        assert m["RMSE"] is None
        assert m["n"] == 1
