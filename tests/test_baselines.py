"""Tests for baseline models."""

import numpy as np
import pytest

from baselines import (
    aa_composition,
    dipeptide_composition,
    physicochemical_features,
    global_features,
    peptide_features,
)


class TestAAComposition:
    def test_single_aa(self):
        comp = aa_composition("A")
        assert comp.shape == (20,)
        assert abs(comp.sum() - 1.0) < 1e-6

    def test_homopolymer(self):
        comp = aa_composition("AAAA")
        assert comp[0] == 1.0  # A is first in sorted order
        assert comp[1:].sum() == 0.0

    def test_empty(self):
        comp = aa_composition("")
        assert comp.shape == (20,)


class TestDipeptideComposition:
    def test_shape(self):
        dp = dipeptide_composition("ACDEF")
        assert dp.shape == (400,)

    def test_single_aa_all_zeros(self):
        dp = dipeptide_composition("A")
        assert dp.sum() == 0.0  # no dipeptides in single AA


class TestPeptideFeatures:
    def test_full_feature_shape(self):
        feat = peptide_features("LIWKL")
        assert feat.shape == (460,)  # 20 + 400 + 32 + 8

    def test_different_sequences_different_features(self):
        f1 = peptide_features("AAAA")
        f2 = peptide_features("WWWW")
        assert not np.allclose(f1, f2)


class TestGlobalFeatures:
    def test_shape(self):
        gf = global_features("LIWKL")
        assert gf.shape == (8,)

    def test_length_correct(self):
        gf = global_features("LIWKL")
        assert gf[0] == 5  # first element is length
