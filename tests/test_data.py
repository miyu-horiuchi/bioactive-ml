"""Tests for the data pipeline."""

import torch
import numpy as np
import pandas as pd
import pytest

from data import (
    _convert_to_nM,
    load_food_peptides,
    peptide_to_graph,
    _peptide_residue_graph,
    AA_FEATURES,
    TARGETS,
)


# ── Unit conversion ──────────────────────────────────────────

class TestConvertToNM:
    def test_nM_passthrough(self):
        assert _convert_to_nM(100.0, "nM") == 100.0

    def test_uM_to_nM(self):
        assert _convert_to_nM(5.0, "uM") == 5000.0

    def test_mM_to_nM(self):
        assert _convert_to_nM(1.0, "mM") == 1e6

    def test_pM_to_nM(self):
        assert _convert_to_nM(500.0, "pM") == 0.5

    def test_molar_to_nM(self):
        assert _convert_to_nM(1e-9, "M") == 1.0

    def test_unknown_units_returns_none(self):
        assert _convert_to_nM(10.0, "ug/mL") is None
        assert _convert_to_nM(10.0, "ug.mL-1") is None

    def test_unrecognized_units_returns_none(self):
        assert _convert_to_nM(10.0, "banana") is None


# ── Targets dict ─────────────────────────────────────────────

class TestTargets:
    def test_has_six_targets(self):
        assert len(TARGETS) == 6

    def test_ace_inhibitor_present(self):
        assert "ace_inhibitor" in TARGETS

    def test_dpp4_inhibitor_present(self):
        assert "dpp4_inhibitor" in TARGETS

    def test_each_target_has_chembl_id(self):
        for name, info in TARGETS.items():
            assert "chembl_id" in info, f"{name} missing chembl_id"
            assert info["chembl_id"].startswith("CHEMBL"), f"{name} bad chembl_id"


# ── Food peptide loading ─────────────────────────────────────

class TestLoadFoodPeptides:
    def test_loads_peptides(self):
        peptides = load_food_peptides()
        assert len(peptides) > 0

    def test_peptide_has_required_fields(self):
        peptides = load_food_peptides()
        for p in peptides[:5]:
            assert "sequence" in p
            assert "target" in p
            assert "pIC50" in p

    def test_includes_inactive_by_default(self):
        peptides = load_food_peptides()
        targets = {p["target"] for p in peptides}
        assert "inactive" in targets

    def test_can_exclude_inactive(self):
        peptides = load_food_peptides(include_inactive=False)
        targets = {p["target"] for p in peptides}
        assert "inactive" not in targets

    def test_missing_file_returns_empty(self):
        peptides = load_food_peptides("/nonexistent/path.csv")
        assert peptides == []


# ── Graph construction ───────────────────────────────────────

class TestPeptideGraph:
    def test_basic_graph(self):
        g = peptide_to_graph("LIWKL", use_residue_level=True)
        assert g is not None
        assert g.x.shape[0] == 5  # 5 residues
        assert g.x.shape[1] == 8  # 8 physicochemical features
        assert g.edge_index.shape[0] == 2

    def test_single_residue(self):
        g = peptide_to_graph("A", use_residue_level=True)
        assert g is not None
        assert g.x.shape[0] == 1

    def test_invalid_residue_returns_none(self):
        g = peptide_to_graph("LIXKL", use_residue_level=True)
        assert g is None  # X not in AA_FEATURES

    def test_all_standard_aa(self):
        """Every standard amino acid should produce a valid graph."""
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            g = peptide_to_graph(aa, use_residue_level=True)
            assert g is not None, f"Failed for {aa}"

    def test_esm_concatenation(self):
        """When ESM cache is provided, features should be 8 + 320 = 328."""
        seq = "LIWKL"
        fake_esm = {seq: torch.randn(5, 320)}
        g = _peptide_residue_graph(seq, esm_cache=fake_esm)
        assert g is not None
        assert g.x.shape == (5, 328)

    def test_esm_missing_sequence_falls_back(self):
        """If sequence not in ESM cache, use 8-dim features only."""
        g = _peptide_residue_graph("LIWKL", esm_cache={})
        assert g is not None
        assert g.x.shape[1] == 8

    def test_edge_connectivity(self):
        """Graph should have sequential + skip edges."""
        g = peptide_to_graph("ABCDE"[0:5].replace("B", "A").replace("C", "G").replace("D", "L").replace("E", "K"),
                             use_residue_level=True)
        # For 4+ residues: sequential (n-1)*2 + skip edges
        assert g.edge_index.shape[1] > 0


# ── AA Features completeness ────────────────────────────────

class TestAAFeatures:
    def test_all_20_amino_acids(self):
        assert len(AA_FEATURES) == 20

    def test_feature_dimensions(self):
        for aa, feats in AA_FEATURES.items():
            assert len(feats) == 8, f"{aa} has {len(feats)} features, expected 8"

    def test_proline_is_rigid(self):
        assert AA_FEATURES["P"][-1] == 1  # rigidity flag
