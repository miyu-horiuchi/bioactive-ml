"""Tests for developability property scoring module."""

import pytest

from properties import score_peptide


# ── Score range validation ──────────────────────────────────────

SCORE_KEYS = {"toxicity", "hemolysis", "solubility", "permeability", "stability", "bitterness"}


class TestScoreRanges:
    def test_all_scores_between_0_and_1(self):
        result = score_peptide("LIWKL")
        for key in SCORE_KEYS:
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0, 1]"

    def test_expected_keys(self):
        result = score_peptide("LIWKL")
        assert SCORE_KEYS.issubset(set(result.keys()))

    def test_developability_score_between_0_and_1(self):
        result = score_peptide("LIWKL")
        assert 0.0 <= result["developability"] <= 1.0

    def test_score_values_are_numeric(self):
        result = score_peptide("LIWKL")
        for key in SCORE_KEYS:
            assert isinstance(result[key], (int, float)), f"{key} is {type(result[key])}"


# ── Toxicity ────────────────────────────────────────────────────

class TestToxicity:
    def test_known_safe_ipp(self):
        result = score_peptide("IPP")
        assert result["toxicity"] < 0.3, "IPP should have low toxicity"

    def test_known_safe_vpp(self):
        result = score_peptide("VPP")
        assert result["toxicity"] < 0.3, "VPP should have low toxicity"


# ── Solubility ──────────────────────────────────────────────────

class TestSolubility:
    def test_hydrophilic_high_solubility(self):
        result = score_peptide("DDEEK")
        assert result["solubility"] > 0.6, "hydrophilic peptide should be soluble"

    def test_hydrophobic_low_solubility(self):
        result = score_peptide("FFFFF")
        assert result["solubility"] < 0.4, "hydrophobic peptide should have low solubility"

    def test_hydrophilic_more_soluble_than_hydrophobic(self):
        hydrophilic = score_peptide("DDEEK")
        hydrophobic = score_peptide("FFFFF")
        assert hydrophilic["solubility"] > hydrophobic["solubility"]


# ── Permeability ────────────────────────────────────────────────

class TestPermeability:
    def test_short_better_permeability_than_long(self):
        short = score_peptide("IPP")
        long = score_peptide("IPPVPPLIWKLDDEEK")
        assert short["permeability"] > long["permeability"], (
            "short peptides should permeate better than long ones"
        )


# ── Stability ───────────────────────────────────────────────────

class TestStability:
    def test_protease_site_kr_lowers_stability(self):
        with_site = score_peptide("AKRGL")
        without_site = score_peptide("AAGGL")
        assert with_site["stability"] < without_site["stability"], (
            "KR protease site should lower stability"
        )

    def test_protease_site_rr_lowers_stability(self):
        with_site = score_peptide("ARRGL")
        without_site = score_peptide("AAGGL")
        assert with_site["stability"] < without_site["stability"], (
            "RR protease site should lower stability"
        )


# ── Bitterness ──────────────────────────────────────────────────

class TestBitterness:
    def test_hydrophobic_peptide_is_bitter(self):
        result = score_peptide("FFFFF")
        assert result["bitterness"] > 0.5, (
            "highly hydrophobic peptide should score high on bitterness"
        )

    def test_hydrophilic_peptide_less_bitter(self):
        hydrophilic = score_peptide("DDEEK")
        hydrophobic = score_peptide("FFFFF")
        assert hydrophilic["bitterness"] < hydrophobic["bitterness"]


# ── Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_sequence(self):
        with pytest.raises(ValueError):
            score_peptide("")

    def test_single_residue(self):
        result = score_peptide("A")
        assert isinstance(result, dict)
        assert SCORE_KEYS.issubset(set(result.keys()))
