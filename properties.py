"""
Meal Shield — Peptide Developability Property Predictor

Scores peptides on safety and practical properties beyond bioactivity.
Implements rule-based / heuristic scoring for toxicity, hemolysis risk,
solubility, membrane permeability, stability, bitterness, and an
overall developability score.

Reference:
    Hong et al. (2026) "AI-Designed Peptides as Tools for Biochemistry"

Usage:
    python properties.py --sequence LIWKL
    python properties.py --csv data/food_peptides.csv
    python properties.py --csv data/food_peptides.csv --min-solubility 0.6 --max-toxicity 0.2
"""

import argparse
import csv
import math
import sys
from typing import Dict, List, Optional, Tuple

# ============================================================
# Amino acid physicochemical features
# Columns: MW, hydrophobicity (Kyte-Doolittle), charge, aromatic,
#          H-bond donors, volume, pI, rigidity
# ============================================================

AA_FEATURES = {
    'A': [89.1,  1.8,  0.0,  0, 0, 71.8,  6.01, 0],
    'R': [174.2, -4.5,  1.0,  0, 5, 148.0, 10.76, 0],
    'N': [132.1, -3.5,  0.0,  0, 2, 114.0, 5.41, 0],
    'D': [133.1, -3.5, -1.0,  0, 1, 111.0, 2.77, 1],
    'C': [121.2,  2.5,  0.0,  0, 1, 108.5, 5.07, 0],
    'E': [147.1, -3.5, -1.0,  0, 1, 138.4, 3.22, 1],
    'Q': [146.2, -3.5,  0.0,  0, 2, 143.8, 5.65, 0],
    'G': [75.0,  -0.4,  0.0,  0, 0, 60.1,  5.97, 0],
    'H': [155.2, -3.2,  0.0,  1, 2, 153.2, 7.59, 0],
    'I': [131.2,  4.5,  0.0,  0, 0, 166.7, 6.02, 0],
    'L': [131.2,  3.8,  0.0,  0, 0, 166.7, 5.98, 0],
    'K': [146.2, -3.9,  1.0,  0, 2, 168.6, 9.74, 0],
    'M': [149.2,  1.9,  0.0,  0, 0, 162.9, 5.74, 0],
    'F': [165.2,  2.8,  0.0,  1, 0, 189.9, 5.48, 0],
    'P': [115.1, -1.6,  0.0,  0, 0, 129.0, 6.30, 1],
    'S': [105.1, -0.8,  0.0,  0, 1, 89.0,  5.68, 0],
    'T': [119.1, -0.7,  0.0,  0, 1, 116.1, 5.60, 0],
    'W': [204.2, -0.9,  0.0,  1, 1, 227.8, 5.89, 0],
    'Y': [181.2, -1.3,  0.0,  1, 1, 193.6, 5.66, 0],
    'V': [117.1,  4.2,  0.0,  0, 0, 140.0, 5.97, 0],
}

# Feature column indices for readability
_MW = 0
_HYDRO = 1
_CHARGE = 2
_AROMATIC = 3
_HBOND = 4
_VOLUME = 5
_PI = 6
_RIGID = 7

VALID_AA = set(AA_FEATURES.keys())

# Hydrophobic residues for stretch detection and amphipathicity
HYDROPHOBIC_AA = set("AILMFVWP")

# ============================================================
# Ney's Q-rule hydrophobicity values (cal/mol) for bitterness
# Based on Tanford (1962) transfer free energies
# ============================================================

NEY_HYDROPHOBICITY = {
    'A':  730, 'R':  -730, 'N': -100, 'D':  -100,
    'C': 1000, 'E':  -100, 'Q': -100, 'G':     0,
    'H':  580, 'I': 2970, 'L': 2420, 'K': -1500,
    'M': 1300, 'F': 2650, 'P': 2620, 'S':  -300,
    'T':  -400, 'W': 3000, 'Y': 2870, 'V': 1690,
}

# Known bitter dipeptide motifs (common in food peptide literature)
BITTER_DIPEPTIDES = {
    "LL", "LF", "FL", "FF", "LV", "VL", "FV", "VF",
    "IL", "LI", "IF", "FI", "IV", "VI", "PF", "FP",
    "PL", "LP", "PI", "IP", "PV", "VP", "WL", "LW",
    "WF", "FW", "WV", "VW",
}

# ============================================================
# Toxic motifs and antimicrobial cationic motifs
# ============================================================

TOXIC_MOTIFS = [
    "RGD",   # Cell-adhesion / integrin-binding
    "KKK",   # Cationic antimicrobial motif
    "RRR",   # Cationic antimicrobial motif
    "KRK",   # Cationic antimicrobial motif
    "RKR",   # Cationic antimicrobial motif
    "KKKK",  # Highly cationic stretch
    "RRRR",  # Highly cationic stretch
    "KGD",   # Cell-adhesion variant
    "NGR",   # Tumor-homing motif
]

# ============================================================
# Protease cleavage motifs for stability scoring
# ============================================================

# Trypsin cleaves after K or R (P1 site)
TRYPSIN_SITES = {"KR", "RR", "KK", "RK"}

# Chymotrypsin cleaves after F, W, Y (aromatic at P1)
CHYMOTRYPSIN_P1 = set("FWY")


# ============================================================
# Helper functions
# ============================================================

def _validate_sequence(seq: str) -> str:
    """Uppercase and validate a peptide sequence. Returns cleaned sequence."""
    seq = seq.strip().upper()
    invalid = set(seq) - VALID_AA
    if invalid:
        raise ValueError(
            f"Invalid amino acid(s) in sequence '{seq}': {sorted(invalid)}"
        )
    if len(seq) == 0:
        raise ValueError("Empty sequence")
    return seq


def _get_features(seq: str) -> List[List[float]]:
    """Return per-residue feature vectors for a sequence."""
    return [AA_FEATURES[aa] for aa in seq]


def _net_charge(seq: str) -> float:
    """Calculate net charge at physiological pH (approx)."""
    return sum(AA_FEATURES[aa][_CHARGE] for aa in seq)


def _molecular_weight(seq: str) -> float:
    """Estimate MW of peptide (sum of residue MWs minus water losses)."""
    if len(seq) == 0:
        return 0.0
    residue_sum = sum(AA_FEATURES[aa][_MW] for aa in seq)
    # Subtract water for each peptide bond
    water_loss = 18.015 * (len(seq) - 1)
    return residue_sum - water_loss


def _gravy(seq: str) -> float:
    """Grand Average of Hydropathy (Kyte-Doolittle)."""
    if len(seq) == 0:
        return 0.0
    return sum(AA_FEATURES[aa][_HYDRO] for aa in seq) / len(seq)


def _hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """
    Estimate hydrophobic moment assuming an ideal alpha-helix.

    Uses Eisenberg's method: project hydrophobicity onto a helical wheel
    with the given angle between residues (100 degrees for alpha-helix).

    Returns the magnitude of the hydrophobic moment vector.
    """
    if len(seq) < 2:
        return 0.0

    angle_rad = math.radians(angle)
    sum_sin = 0.0
    sum_cos = 0.0
    for i, aa in enumerate(seq):
        h = AA_FEATURES[aa][_HYDRO]
        theta = i * angle_rad
        sum_sin += h * math.sin(theta)
        sum_cos += h * math.cos(theta)

    return math.sqrt(sum_sin ** 2 + sum_cos ** 2) / len(seq)


def _longest_hydrophobic_stretch(seq: str) -> int:
    """Find the longest consecutive run of hydrophobic residues."""
    max_run = 0
    current_run = 0
    for aa in seq:
        if aa in HYDROPHOBIC_AA:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _hydrophobic_fraction(seq: str) -> float:
    """Fraction of residues that are hydrophobic."""
    if len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in HYDROPHOBIC_AA) / len(seq)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================
# Property scoring functions
# ============================================================

def score_toxicity(seq: str) -> Tuple[float, List[str]]:
    """
    Rule-based toxicity score. 0 = safe, 1 = toxic.

    Flags:
    - Known toxic / antimicrobial motifs
    - High cationic charge density (>+3 net charge for short peptides)
    - High hydrophobic moment (amphipathic antimicrobial character)

    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    penalty = 0.0
    flags = []

    # Check toxic motifs
    for motif in TOXIC_MOTIFS:
        if motif in seq:
            penalty += 0.3
            flags.append(f"toxic_motif:{motif}")

    # Cationic charge density
    charge = _net_charge(seq)
    if len(seq) <= 10 and charge > 3:
        penalty += 0.3
        flags.append(f"high_cationic_charge:{charge:+.0f}")
    elif len(seq) <= 20 and charge > 5:
        penalty += 0.25
        flags.append(f"high_cationic_charge:{charge:+.0f}")
    elif charge > 0:
        # Mild penalty for high charge density
        charge_density = charge / len(seq)
        if charge_density > 0.4:
            penalty += 0.2
            flags.append(f"cationic_density:{charge_density:.2f}")

    # High hydrophobic moment (amphipathic character)
    hm = _hydrophobic_moment(seq)
    if hm > 2.5:
        penalty += 0.2
        flags.append(f"high_hydrophobic_moment:{hm:.2f}")
    elif hm > 1.5:
        penalty += 0.1
        flags.append(f"moderate_hydrophobic_moment:{hm:.2f}")

    return (_clamp(penalty), flags)


def score_hemolysis(seq: str) -> Tuple[float, List[str]]:
    """
    Hemolysis risk based on amphipathicity and hydrophobic content.

    Helical amphipathic peptides with high hydrophobic moment and
    significant hydrophobic content tend to lyse red blood cells.

    Score 0 = safe, 1 = high hemolysis risk.
    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    penalty = 0.0
    flags = []

    hm = _hydrophobic_moment(seq)
    hfrac = _hydrophobic_fraction(seq)
    charge = _net_charge(seq)

    # Amphipathic peptides: high hydrophobic moment + cationic
    if hm > 2.0 and charge > 1:
        penalty += 0.35
        flags.append(f"amphipathic_cationic:HM={hm:.2f},charge={charge:+.0f}")

    # High hydrophobic fraction
    if hfrac > 0.7:
        penalty += 0.3
        flags.append(f"high_hydrophobic_fraction:{hfrac:.2f}")
    elif hfrac > 0.5:
        penalty += 0.15
        flags.append(f"moderate_hydrophobic_fraction:{hfrac:.2f}")

    # Hydrophobic moment alone
    if hm > 3.0:
        penalty += 0.2
        flags.append(f"very_high_hydrophobic_moment:{hm:.2f}")
    elif hm > 2.0:
        penalty += 0.1
        flags.append(f"high_hydrophobic_moment:{hm:.2f}")

    # Long peptides with helical propensity and amphipathicity
    if len(seq) >= 10 and hm > 1.5 and hfrac > 0.4:
        penalty += 0.15
        flags.append("long_amphipathic_helix_risk")

    return (_clamp(penalty), flags)


def score_solubility(seq: str) -> Tuple[float, List[str]]:
    """
    Solubility score based on GRAVY and hydrophobic stretches.

    Score 1 = highly soluble, 0 = insoluble.

    Peptides with GRAVY < 0 are generally soluble.
    Long hydrophobic stretches (>4 consecutive) reduce solubility.

    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    flags = []

    gravy = _gravy(seq)
    longest_stretch = _longest_hydrophobic_stretch(seq)

    # GRAVY-based score: map [-4.5, +4.5] to [1.0, 0.0]
    # GRAVY < 0 => soluble (score near 1.0)
    # GRAVY > 0 => less soluble (score decreasing)
    gravy_score = _clamp(1.0 - (gravy + 0.5) / 4.0)

    if gravy > 1.0:
        flags.append(f"high_GRAVY:{gravy:.2f}")
    elif gravy > 0:
        flags.append(f"moderate_GRAVY:{gravy:.2f}")

    # Hydrophobic stretch penalty
    stretch_penalty = 0.0
    if longest_stretch > 6:
        stretch_penalty = 0.3
        flags.append(f"long_hydrophobic_stretch:{longest_stretch}")
    elif longest_stretch > 4:
        stretch_penalty = 0.15
        flags.append(f"hydrophobic_stretch:{longest_stretch}")

    score = _clamp(gravy_score - stretch_penalty)

    # Charged residues boost solubility
    charged_frac = sum(1 for aa in seq if abs(AA_FEATURES[aa][_CHARGE]) > 0) / len(seq)
    if charged_frac > 0.3:
        score = _clamp(score + 0.1)
        flags.append(f"good_charged_fraction:{charged_frac:.2f}")

    return (score, flags)


def score_permeability(seq: str) -> Tuple[float, List[str]]:
    """
    Membrane permeability estimate, adapted from Lipinski's Rule of 5
    for peptides.

    Favorable: MW < 500, net charge near 0, moderate hydrophobicity.

    Score 0 = impermeable, 1 = good permeability.
    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    flags = []
    score = 1.0

    mw = _molecular_weight(seq)
    charge = _net_charge(seq)
    gravy = _gravy(seq)
    hbond_donors = sum(AA_FEATURES[aa][_HBOND] for aa in seq)

    # MW penalty: ideal < 500, strongly penalized > 800
    if mw > 800:
        score -= 0.4
        flags.append(f"high_MW:{mw:.0f}")
    elif mw > 500:
        score -= 0.15 * ((mw - 500) / 300)
        flags.append(f"moderate_MW:{mw:.0f}")

    # Charge penalty: ideal near 0
    abs_charge = abs(charge)
    if abs_charge > 3:
        score -= 0.3
        flags.append(f"high_charge:{charge:+.0f}")
    elif abs_charge > 1:
        score -= 0.1
        flags.append(f"charged:{charge:+.0f}")

    # Hydrophobicity: moderate is best (GRAVY around 0 to +2)
    if gravy < -2.0:
        score -= 0.3
        flags.append(f"too_hydrophilic:{gravy:.2f}")
    elif gravy < -0.5:
        score -= 0.1
        flags.append(f"hydrophilic:{gravy:.2f}")
    elif gravy > 3.0:
        score -= 0.2
        flags.append(f"too_hydrophobic:{gravy:.2f}")

    # H-bond donors penalty
    if hbond_donors > 5:
        score -= 0.2
        flags.append(f"many_hbond_donors:{hbond_donors}")

    return (_clamp(score), flags)


def score_stability(seq: str) -> Tuple[float, List[str]]:
    """
    Proteolytic stability estimate. Counts known protease cleavage sites.

    - Trypsin: cleaves at KR, RR, KK, RK dipeptide motifs
    - Chymotrypsin: cleaves after F, W, Y (aromatic at P1 position,
      meaning an aromatic residue followed by any residue)

    Score 1 = stable (no cleavage sites), 0 = many sites.
    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    flags = []
    cleavage_sites = 0

    # Trypsin sites: look for dipeptide motifs
    for i in range(len(seq) - 1):
        dipep = seq[i:i + 2]
        if dipep in TRYPSIN_SITES:
            cleavage_sites += 1
            flags.append(f"trypsin_site:{dipep}@{i}")

    # Chymotrypsin sites: aromatic at P1 (not at terminal position)
    for i in range(len(seq) - 1):
        if seq[i] in CHYMOTRYPSIN_P1:
            cleavage_sites += 1
            flags.append(f"chymotrypsin_site:{seq[i]}@{i}")

    # N-terminal susceptibility: peptidases commonly clip N-terminal residues
    # except pyroglutamate or proline which confer resistance
    if len(seq) > 2 and seq[0] not in ("P", "E", "Q"):
        cleavage_sites += 0.5
        flags.append(f"n_terminal_susceptible:{seq[0]}")

    # Score: each cleavage site reduces stability
    # Normalize relative to peptide length
    if len(seq) <= 1:
        return (1.0, flags)

    site_density = cleavage_sites / (len(seq) - 1)
    score = _clamp(1.0 - site_density * 1.5)

    return (score, flags)


def score_bitterness(seq: str) -> Tuple[float, List[str]]:
    """
    Bitterness prediction for food peptides using Ney's Q-rule.

    Ney (1971): peptides with average hydrophobicity Q > 1400 cal/mol
    tend to be bitter. Also flags known bitter dipeptide motifs.

    Score 0 = not bitter, 1 = very bitter.
    Returns (score, list of flags).
    """
    seq = _validate_sequence(seq)
    flags = []

    # Calculate Ney's Q value (average hydrophobicity)
    q_value = sum(NEY_HYDROPHOBICITY.get(aa, 0) for aa in seq) / len(seq)

    # Q-rule scoring
    # < 1300: generally not bitter
    # 1300-1400: borderline
    # > 1400: likely bitter
    if q_value > 1400:
        q_score = _clamp(0.5 + (q_value - 1400) / 2000)
        flags.append(f"Q_value:{q_value:.0f}_cal/mol(bitter)")
    elif q_value > 1300:
        q_score = 0.3 + 0.2 * (q_value - 1300) / 100
        flags.append(f"Q_value:{q_value:.0f}_cal/mol(borderline)")
    else:
        q_score = _clamp(q_value / 1300 * 0.3)
        flags.append(f"Q_value:{q_value:.0f}_cal/mol(not_bitter)")

    # Bitter dipeptide motif count
    bitter_count = 0
    for i in range(len(seq) - 1):
        dipep = seq[i:i + 2]
        if dipep in BITTER_DIPEPTIDES:
            bitter_count += 1
            flags.append(f"bitter_dipeptide:{dipep}@{i}")

    # Add penalty for bitter dipeptides
    if bitter_count > 0:
        dipep_penalty = min(bitter_count * 0.1, 0.3)
        q_score = _clamp(q_score + dipep_penalty)

    return (q_score, flags)


# ============================================================
# Overall developability score
# ============================================================

# Default weights for the overall developability score
DEFAULT_WEIGHTS = {
    "toxicity":       0.25,   # Safety is paramount
    "hemolysis":      0.15,   # Safety-related
    "solubility":     0.20,   # Critical for formulation
    "permeability":   0.10,   # Nice to have for oral delivery
    "stability":      0.15,   # Important for shelf life
    "bitterness":     0.15,   # Important for food peptides
}


def score_developability(
    seq: str,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute an overall developability score as a weighted combination
    of all property scores.

    For toxicity, hemolysis, and bitterness, we invert the score
    (1 - score) because lower is better for those properties.

    Returns (overall_score, dict of individual scores).
    """
    w = weights or DEFAULT_WEIGHTS
    seq = _validate_sequence(seq)

    # Call individual scorers directly (not score_peptide) to avoid recursion
    tox_score, _ = score_toxicity(seq)
    hem_score, _ = score_hemolysis(seq)
    sol_score, _ = score_solubility(seq)
    perm_score, _ = score_permeability(seq)
    stab_score, _ = score_stability(seq)
    bit_score, _ = score_bitterness(seq)

    individual = {
        "toxicity": round(tox_score, 4),
        "hemolysis": round(hem_score, 4),
        "solubility": round(sol_score, 4),
        "permeability": round(perm_score, 4),
        "stability": round(stab_score, 4),
        "bitterness": round(bit_score, 4),
    }

    # Invert scores where lower raw value = better
    # toxicity: 0=safe => contributes 1.0
    # hemolysis: 0=safe => contributes 1.0
    # bitterness: 0=not bitter => contributes 1.0
    # solubility: 1=soluble => contributes 1.0 (no inversion)
    # permeability: 1=permeable => contributes 1.0 (no inversion)
    # stability: 1=stable => contributes 1.0 (no inversion)

    weighted_sum = (
        w.get("toxicity", 0)     * (1.0 - tox_score)
        + w.get("hemolysis", 0)  * (1.0 - hem_score)
        + w.get("solubility", 0) * sol_score
        + w.get("permeability", 0) * perm_score
        + w.get("stability", 0)  * stab_score
        + w.get("bitterness", 0) * (1.0 - bit_score)
    )

    total_weight = sum(w.values())
    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    return (_clamp(overall), individual)


# ============================================================
# Main scoring API
# ============================================================

def score_peptide(sequence: str) -> Dict:
    """
    Score a single peptide on all developability properties.

    Args:
        sequence: Amino acid sequence (e.g., "LIWKL")

    Returns:
        dict with keys: sequence, toxicity, hemolysis, solubility,
        permeability, stability, bitterness, developability, flags, mw,
        net_charge, gravy, length
    """
    seq = _validate_sequence(sequence)

    tox_score, tox_flags = score_toxicity(seq)
    hem_score, hem_flags = score_hemolysis(seq)
    sol_score, sol_flags = score_solubility(seq)
    perm_score, perm_flags = score_permeability(seq)
    stab_score, stab_flags = score_stability(seq)
    bit_score, bit_flags = score_bitterness(seq)

    all_flags = tox_flags + hem_flags + sol_flags + perm_flags + stab_flags + bit_flags

    result = {
        "sequence": seq,
        "length": len(seq),
        "mw": round(_molecular_weight(seq), 1),
        "net_charge": _net_charge(seq),
        "gravy": round(_gravy(seq), 3),
        "toxicity": round(tox_score, 4),
        "hemolysis": round(hem_score, 4),
        "solubility": round(sol_score, 4),
        "permeability": round(perm_score, 4),
        "stability": round(stab_score, 4),
        "bitterness": round(bit_score, 4),
        "flags": all_flags,
    }

    # Add overall developability
    dev_score, _ = score_developability(seq)
    result["developability"] = round(dev_score, 4)

    return result


def score_peptides_batch(sequences: List[str]) -> "pd.DataFrame":
    """
    Score a batch of peptides and return results as a DataFrame.

    Args:
        sequences: List of amino acid sequences.

    Returns:
        pandas DataFrame with one row per peptide and columns for
        each property score.
    """
    import pandas as pd

    rows = []
    for seq in sequences:
        try:
            result = score_peptide(seq)
            # Convert flags list to semicolon-separated string for CSV
            result["flags"] = "; ".join(result["flags"]) if result["flags"] else ""
            rows.append(result)
        except ValueError as e:
            rows.append({
                "sequence": seq,
                "length": None,
                "mw": None,
                "net_charge": None,
                "gravy": None,
                "toxicity": None,
                "hemolysis": None,
                "solubility": None,
                "permeability": None,
                "stability": None,
                "bitterness": None,
                "developability": None,
                "flags": f"ERROR: {e}",
            })

    return pd.DataFrame(rows)


def filter_candidates(
    sequences: List[str],
    min_solubility: float = 0.5,
    max_toxicity: float = 0.3,
    max_hemolysis: float = 0.3,
    max_bitterness: float = 0.5,
    min_stability: float = 0.4,
    min_permeability: float = 0.0,
    min_developability: float = 0.5,
) -> List[Dict]:
    """
    Filter candidate peptides by developability criteria.

    Args:
        sequences: List of amino acid sequences.
        min_solubility: Minimum solubility score (0-1, 1=soluble).
        max_toxicity: Maximum toxicity score (0-1, 0=safe).
        max_hemolysis: Maximum hemolysis score (0-1, 0=safe).
        max_bitterness: Maximum bitterness score (0-1, 0=not bitter).
        min_stability: Minimum stability score (0-1, 1=stable).
        min_permeability: Minimum permeability score (0-1, 1=permeable).
        min_developability: Minimum overall developability score.

    Returns:
        List of score dicts for peptides passing all filters,
        sorted by developability (descending).
    """
    passed = []
    for seq in sequences:
        try:
            result = score_peptide(seq)
        except ValueError:
            continue

        if (
            result["solubility"] >= min_solubility
            and result["toxicity"] <= max_toxicity
            and result["hemolysis"] <= max_hemolysis
            and result["bitterness"] <= max_bitterness
            and result["stability"] >= min_stability
            and result["permeability"] >= min_permeability
            and result["developability"] >= min_developability
        ):
            passed.append(result)

    passed.sort(key=lambda x: x["developability"], reverse=True)
    return passed


# ============================================================
# CLI
# ============================================================

def _print_single_result(result: Dict) -> None:
    """Pretty-print a single peptide scoring result."""
    print(f"\n{'=' * 60}")
    print(f"  Peptide: {result['sequence']}")
    print(f"  Length:  {result['length']} aa  |  MW: {result['mw']} Da"
          f"  |  Charge: {result['net_charge']:+.0f}  |  GRAVY: {result['gravy']:.3f}")
    print(f"{'=' * 60}")
    print(f"  Toxicity:      {result['toxicity']:.4f}  {'[PASS]' if result['toxicity'] <= 0.3 else '[WARN]'}")
    print(f"  Hemolysis:     {result['hemolysis']:.4f}  {'[PASS]' if result['hemolysis'] <= 0.3 else '[WARN]'}")
    print(f"  Solubility:    {result['solubility']:.4f}  {'[PASS]' if result['solubility'] >= 0.5 else '[WARN]'}")
    print(f"  Permeability:  {result['permeability']:.4f}")
    print(f"  Stability:     {result['stability']:.4f}  {'[PASS]' if result['stability'] >= 0.4 else '[WARN]'}")
    print(f"  Bitterness:    {result['bitterness']:.4f}  {'[PASS]' if result['bitterness'] <= 0.5 else '[WARN]'}")
    print(f"  ----------------------------------------")
    print(f"  Developability: {result['developability']:.4f}")

    flags = result.get("flags", [])
    if flags:
        print(f"\n  Flags:")
        for flag in flags:
            print(f"    - {flag}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Meal Shield -- Peptide Developability Property Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python properties.py --sequence LIWKL
  python properties.py --sequence IPAVF --verbose
  python properties.py --csv data/food_peptides.csv
  python properties.py --csv data/food_peptides.csv --filter --min-solubility 0.6
        """,
    )
    parser.add_argument("--sequence", "-s", type=str, help="Single peptide sequence to score")
    parser.add_argument("--csv", type=str, help="Path to CSV with a 'sequence' column")
    parser.add_argument("--output", "-o", type=str, help="Output CSV path (for batch mode)")
    parser.add_argument("--filter", action="store_true", help="Filter candidates by thresholds")
    parser.add_argument("--min-solubility", type=float, default=0.5)
    parser.add_argument("--max-toxicity", type=float, default=0.3)
    parser.add_argument("--max-hemolysis", type=float, default=0.3)
    parser.add_argument("--max-bitterness", type=float, default=0.5)
    parser.add_argument("--min-stability", type=float, default=0.4)
    parser.add_argument("--min-permeability", type=float, default=0.0)
    parser.add_argument("--min-developability", type=float, default=0.5)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed flags")

    args = parser.parse_args()

    if not args.sequence and not args.csv:
        parser.print_help()
        sys.exit(1)

    # --- Single sequence mode ---
    if args.sequence:
        try:
            result = score_peptide(args.sequence)
            _print_single_result(result)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Batch CSV mode ---
    if args.csv:
        import pandas as pd

        try:
            df_input = pd.read_csv(args.csv)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(1)

        if "sequence" not in df_input.columns:
            print("Error: CSV must have a 'sequence' column", file=sys.stderr)
            sys.exit(1)

        sequences = df_input["sequence"].dropna().tolist()
        print(f"Scoring {len(sequences)} peptides...")

        if args.filter:
            results = filter_candidates(
                sequences,
                min_solubility=args.min_solubility,
                max_toxicity=args.max_toxicity,
                max_hemolysis=args.max_hemolysis,
                max_bitterness=args.max_bitterness,
                min_stability=args.min_stability,
                min_permeability=args.min_permeability,
                min_developability=args.min_developability,
            )
            df_out = pd.DataFrame(results)
            # Convert flags list to string
            if "flags" in df_out.columns:
                df_out["flags"] = df_out["flags"].apply(
                    lambda x: "; ".join(x) if isinstance(x, list) else x
                )
            print(f"  {len(df_out)} / {len(sequences)} passed filters")
        else:
            df_out = score_peptides_batch(sequences)

        # Summary statistics
        print(f"\n{'=' * 60}")
        print("  Summary Statistics")
        print(f"{'=' * 60}")
        score_cols = ["toxicity", "hemolysis", "solubility", "permeability",
                      "stability", "bitterness", "developability"]
        for col in score_cols:
            if col in df_out.columns:
                vals = pd.to_numeric(df_out[col], errors="coerce").dropna()
                if len(vals) > 0:
                    print(f"  {col:15s}  mean={vals.mean():.3f}  "
                          f"min={vals.min():.3f}  max={vals.max():.3f}")
        print()

        # Output
        output_path = args.output
        if output_path:
            df_out.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            # Print top results to stdout
            display_cols = ["sequence", "developability", "toxicity", "solubility",
                            "bitterness", "stability"]
            cols_available = [c for c in display_cols if c in df_out.columns]
            print(df_out[cols_available].to_string(index=False))


if __name__ == "__main__":
    main()
