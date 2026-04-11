"""
Meal Shield — Multi-Objective Scoring & Pareto-Optimal Selection

Selects the best peptide candidates by considering multiple objectives
simultaneously: bioactivity, solubility, toxicity, stability, and
bitterness. Implements Pareto front extraction, weighted ranking, and
diversity-aware selection so the final shortlist covers a broad region
of the objective space without redundant near-duplicates.

Reference: "AI-Designed Peptides as Tools for Biochemistry" — emphasises
multi-objective optimisation when designing peptide candidates.

Usage:
    python pareto.py --input data/generated_peptides.csv --target ace_inhibitor --top-k 20
    python pareto.py --input data/generated_peptides.csv --target dpp4_inhibitor --top-k 10 --output results/
"""

import argparse
import os
import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# Default objective weights
# ============================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bioactivity": 0.4,
    "solubility": 0.2,
    "low_toxicity": 0.2,
    "stability": 0.1,
    "not_bitter": 0.1,
}

# All objectives are "higher is better" after scoring.  Raw values
# that are "lower is better" (e.g. toxicity) must be inverted before
# they enter the pipeline — see score_candidate().

OBJECTIVES: List[str] = list(DEFAULT_WEIGHTS.keys())


# ============================================================
# Helper functions
# ============================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    n, m = len(s1), len(s2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use a single-row DP approach for O(min(n,m)) memory.
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n

    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,       # insertion
                prev[i] + 1,           # deletion
                prev[i - 1] + cost,    # substitution
            )
        prev = curr

    return prev[n]


def is_dominated(a: Dict[str, float], b: Dict[str, float],
                 objectives: List[str]) -> bool:
    """Return True if *b* dominates *a* on all *objectives*.

    Domination means b is >= a on every objective and strictly > on at
    least one.  All objectives are treated as "higher is better".
    """
    dominated_on_all = True
    strictly_better_on_any = False
    for obj in objectives:
        a_val = a.get(obj, 0.0)
        b_val = b.get(obj, 0.0)
        if b_val < a_val:
            dominated_on_all = False
            break
        if b_val > a_val:
            strictly_better_on_any = True
    return dominated_on_all and strictly_better_on_any


def normalize_scores(candidates: List[Dict], objectives: List[str]) -> List[Dict]:
    """Min-max normalise each objective to [0, 1] across *candidates*.

    Returns a new list of candidate dicts with normalised objective
    values.  Non-objective fields are copied unchanged.
    """
    if not candidates:
        return []

    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for obj in objectives:
        vals = [c.get(obj, 0.0) for c in candidates]
        mins[obj] = min(vals)
        maxs[obj] = max(vals)

    normalised = []
    for c in candidates:
        nc = dict(c)
        for obj in objectives:
            rng = maxs[obj] - mins[obj]
            if rng == 0:
                nc[obj] = 1.0  # all identical -> treat as perfect
            else:
                nc[obj] = (c.get(obj, 0.0) - mins[obj]) / rng
        normalised.append(nc)

    return normalised


# ============================================================
# Core public API
# ============================================================

def pareto_front(candidates: List[Dict],
                 objectives: Optional[List[str]] = None) -> List[Dict]:
    """Return the Pareto-optimal subset of *candidates*.

    A candidate is Pareto-optimal if no other candidate dominates it on
    every objective.  The returned list preserves the input order.

    Parameters
    ----------
    candidates : list of dict
        Each dict must contain float values for every key in *objectives*.
    objectives : list of str, optional
        Objective keys to consider.  Defaults to ``OBJECTIVES``.

    Returns
    -------
    list of dict
        The non-dominated (Pareto-optimal) candidates.
    """
    if objectives is None:
        objectives = OBJECTIVES

    if not candidates:
        return []

    front = []
    for i, ci in enumerate(candidates):
        dominated = False
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            if is_dominated(ci, cj, objectives):
                dominated = True
                break
        if not dominated:
            front.append(ci)

    return front


def score_candidate(
    sequence: str,
    target: str,
    model_predictions: Dict[str, float],
    properties: Dict[str, float],
) -> Dict:
    """Build a unified candidate profile from predictions and properties.

    Combines raw model outputs with developability properties so every
    downstream function receives a consistent dict.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    target : str
        Primary bioactivity target name (e.g. ``"ace_inhibitor"``).
    model_predictions : dict
        Keys are target names, values are predicted pIC50 or probability.
    properties : dict
        Developability scores.  Expected keys (all optional — missing
        values default to 0):
        - ``solubility``  : 0-1 predicted aqueous solubility
        - ``toxicity``    : 0-1 predicted toxicity (will be inverted)
        - ``stability``   : 0-1 predicted proteolytic / thermal stability
        - ``bitterness``  : 0-1 predicted bitterness (will be inverted)

    Returns
    -------
    dict
        Candidate profile with keys: ``sequence``, ``target``,
        ``bioactivity``, ``solubility``, ``low_toxicity``, ``stability``,
        ``not_bitter``, plus a copy of raw ``model_predictions``.
    """
    bioactivity = model_predictions.get(target, 0.0)
    solubility = properties.get("solubility", 0.0)
    toxicity = properties.get("toxicity", 0.0)
    stability = properties.get("stability", 0.0)
    bitterness = properties.get("bitterness", 0.0)

    return {
        "sequence": sequence,
        "target": target,
        "bioactivity": float(bioactivity),
        "solubility": float(solubility),
        "low_toxicity": 1.0 - float(toxicity),  # invert: higher = less toxic
        "stability": float(stability),
        "not_bitter": 1.0 - float(bitterness),   # invert: higher = less bitter
        "model_predictions": dict(model_predictions),
    }


def rank_candidates(
    candidates: List[Dict],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Rank candidates by weighted multi-objective score.

    Scores are min-max normalised before weighting so that each
    objective contributes proportionally regardless of its raw scale.

    Parameters
    ----------
    candidates : list of dict
        Each dict must contain float values for every objective key.
    weights : dict, optional
        ``{objective: weight}``.  Defaults to ``DEFAULT_WEIGHTS``.

    Returns
    -------
    list of dict
        Candidates sorted best-first, each augmented with a
        ``"weighted_score"`` field.
    """
    if not candidates:
        return []

    if weights is None:
        weights = DEFAULT_WEIGHTS

    objectives = list(weights.keys())
    normed = normalize_scores(candidates, objectives)

    scored = []
    for nc in normed:
        ws = sum(weights.get(obj, 0.0) * nc.get(obj, 0.0) for obj in objectives)
        nc["weighted_score"] = ws
        scored.append(nc)

    scored.sort(key=lambda c: c["weighted_score"], reverse=True)
    return scored


def select_diverse(
    candidates: List[Dict],
    n: int,
    diversity_threshold: float = 0.5,
) -> List[Dict]:
    """Select up to *n* top candidates that are mutually diverse.

    Greedily picks the highest-ranked candidate, then skips any
    subsequent candidate whose sequence is too similar (Levenshtein
    distance < threshold * max_length) to an already-selected one.

    Parameters
    ----------
    candidates : list of dict
        Pre-sorted best-first (e.g. output of ``rank_candidates``).
        Each dict must have a ``"sequence"`` key.
    n : int
        Maximum number of candidates to return.
    diversity_threshold : float
        Minimum relative edit distance between any pair of selected
        sequences.  0.5 means at least 50 % of the longer sequence
        length must differ.

    Returns
    -------
    list of dict
        Up to *n* diverse candidates in rank order.
    """
    if not candidates:
        return []

    selected: List[Dict] = []
    for c in candidates:
        if len(selected) >= n:
            break
        seq = c.get("sequence", "")
        too_similar = False
        for s in selected:
            s_seq = s.get("sequence", "")
            max_len = max(len(seq), len(s_seq))
            if max_len == 0:
                continue
            dist = levenshtein_distance(seq, s_seq)
            if dist < diversity_threshold * max_len:
                too_similar = True
                break
        if not too_similar:
            selected.append(c)

    return selected


def _assign_pareto_ranks(candidates: List[Dict],
                         objectives: Optional[List[str]] = None) -> List[Dict]:
    """Assign successive Pareto ranks (1 = front, 2 = next layer, ...).

    Each candidate dict is augmented with a ``"pareto_rank"`` field.
    Returns the full list (order preserved).
    """
    if objectives is None:
        objectives = OBJECTIVES

    remaining = list(range(len(candidates)))
    ranks = [0] * len(candidates)
    rank = 1

    while remaining:
        subset = [candidates[i] for i in remaining]
        front = pareto_front(subset, objectives)

        # Build lookup keys from the front (sequence + objective values)
        # since pareto_front may return the same or different dict objects.
        front_keys = set()
        for f in front:
            key = f.get("sequence", "") + "|" + "|".join(
                str(f.get(o, "")) for o in objectives
            )
            front_keys.add(key)

        next_remaining = []
        for i in remaining:
            c = candidates[i]
            key = c.get("sequence", "") + "|" + "|".join(
                str(c.get(o, "")) for o in objectives
            )
            if key in front_keys:
                ranks[i] = rank
            else:
                next_remaining.append(i)
        remaining = next_remaining
        rank += 1

    out = []
    for i, c in enumerate(candidates):
        augmented = dict(c)
        augmented["pareto_rank"] = ranks[i]
        out.append(augmented)
    return out


# ============================================================
# Report generation
# ============================================================

def _sequence_properties(seq: str) -> Dict:
    """Compute basic physicochemical properties from sequence."""
    # Amino acid molecular weights (monoisotopic, Da)
    mw_table = {
        "A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01,
        "E": 129.04, "Q": 128.06, "G": 57.02, "H": 137.06, "I": 113.08,
        "L": 113.08, "K": 128.09, "M": 131.04, "F": 147.07, "P": 97.05,
        "S": 87.03, "T": 101.05, "W": 186.08, "Y": 163.06, "V": 99.07,
    }
    hydrophobicity = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }

    seq_upper = seq.upper()
    length = len(seq_upper)
    mw = sum(mw_table.get(aa, 0.0) for aa in seq_upper) + 18.02  # + water
    avg_hydro = (
        np.mean([hydrophobicity.get(aa, 0.0) for aa in seq_upper])
        if length > 0
        else 0.0
    )
    charge = sum(1 for aa in seq_upper if aa in "RK") - sum(
        1 for aa in seq_upper if aa in "DE"
    )

    return {
        "length": length,
        "molecular_weight": round(mw, 2),
        "avg_hydrophobicity": round(float(avg_hydro), 3),
        "net_charge": charge,
    }


def generate_report(
    candidates: List[Dict],
    target: str,
    output_path: Optional[str] = None,
) -> str:
    """Generate a Markdown report of ranked peptide candidates.

    Parameters
    ----------
    candidates : list of dict
        Candidates to report.  Should already be ranked/scored.
    target : str
        Primary bioactivity target.
    output_path : str, optional
        Directory or file path.  If a directory, the report is written as
        ``<output_path>/pareto_report_<target>.md``.  If *None*, no file
        is written.

    Returns
    -------
    str
        The Markdown report text.
    """
    objectives = OBJECTIVES
    ranked = rank_candidates(candidates)
    ranked = _assign_pareto_ranks(ranked, objectives)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Pareto Selection Report — {target}",
        "",
        f"Generated: {timestamp}  ",
        f"Candidates evaluated: {len(candidates)}  ",
        f"Pareto-optimal (rank 1): "
        f"{sum(1 for c in ranked if c.get('pareto_rank') == 1)}",
        "",
        "## Objectives & weights",
        "",
        "| Objective | Weight |",
        "|-----------|--------|",
    ]
    for obj, w in DEFAULT_WEIGHTS.items():
        lines.append(f"| {obj} | {w} |")
    lines.append("")

    # Top candidates table
    lines.append("## Top candidates")
    lines.append("")
    header_cols = (
        ["Rank", "Sequence", "Pareto", "Score"]
        + [o.replace("_", " ").title() for o in objectives]
        + ["Length", "MW (Da)", "Avg Hydro", "Charge"]
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for i, c in enumerate(ranked, 1):
        seq = c.get("sequence", "?")
        props = _sequence_properties(seq)
        row = [
            str(i),
            f"`{seq}`",
            str(c.get("pareto_rank", "-")),
            f"{c.get('weighted_score', 0):.3f}",
        ]
        for obj in objectives:
            row.append(f"{c.get(obj, 0):.3f}")
        row += [
            str(props["length"]),
            f"{props['molecular_weight']:.1f}",
            f"{props['avg_hydrophobicity']:.2f}",
            str(props["net_charge"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by `pareto.py`*")

    report = "\n".join(lines) + "\n"

    # Write to disk if requested
    if output_path is not None:
        if os.path.isdir(output_path):
            filepath = os.path.join(output_path, f"pareto_report_{target}.md")
        else:
            filepath = output_path
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(report)
        print(f"Report written to {filepath}")

    return report


# ============================================================
# CSV loading helper
# ============================================================

def _load_candidates_from_csv(
    path: str,
    target: str,
) -> List[Dict]:
    """Read a generated-peptides CSV and build candidate dicts.

    Expected columns: ``sequence``, plus any scored property columns.
    Recognised property columns (case-insensitive):
        bioactivity / pIC50 / ic50_uM, solubility, toxicity, stability,
        bitterness, low_toxicity, not_bitter.

    If the CSV contains ``ic50_uM`` instead of ``pIC50``, it is
    converted: pIC50 = -log10(IC50_uM * 1e-6).
    """
    df = pd.read_csv(path)
    col_lower = {c.lower(): c for c in df.columns}

    # Resolve bioactivity column
    if "bioactivity" in col_lower:
        bio_col = col_lower["bioactivity"]
    elif "pic50" in col_lower:
        bio_col = col_lower["pic50"]
    elif "pIC50" in df.columns:
        bio_col = "pIC50"
    elif "ic50_um" in col_lower:
        raw_col = col_lower["ic50_um"]
        df["pIC50"] = -np.log10(df[raw_col].clip(lower=1e-12) * 1e-6)
        bio_col = "pIC50"
    else:
        bio_col = None

    candidates: List[Dict] = []
    for _, row in df.iterrows():
        seq = str(row.get(col_lower.get("sequence", "sequence"), ""))
        if not seq or seq == "nan":
            continue

        model_preds = {}
        if bio_col is not None:
            model_preds[target] = float(row[bio_col])

        props: Dict[str, float] = {}

        # Direct objective columns
        for prop_key in ("solubility", "stability"):
            if prop_key in col_lower:
                props[prop_key] = float(row[col_lower[prop_key]])

        # Columns that may already be inverted
        if "toxicity" in col_lower:
            props["toxicity"] = float(row[col_lower["toxicity"]])
        elif "low_toxicity" in col_lower:
            # Already inverted — undo so score_candidate re-inverts
            props["toxicity"] = 1.0 - float(row[col_lower["low_toxicity"]])

        if "bitterness" in col_lower:
            props["bitterness"] = float(row[col_lower["bitterness"]])
        elif "not_bitter" in col_lower:
            props["bitterness"] = 1.0 - float(row[col_lower["not_bitter"]])

        candidate = score_candidate(seq, target, model_preds, props)
        candidates.append(candidate)

    return candidates


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-objective Pareto selection for peptide candidates",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to CSV with candidate peptides and scored properties",
    )
    parser.add_argument(
        "--target", required=True,
        help="Primary bioactivity target (e.g. ace_inhibitor, dpp4_inhibitor)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top candidates to report (default: 20)",
    )
    parser.add_argument(
        "--diversity", type=float, default=0.5,
        help="Diversity threshold for select_diverse (default: 0.5)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory or file path for the Markdown report",
    )

    args = parser.parse_args()

    # Load
    print(f"Loading candidates from {args.input} ...")
    candidates = _load_candidates_from_csv(args.input, args.target)
    print(f"  {len(candidates)} candidates loaded")

    if not candidates:
        print("No candidates found. Check the input CSV.")
        return

    # Pareto front
    front = pareto_front(candidates)
    print(f"  Pareto-optimal candidates: {len(front)}")

    # Rank
    ranked = rank_candidates(candidates)
    print(f"  Top score: {ranked[0]['weighted_score']:.4f}  "
          f"(sequence: {ranked[0]['sequence']})")

    # Diversity filter
    diverse = select_diverse(ranked, args.top_k, args.diversity)
    print(f"  Diverse top-{args.top_k}: {len(diverse)} selected")

    # Report
    report = generate_report(diverse, args.target, args.output)
    if args.output is None:
        print("\n" + report)


if __name__ == "__main__":
    main()
