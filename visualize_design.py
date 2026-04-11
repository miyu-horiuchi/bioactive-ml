"""
Meal Shield GNN -- Design Pipeline Visualization

Visualization functions for peptide design output: Pareto fronts,
property radar charts, generation landscapes, candidate comparisons,
and sequence logos.  Complements visualize.py (dataset overview,
peptide graphs, training metrics).

Each candidate is expected to be a dict with at least:
    sequence  : str   -- amino acid sequence
    pIC50     : float -- predicted pIC50 for the primary target
    properties: dict  -- developability scores, each in [0, 1]:
        toxicity, hemolysis, solubility, permeability, stability, bitterness

Optional fields used when present:
    IC50_uM          : float
    pareto_rank      : int   (1 = non-dominated front)
    overall_score    : float (aggregate developability score)

Usage:
    from visualize_design import create_design_report_figures
    paths = create_design_report_figures(candidates, target, "figures/design")
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------
# Style constants (matching visualize.py)
# -----------------------------------------------------------------------

_BORDER_COLOR = "#2c3e50"
_CMAP = plt.cm.RdYlBu_r
_DPI = 150
_FACECOLOR = "white"

# Developability property names in display order
PROPERTY_NAMES = [
    "toxicity",
    "hemolysis",
    "solubility",
    "permeability",
    "stability",
    "bitterness",
]

PROPERTY_LABELS = {
    "toxicity": "Toxicity\n(lower is better)",
    "hemolysis": "Hemolysis\n(lower is better)",
    "solubility": "Solubility\n(higher is better)",
    "permeability": "Permeability\n(higher is better)",
    "stability": "Stability\n(higher is better)",
    "bitterness": "Bitterness\n(lower is better)",
}

# Standard amino acid one-letter codes
AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _ensure_dir(path):
    """Create parent directory for *path* if it does not exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _save_and_close(fig, save_path):
    """Save figure and close it, returning the path."""
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=_DPI, bbox_inches="tight", facecolor=_FACECOLOR)
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return save_path


def _get_property_value(candidate, prop_name):
    """Safely retrieve a property value from a candidate dict."""
    props = candidate.get("properties", {})
    return props.get(prop_name, 0.0)


def _compute_pareto_ranks(candidates, obj_x, obj_y, maximize_x=True, maximize_y=True):
    """Assign Pareto ranks via iterative non-dominated sorting.

    Parameters
    ----------
    candidates : list of dict
    obj_x, obj_y : str
        Keys into each candidate dict (or into candidate["properties"]).
    maximize_x, maximize_y : bool
        Whether higher values are better for each objective.

    Returns
    -------
    ranks : list of int
        Pareto rank for each candidate (1 = non-dominated front).
    """

    def _val(c, key):
        if key in c:
            return c[key]
        return c.get("properties", {}).get(key, 0.0)

    n = len(candidates)
    xs = np.array([_val(c, obj_x) for c in candidates])
    ys = np.array([_val(c, obj_y) for c in candidates])

    # Negate objectives that should be maximized so that lower = better
    if maximize_x:
        xs = -xs
    if maximize_y:
        ys = -ys

    ranks = np.zeros(n, dtype=int)
    remaining = set(range(n))
    current_rank = 1

    while remaining:
        non_dominated = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i if j is <= on both and < on at least one
                if (xs[j] <= xs[i] and ys[j] <= ys[i]) and (
                    xs[j] < xs[i] or ys[j] < ys[i]
                ):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(i)
        for idx in non_dominated:
            ranks[idx] = current_rank
            remaining.discard(idx)
        current_rank += 1

    return ranks.tolist()


# -----------------------------------------------------------------------
# 1. Pareto Front
# -----------------------------------------------------------------------

def plot_pareto_front(
    candidates,
    obj_x="pIC50",
    obj_y="solubility",
    save_path=None,
    figsize=(9, 7),
    maximize_x=True,
    maximize_y=True,
):
    """2-D scatter of candidates colored by Pareto rank.

    Non-dominated points (rank 1) are highlighted and connected with a line.

    Parameters
    ----------
    candidates : list of dict
        Each dict must contain *obj_x* and *obj_y* as top-level keys or
        inside the ``properties`` sub-dict.
    obj_x, obj_y : str
        Objective names for x and y axes.
    save_path : str or None
        File path to write the figure.
    figsize : tuple
    maximize_x, maximize_y : bool
        Whether higher values are better for each objective.

    Returns
    -------
    matplotlib.figure.Figure
    """

    def _val(c, key):
        if key in c:
            return c[key]
        return c.get("properties", {}).get(key, 0.0)

    ranks = _compute_pareto_ranks(
        candidates, obj_x, obj_y, maximize_x, maximize_y
    )

    xs = np.array([_val(c, obj_x) for c in candidates])
    ys = np.array([_val(c, obj_y) for c in candidates])
    ranks_arr = np.array(ranks)

    max_rank = int(ranks_arr.max())
    norm = mcolors.Normalize(vmin=1, vmax=max(max_rank, 2))

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)

    # Background points (rank > 1)
    for rank in range(max_rank, 0, -1):
        mask = ranks_arr == rank
        color = _CMAP(norm(rank))
        ax.scatter(
            xs[mask],
            ys[mask],
            c=[color],
            s=50 if rank > 1 else 120,
            alpha=0.5 if rank > 1 else 0.95,
            edgecolors=_BORDER_COLOR if rank == 1 else "white",
            linewidths=1.2 if rank == 1 else 0.5,
            label=f"Rank {rank}" if rank <= 3 or rank == max_rank else None,
            zorder=2 if rank > 1 else 3,
        )

    # Connect Pareto front (rank 1) with a line
    front_mask = ranks_arr == 1
    front_xs = xs[front_mask]
    front_ys = ys[front_mask]
    sort_idx = np.argsort(front_xs)
    ax.plot(
        front_xs[sort_idx],
        front_ys[sort_idx],
        "-",
        color=_BORDER_COLOR,
        linewidth=1.5,
        alpha=0.7,
        zorder=2,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Pareto Rank", fontsize=10)

    # Annotate top-3 front points by combined objective
    if front_mask.sum() > 0:
        combined = front_xs + front_ys
        top_indices = np.where(front_mask)[0]
        best_order = np.argsort(-combined)[:3]
        for order_idx in best_order:
            ci = top_indices[order_idx]
            seq = candidates[ci].get("sequence", "")
            if seq:
                ax.annotate(
                    seq,
                    (xs[ci], ys[ci]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#ffeaa7",
                        alpha=0.8,
                    ),
                    arrowprops=dict(arrowstyle="->", color=_BORDER_COLOR),
                )

    ax.set_xlabel(obj_x.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(obj_y.replace("_", " ").title(), fontsize=11)
    ax.set_title("Pareto Front: Design Candidates", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# 2. Property Radar
# -----------------------------------------------------------------------

def plot_property_radar(candidate, save_path=None, figsize=(8, 8)):
    """Radar chart of all developability properties for a single candidate.

    Parameters
    ----------
    candidate : dict
        Must contain ``sequence`` and ``properties`` (dict with keys from
        PROPERTY_NAMES, values in [0, 1]).
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    props = candidate.get("properties", {})
    labels = []
    values = []
    for name in PROPERTY_NAMES:
        labels.append(PROPERTY_LABELS.get(name, name))
        values.append(props.get(name, 0.0))

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw=dict(polar=True), facecolor=_FACECOLOR
    )

    ax.plot(
        angles_closed,
        values_closed,
        "o-",
        linewidth=2,
        color="#2980B9",
        markersize=7,
    )
    ax.fill(angles_closed, values_closed, alpha=0.15, color="#2980B9")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")

    sequence = candidate.get("sequence", "?")
    pic50 = candidate.get("pIC50")
    title = f"Developability: {sequence}"
    if pic50 is not None:
        title += f"  (pIC50 = {pic50:.2f})"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=25)

    # Annotate best and worst properties
    best_idx = int(np.argmax(values))
    worst_idx = int(np.argmin(values))
    ax.annotate(
        f"Best: {PROPERTY_NAMES[best_idx]} ({values[best_idx]:.2f})",
        xy=(angles[best_idx], values[best_idx]),
        xytext=(30, 15),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#27AE60"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", alpha=0.8),
    )
    if best_idx != worst_idx:
        ax.annotate(
            f"Worst: {PROPERTY_NAMES[worst_idx]} ({values[worst_idx]:.2f})",
            xy=(angles[worst_idx], values[worst_idx]),
            xytext=(-30, -20),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", alpha=0.8),
        )

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# 3. Generation Landscape
# -----------------------------------------------------------------------

def plot_generation_landscape(candidates, target, save_path=None, figsize=(10, 7)):
    """Scatter of generated candidates: x=length, y=pIC50, color=score, size=rank.

    Parameters
    ----------
    candidates : list of dict
    target : str
        Target name (used in the title).
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    lengths = np.array([len(c.get("sequence", "")) for c in candidates])
    pic50s = np.array([c.get("pIC50", 0.0) for c in candidates])

    # Overall developability score
    scores = np.array([c.get("overall_score", 0.0) for c in candidates])
    if scores.max() == 0:
        # Fallback: compute mean of available properties
        computed = []
        for c in candidates:
            vals = [
                c.get("properties", {}).get(p, 0.0) for p in PROPERTY_NAMES
            ]
            computed.append(np.mean(vals) if any(v > 0 for v in vals) else 0.5)
        scores = np.array(computed)

    # Pareto rank for sizing (smaller rank = larger marker)
    ranks = np.array([c.get("pareto_rank", 1) for c in candidates], dtype=float)
    if ranks.max() == ranks.min():
        sizes = np.full(len(candidates), 60.0)
    else:
        # Invert: rank 1 gets the biggest marker
        sizes = 30 + 150 * (1 - (ranks - ranks.min()) / (ranks.max() - ranks.min()))

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)

    norm = mcolors.Normalize(vmin=max(scores.min(), 0), vmax=min(scores.max(), 1))
    scatter = ax.scatter(
        lengths,
        pic50s,
        c=scores,
        s=sizes,
        cmap=_CMAP,
        norm=norm,
        alpha=0.8,
        edgecolors=_BORDER_COLOR,
        linewidths=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Developability Score", fontsize=10)

    # Add jitter to lengths for readability when many share a length
    if len(set(lengths)) < 4:
        ax.set_xlabel("Sequence Length (residues)", fontsize=11)
    else:
        ax.set_xlabel("Sequence Length (residues)", fontsize=11)

    ax.set_ylabel("Predicted pIC50", fontsize=11)
    ax.set_title(
        f"Design Landscape: {target.replace('_', ' ').title()}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.15)

    # Annotate the best candidate
    best_idx = int(np.argmax(pic50s))
    best_seq = candidates[best_idx].get("sequence", "")
    if best_seq:
        ax.annotate(
            f"{best_seq}\npIC50={pic50s[best_idx]:.2f}",
            (lengths[best_idx], pic50s[best_idx]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=_BORDER_COLOR),
        )

    # Size legend
    for rank_val, label in [(1, "Rank 1"), (max(int(ranks.max()), 2), f"Rank {int(ranks.max())}")]:
        if ranks.max() == ranks.min():
            break
        s = 30 + 150 * (1 - (rank_val - ranks.min()) / (ranks.max() - ranks.min()))
        ax.scatter([], [], s=s, c="grey", alpha=0.6, edgecolors=_BORDER_COLOR,
                   linewidths=0.5, label=label)
    if ranks.max() != ranks.min():
        ax.legend(loc="lower right", fontsize=9, title="Pareto Rank", framealpha=0.8)

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# 4. Top Candidates Comparison
# -----------------------------------------------------------------------

def plot_top_candidates_comparison(
    candidates, top_k=5, save_path=None, figsize=(12, 6)
):
    """Grouped bar chart comparing top-K candidates across all scored properties.

    Each group is a candidate, each bar within the group is a property score.

    Parameters
    ----------
    candidates : list of dict
        Will be sorted by pIC50 descending; the first *top_k* are plotted.
    top_k : int
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Sort by pIC50 descending, take top_k
    sorted_cands = sorted(candidates, key=lambda c: c.get("pIC50", 0.0), reverse=True)
    display = sorted_cands[:top_k]

    if not display:
        fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)
        ax.text(0.5, 0.5, "No candidates to display", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        _save_and_close(fig, save_path)
        return fig

    n_props = len(PROPERTY_NAMES)
    n_cands = len(display)
    bar_width = 0.8 / n_props
    x = np.arange(n_cands)

    # Property colors
    prop_colors = [
        "#E74C3C",  # toxicity -- red
        "#C0392B",  # hemolysis -- dark red
        "#3498DB",  # solubility -- blue
        "#2ECC71",  # permeability -- green
        "#F39C12",  # stability -- orange
        "#9B59B6",  # bitterness -- purple
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)

    for i, prop_name in enumerate(PROPERTY_NAMES):
        vals = [_get_property_value(c, prop_name) for c in display]
        offset = (i - n_props / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=prop_name.replace("_", " ").title(),
            color=prop_colors[i % len(prop_colors)],
            edgecolor="white",
            linewidth=0.5,
        )

    # X-axis labels: sequence + pIC50
    xlabels = []
    for c in display:
        seq = c.get("sequence", "?")
        pic50 = c.get("pIC50")
        lbl = seq
        if pic50 is not None:
            lbl += f"\n(pIC50={pic50:.2f})"
        xlabels.append(lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Property Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Top-{n_cands} Candidates: Property Comparison",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        framealpha=0.85,
    )
    ax.grid(True, axis="y", alpha=0.15)

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# 5. Sequence Logo (Amino Acid Frequency Heatmap)
# -----------------------------------------------------------------------

def plot_sequence_logo(sequences, save_path=None, figsize=(12, 7)):
    """Amino acid frequency heatmap across positions of aligned sequences.

    Rows = sequence positions, columns = amino acids, color = frequency.
    Shows what residues the design process converges on at each position.

    Parameters
    ----------
    sequences : list of str
        Peptide sequences (should be same length for clean alignment;
        shorter sequences are right-padded with gaps for display).
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not sequences:
        fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)
        ax.text(0.5, 0.5, "No sequences to display", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        _save_and_close(fig, save_path)
        return fig

    max_len = max(len(s) for s in sequences)
    n_aa = len(AMINO_ACIDS)
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    # Build frequency matrix (positions x amino acids)
    freq_matrix = np.zeros((max_len, n_aa))
    for seq in sequences:
        for pos, aa in enumerate(seq):
            if aa in aa_to_idx:
                freq_matrix[pos, aa_to_idx[aa]] += 1

    # Normalize each position to [0, 1]
    row_sums = freq_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    freq_matrix = freq_matrix / row_sums

    fig, ax = plt.subplots(figsize=figsize, facecolor=_FACECOLOR)

    im = ax.imshow(
        freq_matrix,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(n_aa))
    ax.set_xticklabels(AMINO_ACIDS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(max_len))
    ax.set_yticklabels([f"Pos {i + 1}" for i in range(max_len)], fontsize=9)

    # Annotate cells with frequency > 0
    for i in range(freq_matrix.shape[0]):
        for j in range(freq_matrix.shape[1]):
            val = freq_matrix[i, j]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Frequency", fontsize=10)

    ax.set_xlabel("Amino Acid", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title(
        f"Sequence Logo: Residue Frequencies ({len(sequences)} sequences)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    _save_and_close(fig, save_path)
    return fig


# -----------------------------------------------------------------------
# 6. Report Generator
# -----------------------------------------------------------------------

def create_design_report_figures(candidates, target, output_dir="figures/design"):
    """Generate all design-pipeline figures for a run.

    Parameters
    ----------
    candidates : list of dict
        Design output.  Each dict should contain ``sequence``, ``pIC50``,
        and ``properties``.
    target : str
        Name of the bioactivity target (e.g. ``"ace_inhibitor"``).
    output_dir : str
        Directory to write figures into.

    Returns
    -------
    list of str
        Paths of all generated figure files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # 1. Pareto front (pIC50 vs solubility)
    path = os.path.join(output_dir, "pareto_front.png")
    plot_pareto_front(candidates, obj_x="pIC50", obj_y="solubility", save_path=path)
    paths.append(path)

    # 2. Property radar for the top candidate (by pIC50)
    sorted_cands = sorted(
        candidates, key=lambda c: c.get("pIC50", 0.0), reverse=True
    )
    if sorted_cands:
        top = sorted_cands[0]
        seq = top.get("sequence", "top")
        path = os.path.join(output_dir, f"radar_{seq}.png")
        plot_property_radar(top, save_path=path)
        paths.append(path)

    # 3. Generation landscape
    path = os.path.join(output_dir, "generation_landscape.png")
    plot_generation_landscape(candidates, target, save_path=path)
    paths.append(path)

    # 4. Top-5 candidates comparison
    path = os.path.join(output_dir, "top_candidates_comparison.png")
    plot_top_candidates_comparison(candidates, top_k=5, save_path=path)
    paths.append(path)

    # 5. Sequence logo of top-20 candidates
    top_seqs = [c.get("sequence", "") for c in sorted_cands[:20] if c.get("sequence")]
    if top_seqs:
        path = os.path.join(output_dir, "sequence_logo.png")
        plot_sequence_logo(top_seqs, save_path=path)
        paths.append(path)

    print(f"\nDesign report: {len(paths)} figures saved to {output_dir}/")
    return paths


# -----------------------------------------------------------------------
# CLI demo
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meal Shield -- Design Pipeline Visualization (demo mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/design",
        help="Directory to save figures (default: figures/design)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=60,
        help="Number of synthetic demo candidates to generate (default: 60)",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to a design.py output CSV. If provided, real candidates are "
             "loaded and demo mode is skipped.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ace_inhibitor",
        help="Target name to label figures with (default: ace_inhibitor).",
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.input_csv:
        print("MEAL SHIELD GNN -- Design Visualization")
    else:
        print("MEAL SHIELD GNN -- Design Visualization (demo)")
    print("=" * 60)

    if args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv)
        real_candidates = []
        for _, row in df.iterrows():
            props = {
                "toxicity": float(row.get("toxicity", 0.0)),
                "hemolysis": float(row.get("hemolysis", 0.0)),
                "solubility": float(row.get("solubility", 0.0)),
                "permeability": float(row.get("permeability", 0.0)),
                "stability": float(row.get("stability", 0.0)),
                "bitterness": float(row.get("bitterness", 0.0)),
            }
            real_candidates.append({
                "sequence": str(row["sequence"]),
                "pIC50": float(row.get("pIC50", 0.0)),
                "IC50_uM": float(row.get("IC50_uM", 0.0)),
                "properties": props,
                "overall_score": float(row.get("developability", row.get("composite", 0.0))),
                "pareto_rank": int(row.get("rank", 1)),
            })
        print(f"Loaded {len(real_candidates)} candidates from {args.input_csv}")
        paths = create_design_report_figures(
            real_candidates, args.target, args.output_dir
        )
        print(f"\nAll {len(paths)} figures saved to {args.output_dir}/")
        sys.exit(0)

    # Generate synthetic candidate data for demonstration
    np.random.seed(42)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    demo_target = "ace_inhibitor"
    demo_candidates = []

    for i in range(args.n_candidates):
        length = np.random.choice([3, 4, 5, 6, 7])
        seq = "".join(np.random.choice(amino_acids, size=length))
        pic50 = np.random.normal(5.0, 1.2)
        pic50 = max(2.0, min(8.0, pic50))

        props = {
            "toxicity": np.clip(np.random.beta(2, 5), 0, 1),
            "hemolysis": np.clip(np.random.beta(2, 5), 0, 1),
            "solubility": np.clip(np.random.beta(5, 2), 0, 1),
            "permeability": np.clip(np.random.beta(3, 3), 0, 1),
            "stability": np.clip(np.random.beta(4, 2), 0, 1),
            "bitterness": np.clip(np.random.beta(2, 4), 0, 1),
        }

        overall = (
            (1 - props["toxicity"])
            + (1 - props["hemolysis"])
            + props["solubility"]
            + props["permeability"]
            + props["stability"]
            + (1 - props["bitterness"])
        ) / 6.0

        demo_candidates.append(
            {
                "sequence": seq,
                "pIC50": round(pic50, 3),
                "IC50_uM": round(10 ** (6 - pic50), 2) if pic50 > 0 else 1e6,
                "properties": props,
                "overall_score": round(overall, 3),
                "pareto_rank": np.random.randint(1, 6),
            }
        )

    paths = create_design_report_figures(
        demo_candidates, demo_target, args.output_dir
    )
    print(f"\nAll {len(paths)} demo figures saved to {args.output_dir}/")
