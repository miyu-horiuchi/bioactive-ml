"""
Meal Shield — End-to-End Peptide Design Pipeline

Orchestrates candidate generation, property scoring, bioactivity prediction,
Pareto selection, and optional structure prediction into a single pipeline
for designing novel food peptides.

Usage:
    python design.py --target ace_inhibitor --method genetic --length 5 --n-generate 500 --top-k 20
    python design.py --target alpha_glucosidase --method enumerate --length 3 --predict-structures
    python design.py --target dpp4_inhibitor --method mc --length 4 --top-k 10
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from generate import generate_mc, generate_genetic, generate_enumerate
from properties import score_peptides_batch, filter_candidates
from pareto import rank_candidates, select_diverse, pareto_front
from train import predict_peptide, pad_features
from model import MealShieldGNN
from data import TARGETS

# ---------------------------------------------------------------------------
# Target metadata (friendly descriptions for the report)
# ---------------------------------------------------------------------------

TARGET_DESCRIPTIONS = {
    "alpha_glucosidase": "Alpha-glucosidase inhibitor -- blocks starch-to-glucose conversion, useful for blood sugar management.",
    "lipase": "Pancreatic lipase inhibitor -- reduces dietary fat absorption.",
    "bile_acid_receptor": "FXR bile acid receptor modulator -- regulates bile acid and cholesterol metabolism.",
    "sodium_hydrogen_exchanger": "NHE3 inhibitor -- modulates sodium absorption in the gut.",
    "ace_inhibitor": "ACE inhibitor -- lowers blood pressure by blocking angiotensin-converting enzyme.",
    "dpp4_inhibitor": "DPP-4 inhibitor -- prolongs incretin hormone activity for blood sugar control.",
}

GENERATION_METHODS = {
    "mc": "Monte Carlo random walk with acceptance criterion",
    "genetic": "Genetic algorithm with crossover, mutation, and selection",
    "enumerate": "Exhaustive enumeration of all possible sequences",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str = "checkpoints/meal_shield_gnn.pt",
    device: Optional[torch.device] = None,
) -> Tuple[MealShieldGNN, List[str], int, torch.device]:
    """
    Load a trained MealShieldGNN from checkpoint.

    Returns:
        (model, target_names, feature_dim, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.__version__ >= "2.4":
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found at {checkpoint_path}")
        print("Train a model first:  python train.py --multitask")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Support both wrapped (metadata dict) and bare state_dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        feature_dim = checkpoint.get("feature_dim", 11)
        hidden_dim = checkpoint.get("hidden_dim", 128)
        target_names = checkpoint.get("target_names", list(TARGETS.keys()))
    else:
        model_state = checkpoint
        # Infer feature_dim from input projection layer
        ip = model_state.get("input_proj.weight")
        feature_dim = int(ip.shape[1]) if ip is not None else 11
        hidden_dim = 128
        target_names = list(TARGETS.keys())

    model = MealShieldGNN(
        node_feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=3,
        target_names=target_names,
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  feature_dim={feature_dim}, hidden_dim={hidden_dim}, device={device}")
    print(f"  targets: {target_names}")

    return model, target_names, feature_dim, device


# ---------------------------------------------------------------------------
# Bioactivity prediction for a batch of sequences
# ---------------------------------------------------------------------------

def predict_batch(
    model: MealShieldGNN,
    sequences: List[str],
    target: str,
    target_names: List[str],
    feature_dim: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Predict bioactivity for a batch of peptide sequences.

    Returns a DataFrame with columns: sequence, pIC50, IC50_uM
    """
    results = []
    for seq in sequences:
        preds = predict_peptide(
            model, seq, target_names, device, feature_dim=feature_dim
        )
        if preds is not None and target in preds:
            results.append({
                "sequence": seq,
                "pIC50": preds[target]["pIC50"],
                "IC50_uM": preds[target]["IC50_uM"],
            })
        else:
            results.append({
                "sequence": seq,
                "pIC50": float("nan"),
                "IC50_uM": float("nan"),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    target: str,
    method: str,
    length: int,
    n_generated: int,
    n_after_filter: int,
    n_selected: int,
    results_df: pd.DataFrame,
    output_path: str,
    predict_structures: bool,
    min_solubility: float,
    max_toxicity: float,
    max_bitterness: float,
    elapsed_seconds: float,
) -> None:
    """Write a markdown report summarizing the design run."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_desc = TARGET_DESCRIPTIONS.get(target, target)
    method_desc = GENERATION_METHODS.get(method, method)

    lines = [
        f"# Peptide Design Report: {target}",
        "",
        f"**Generated:** {timestamp}",
        f"**Runtime:** {elapsed_seconds:.1f}s",
        "",
        "## Target",
        "",
        f"**{target}** -- {target_desc}",
        "",
        "## Generation Parameters",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Method | {method} ({method_desc}) |",
        f"| Peptide length | {length} |",
        f"| Candidates generated | {n_generated} |",
        f"| After property filter | {n_after_filter} |",
        f"| Final selected | {n_selected} |",
        f"| Min solubility | {min_solubility} |",
        f"| Max toxicity | {max_toxicity} |",
        f"| Max bitterness | {max_bitterness} |",
        f"| Structure prediction | {'Yes' if predict_structures else 'No'} |",
        "",
        "## Pipeline Summary",
        "",
        f"```",
        f"Generated:  {n_generated} candidates",
        f"Filtered:   {n_after_filter} passed safety/developability ({n_generated - n_after_filter} removed)",
        f"Selected:   {n_selected} after Pareto ranking + diversity selection",
        f"```",
        "",
        "## Top Candidates",
        "",
    ]

    # Build the results table
    display_cols = ["rank", "sequence", "pIC50", "IC50_uM"]
    property_cols = ["solubility", "toxicity", "bitterness"]
    pareto_cols = ["pareto_rank"]

    available_cols = [c for c in display_cols + property_cols + pareto_cols if c in results_df.columns]

    # Header
    header = "| " + " | ".join(available_cols) + " |"
    separator = "|" + "|".join(["---" for _ in available_cols]) + "|"
    lines.append(header)
    lines.append(separator)

    # Rows
    for _, row in results_df.head(min(len(results_df), 50)).iterrows():
        cells = []
        for col in available_cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if col == "IC50_uM":
                    cells.append(f"{val:.2f}")
                elif col == "pIC50":
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    if predict_structures and "structure_file" in results_df.columns:
        lines.append("## Predicted Structures")
        lines.append("")
        for _, row in results_df.iterrows():
            sf = row.get("structure_file", "")
            if sf:
                lines.append(f"- **{row['sequence']}**: `{sf}`")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by Meal Shield peptide design pipeline*")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def design_peptides(
    target: str,
    method: str = "genetic",
    length: int = 5,
    n_generate: int = 500,
    min_solubility: float = 0.5,
    max_toxicity: float = 0.3,
    max_bitterness: float = 0.7,
    top_k: int = 20,
    predict_structures: bool = False,
    output_dir: str = "designs",
    checkpoint_path: str = "checkpoints/meal_shield_gnn.pt",
) -> pd.DataFrame:
    """
    End-to-end peptide design pipeline.

    Pipeline stages:
        1. Load trained model
        2. Generate candidate sequences
        3. Score candidates on developability properties
        4. Filter out candidates that fail safety/developability thresholds
        5. Predict bioactivity for remaining candidates
        6. Run Pareto selection for multi-objective ranking
        7. Optionally predict 3D structures for top candidates
        8. Save results and generate report

    Args:
        target: Bioactivity target (e.g., "ace_inhibitor", "alpha_glucosidase").
        method: Generation method -- "mc", "genetic", or "enumerate".
        length: Peptide length (number of amino acids).
        n_generate: Number of candidate sequences to generate.
        min_solubility: Minimum solubility score to pass filter (0-1).
        max_toxicity: Maximum toxicity score to pass filter (0-1).
        max_bitterness: Maximum bitterness score to pass filter (0-1).
        top_k: Number of final candidates to select.
        predict_structures: Whether to predict 3D structures for top candidates.
        output_dir: Directory to save results.
        checkpoint_path: Path to trained model checkpoint.

    Returns:
        DataFrame of top designed peptides with scores and rankings.
    """
    t_start = time.time()

    # Validate target
    valid_targets = list(TARGETS.keys())
    if target not in valid_targets:
        print(f"Error: unknown target '{target}'")
        print(f"Valid targets: {valid_targets}")
        sys.exit(1)

    print("=" * 60)
    print("MEAL SHIELD -- PEPTIDE DESIGN PIPELINE")
    print("=" * 60)
    print(f"Target:    {target}")
    print(f"Method:    {method}")
    print(f"Length:    {length}")
    print(f"Generate:  {n_generate} candidates")
    print(f"Top-k:     {top_k}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print("[1/7] Loading trained model...")
    model, target_names, feature_dim, device = load_model(checkpoint_path)

    if target not in target_names:
        print(f"Warning: target '{target}' not in model targets {target_names}")
        print("Predictions may not be meaningful.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Generate candidate sequences
    # ------------------------------------------------------------------
    print(f"[2/7] Generating {n_generate} candidates (method={method}, length={length})...")

    generators = {
        "mc": generate_mc,
        "genetic": generate_genetic,
        "enumerate": generate_enumerate,
    }
    if method not in generators:
        print(f"Error: unknown method '{method}'. Choose from: {list(generators.keys())}")
        sys.exit(1)

    generate_fn = generators[method]
    raw_results = generate_fn(
        model=model,
        target=target,
        target_names=target_names,
        device=device,
        feature_dim=feature_dim,
        length=length,
        n_candidates=n_generate,
    )

    # raw_results is List[(sequence, pIC50)] — extract sequences
    candidates = list(dict.fromkeys([seq for seq, _ in raw_results]))
    n_generated = len(candidates)
    print(f"  Generated {n_generated} unique candidates")
    print()

    # ------------------------------------------------------------------
    # Step 3: Score developability properties
    # ------------------------------------------------------------------
    print(f"[3/7] Scoring developability properties...")
    properties_df = score_peptides_batch(candidates)
    print(f"  Scored {len(properties_df)} peptides on solubility, toxicity, hemolysis, stability, bitterness")
    print()

    # ------------------------------------------------------------------
    # Step 4: Filter candidates
    # ------------------------------------------------------------------
    print(f"[4/7] Filtering candidates (solubility>={min_solubility}, toxicity<={max_toxicity}, bitterness<={max_bitterness})...")
    filtered_seqs = filter_candidates(
        candidates,
        min_solubility=min_solubility,
        max_toxicity=max_toxicity,
        max_bitterness=max_bitterness,
    )
    filtered_df = properties_df[properties_df["sequence"].isin(filtered_seqs)]
    n_after_filter = len(filtered_df)
    print(f"  {n_after_filter} / {n_generated} candidates passed filters")

    if n_after_filter == 0:
        print("  Warning: no candidates passed filters. Relaxing thresholds...")
        # Fall back to top 50% by composite score
        properties_df["composite"] = (
            properties_df.get("solubility", 0.5)
            - properties_df.get("toxicity", 0.5)
            - properties_df.get("bitterness", 0.5)
        )
        properties_df = properties_df.sort_values("composite", ascending=False)
        filtered_df = properties_df.head(max(n_generated // 2, top_k))
        n_after_filter = len(filtered_df)
        print(f"  Relaxed selection: {n_after_filter} candidates")

    filtered_sequences = filtered_df["sequence"].tolist()
    print()

    # ------------------------------------------------------------------
    # Step 5: Predict bioactivity
    # ------------------------------------------------------------------
    print(f"[5/7] Predicting bioactivity for {n_after_filter} candidates...")
    activity_df = predict_batch(
        model, filtered_sequences, target, target_names, feature_dim, device
    )

    # Merge properties and activity
    merged_df = filtered_df.merge(activity_df, on="sequence", how="inner")

    # Drop rows where prediction failed
    merged_df = merged_df.dropna(subset=["pIC50"])
    print(f"  {len(merged_df)} candidates with valid predictions")
    if len(merged_df) == 0:
        print("Error: no valid predictions. Check model and sequences.")
        sys.exit(1)
    print()

    # ------------------------------------------------------------------
    # Step 6: Pareto selection
    # ------------------------------------------------------------------
    print(f"[6/7] Running Pareto ranking and diversity selection (top_k={top_k})...")

    # Define objectives for Pareto ranking:
    # Maximize pIC50 (bioactivity), maximize solubility, minimize toxicity
    objectives = {
        "pIC50": "maximize",
        "solubility": "maximize",
        "toxicity": "minimize",
    }
    # Add bitterness if available
    if "bitterness" in merged_df.columns:
        objectives["bitterness"] = "minimize"

    # Convert DataFrame to list of dicts for pareto module
    candidates_list = merged_df.to_dict("records")
    ranked_list = rank_candidates(candidates_list)
    front_list = pareto_front(candidates_list)
    print(f"  Pareto front: {len(front_list)} non-dominated candidates")

    selected_list = select_diverse(ranked_list, n=top_k)
    n_selected = len(selected_list)
    selected_df = pd.DataFrame(selected_list)
    print(f"  Selected {n_selected} diverse candidates")

    # Add rank column
    selected_df = selected_df.reset_index(drop=True)
    selected_df.insert(0, "rank", range(1, n_selected + 1))
    print()

    # ------------------------------------------------------------------
    # Step 7: Structure prediction (optional)
    # ------------------------------------------------------------------
    if predict_structures:
        print(f"[7/7] Predicting 3D structures for {n_selected} candidates...")
        try:
            from structure import predict_structures as pred_struct

            struct_dir = os.path.join(output_dir, "structures")
            os.makedirs(struct_dir, exist_ok=True)

            structure_files = []
            for _, row in selected_df.iterrows():
                seq = row["sequence"]
                pdb_path = os.path.join(struct_dir, f"{seq}.pdb")
                pred_struct(seq, output_path=pdb_path)
                structure_files.append(pdb_path)
                print(f"  {seq} -> {pdb_path}")

            selected_df["structure_file"] = structure_files
        except ImportError:
            print("  Warning: structure.py not available, skipping structure prediction.")
        except Exception as e:
            print(f"  Warning: structure prediction failed: {e}")
        print()
    else:
        print("[7/7] Structure prediction skipped (use --predict-structures to enable)")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{target}_designs.csv")
    selected_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    report_path = os.path.join(output_dir, f"{target}_report.md")
    elapsed = time.time() - t_start
    generate_report(
        target=target,
        method=method,
        length=length,
        n_generated=n_generated,
        n_after_filter=n_after_filter,
        n_selected=n_selected,
        results_df=selected_df,
        output_path=report_path,
        predict_structures=predict_structures,
        min_solubility=min_solubility,
        max_toxicity=max_toxicity,
        max_bitterness=max_bitterness,
        elapsed_seconds=elapsed,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("DESIGN COMPLETE")
    print("=" * 60)
    print(f"Target:      {target}")
    print(f"Generated:   {n_generated} -> Filtered: {n_after_filter} -> Selected: {n_selected}")
    print(f"Runtime:     {elapsed:.1f}s")
    print(f"Results:     {csv_path}")
    print(f"Report:      {report_path}")
    print()

    # Show top 5
    display_cols = ["rank", "sequence", "pIC50", "IC50_uM"]
    prop_cols = [c for c in ["solubility", "toxicity", "bitterness", "pareto_rank"] if c in selected_df.columns]
    show_cols = [c for c in display_cols + prop_cols if c in selected_df.columns]
    print("Top candidates:")
    print(selected_df[show_cols].head(5).to_string(index=False))
    print()

    return selected_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Meal Shield -- design novel food peptides with multi-objective optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python design.py --target ace_inhibitor --method genetic --length 5 --n-generate 500 --top-k 20
  python design.py --target alpha_glucosidase --method enumerate --length 3 --predict-structures
  python design.py --target dpp4_inhibitor --method mc --length 4 --top-k 10
        """,
    )
    parser.add_argument(
        "--target", required=True,
        choices=list(TARGETS.keys()),
        help="Bioactivity target to design peptides for",
    )
    parser.add_argument(
        "--method", default="genetic",
        choices=["mc", "genetic", "enumerate"],
        help="Candidate generation method (default: genetic)",
    )
    parser.add_argument(
        "--length", type=int, default=5,
        help="Peptide length in amino acids (default: 5)",
    )
    parser.add_argument(
        "--n-generate", type=int, default=500,
        help="Number of candidates to generate (default: 500)",
    )
    parser.add_argument(
        "--min-solubility", type=float, default=0.5,
        help="Minimum solubility score for filtering (default: 0.5)",
    )
    parser.add_argument(
        "--max-toxicity", type=float, default=0.3,
        help="Maximum toxicity score for filtering (default: 0.3)",
    )
    parser.add_argument(
        "--max-bitterness", type=float, default=0.7,
        help="Maximum bitterness score for filtering (default: 0.7)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of final candidates to select (default: 20)",
    )
    parser.add_argument(
        "--predict-structures", action="store_true",
        help="Predict 3D structures for top candidates",
    )
    parser.add_argument(
        "--output-dir", default="designs",
        help="Output directory for results (default: designs)",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/meal_shield_gnn.pt",
        help="Path to trained model checkpoint",
    )

    args = parser.parse_args()

    design_peptides(
        target=args.target,
        method=args.method,
        length=args.length,
        n_generate=args.n_generate,
        min_solubility=args.min_solubility,
        max_toxicity=args.max_toxicity,
        max_bitterness=args.max_bitterness,
        top_k=args.top_k,
        predict_structures=args.predict_structures,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
