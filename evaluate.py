"""
Meal Shield — Full Evaluation Pipeline

Trains both GNN-only and GNN+TDA models, saves checkpoints,
runs the comparison, and outputs results to RESULTS.md.

Usage:
    python evaluate.py
"""

import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from data import download_all_data, build_dataset, TARGETS, load_food_peptides
from model import MealShieldGNN, MealShieldGNN_TDA
from train import prepare_data, pad_features, collate_mixed, train_single_target, evaluate_target
from topology import compute_tda_for_dataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate_gnn(graphs, target_names, device, epochs=100,
                            node_feature_dim=11):
    """Train GNN-only model and return per-target metrics."""
    model = MealShieldGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        target_names=target_names,
    ).to(device)

    results = {}
    for target in target_names:
        train_g, val_g, test_g = prepare_data(graphs, target)
        if len(train_g) < 10:
            print(f"  Skipping {target}: only {len(train_g)} training samples")
            continue

        print(f"\n  Training GNN on {target} ({len(train_g)} train, {len(val_g)} val, {len(test_g)} test)")
        train_single_target(model, train_g, val_g, target, device, epochs=epochs)
        metrics = evaluate_target(model, test_g, target, device)
        results[target] = metrics
        print(f"  {target}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return model, results


def train_and_evaluate_tda(graphs, tda_cache, target_names, device, epochs=100,
                            node_feature_dim=11):
    """Train GNN+TDA model and return per-target metrics."""
    tda_dim = 42

    # Attach TDA features to graphs
    graphs_with_tda = []
    for g in graphs:
        key = getattr(g, "smiles", None) or getattr(g, "sequence", None) or getattr(g, "mol_key", None)
        if key and key in tda_cache:
            g_copy = g.clone()
            g_copy.tda = tda_cache[key].unsqueeze(0) if tda_cache[key].dim() == 1 else tda_cache[key]
            graphs_with_tda.append(g_copy)
        else:
            g_copy = g.clone()
            g_copy.tda = torch.zeros(1, tda_dim)
            graphs_with_tda.append(g_copy)

    model = MealShieldGNN_TDA(
        node_feature_dim=node_feature_dim,
        tda_feature_dim=tda_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        target_names=target_names,
    ).to(device)

    results = {}
    for target in target_names:
        train_g, val_g, test_g = prepare_data(graphs_with_tda, target)
        if len(train_g) < 10:
            print(f"  Skipping {target}: only {len(train_g)} training samples")
            continue

        print(f"\n  Training GNN+TDA on {target} ({len(train_g)} train, {len(val_g)} val, {len(test_g)} test)")
        train_single_target(model, train_g, val_g, target, device, epochs=epochs)
        metrics = evaluate_target(model, test_g, target, device)
        results[target] = metrics
        print(f"  {target}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    return model, results


def write_results_markdown(gnn_results, tda_results, output_path="RESULTS.md"):
    """Write comparison results to markdown file."""
    lines = [
        "# Meal Shield — Evaluation Results",
        "",
        "## GNN vs GNN+TDA Performance Comparison",
        "",
        "| Target | GNN R\u00b2 | GNN RMSE | GNN+TDA R\u00b2 | GNN+TDA RMSE | \u0394R\u00b2 |",
        "|--------|--------|----------|-------------|--------------|------|",
    ]

    deltas = []
    for target in sorted(set(list(gnn_results.keys()) + list(tda_results.keys()))):
        gnn = gnn_results.get(target, {})
        tda = tda_results.get(target, {})

        gnn_r2 = gnn.get("R2", float("nan"))
        gnn_rmse = gnn.get("RMSE", float("nan"))
        tda_r2 = tda.get("R2", float("nan"))
        tda_rmse = tda.get("RMSE", float("nan"))

        delta = tda_r2 - gnn_r2 if not (np.isnan(tda_r2) or np.isnan(gnn_r2)) else float("nan")
        if not np.isnan(delta):
            deltas.append(delta)

        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {target} | {gnn_r2:.4f} | {gnn_rmse:.4f} | {tda_r2:.4f} | {tda_rmse:.4f} | {sign}{delta:.4f} |"
        )

    lines.append("")

    if deltas:
        avg_delta = np.mean(deltas)
        sign = "+" if avg_delta > 0 else ""
        lines.append(f"**Average \u0394R\u00b2: {sign}{avg_delta:.4f}**")
        lines.append("")
        if avg_delta > 0:
            lines.append(
                f"TDA improved prediction by an average of {sign}{avg_delta:.4f} R\u00b2 across targets."
            )
        else:
            lines.append(
                f"TDA did not improve prediction in this run (\u0394R\u00b2 = {avg_delta:.4f})."
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **R\u00b2**: Coefficient of determination. 1.0 = perfect, 0.0 = mean baseline.")
    lines.append("- **RMSE**: Root mean squared error in pIC50 units.")
    lines.append("- **\u0394R\u00b2**: Improvement from adding TDA features (positive = TDA helped).")
    lines.append("")
    lines.append("## Training Details")
    lines.append("")
    lines.append("- Model: 3-layer GAT, 128 hidden dims, 4 attention heads")
    lines.append("- TDA: 42-dim features (30 persistence statistics + 12 cocycle features)")
    lines.append("- Data: ChEMBL bioactivity + 212 curated food peptides")
    lines.append("- Training: Adam optimizer, early stopping (patience=30), lr scheduling")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to {output_path}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # 1. Load data
    print("\n=== Loading data ===")
    chembl_df = download_all_data()
    peptides = load_food_peptides()

    print(f"ChEMBL records: {len(chembl_df)}")
    print(f"Food peptides: {len(peptides)}")

    # 2. Build graphs
    print("\n=== Building molecular graphs ===")
    graphs, stats = build_dataset(chembl_df, peptides)
    print(f"Total graphs: {len(graphs)}")

    # Auto-detect feature dimension
    feature_dims = set(g.x.shape[1] for g in graphs if g.x is not None)
    feature_dim = max(feature_dims) if feature_dims else 11

    target_names = list(TARGETS.keys())

    # 3. Train GNN-only
    print("\n=== Training GNN-only model ===")
    t0 = time.time()
    gnn_model, gnn_results = train_and_evaluate_gnn(
        graphs, target_names, device, node_feature_dim=feature_dim)
    gnn_time = time.time() - t0
    print(f"\nGNN training time: {gnn_time:.1f}s")

    torch.save(gnn_model.state_dict(), "checkpoints/meal_shield_gnn.pt")
    print("Saved: checkpoints/meal_shield_gnn.pt")

    # 4. Compute TDA features
    print("\n=== Computing TDA features ===")
    tda_cache_path = "data/tda_cache.pt"
    if os.path.exists(tda_cache_path):
        tda_cache = torch.load(tda_cache_path, weights_only=False)
        print(f"Loaded TDA cache: {len(tda_cache)} entries")
    else:
        tda_cache = compute_tda_for_dataset(chembl_df, peptide_list)
        torch.save(tda_cache, tda_cache_path)
        print(f"Computed and cached TDA features: {len(tda_cache)} entries")

    # 5. Train GNN+TDA
    print("\n=== Training GNN+TDA model ===")
    t0 = time.time()
    tda_model, tda_results = train_and_evaluate_tda(
        graphs, tda_cache, target_names, device, node_feature_dim=feature_dim)
    tda_time = time.time() - t0
    print(f"\nGNN+TDA training time: {tda_time:.1f}s")

    torch.save(tda_model.state_dict(), "checkpoints/meal_shield_gnn_tda.pt")
    print("Saved: checkpoints/meal_shield_gnn_tda.pt")

    # 6. Save comparison
    comparison = {
        "gnn": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in gnn_results.items()},
        "tda": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in tda_results.items()},
    }
    with open("checkpoints/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 7. Write results markdown
    write_results_markdown(gnn_results, tda_results)

    print("\n=== Done ===")
    print("Artifacts:")
    print("  checkpoints/meal_shield_gnn.pt       (GNN model)")
    print("  checkpoints/meal_shield_gnn_tda.pt    (GNN+TDA model)")
    print("  checkpoints/comparison_results.json   (metrics)")
    print("  RESULTS.md                            (formatted results)")


if __name__ == "__main__":
    main()
