"""
Meal Shield — GNN + TDA Training Script
Compares GNN-only vs GNN+PersistentHomology to measure the TDA boost.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from model import MealShieldGNN, MealShieldGNN_TDA
from data import (
    download_all_data,
    load_food_peptides,
    smiles_to_graph,
    peptide_to_graph,
    TARGETS,
)
from topology import compute_tda_for_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_features(graph, target_dim):
    """Pad node features to target dimension. Strips non-tensor attrs
    for batching but keeps _smiles/_sequence for TDA lookup."""
    if graph.x.shape[1] < target_dim:
        padding = torch.zeros(graph.x.shape[0], target_dim - graph.x.shape[1])
        graph.x = torch.cat([graph.x, padding], dim=1)
    elif graph.x.shape[1] > target_dim:
        graph.x = graph.x[:, :target_dim]
    for attr in ['edge_attr', 'sequence', 'mol_type', 'smiles']:
        if hasattr(graph, attr):
            delattr(graph, attr)
    return graph


def attach_tda_features(graphs, tda_cache, tda_dim=42):
    """Attach precomputed TDA features to each graph."""
    success = 0
    for graph in graphs:
        key = getattr(graph, '_smiles', None) or getattr(graph, '_sequence', None)
        if key and key in tda_cache:
            graph.tda = tda_cache[key].unsqueeze(0)
            success += 1
        else:
            graph.tda = torch.zeros(1, tda_dim)
    return graphs, success


def build_graphs_with_keys(chembl_df, peptide_list, max_per_target=2000,
                           esm_cache=None):
    """Build graphs and store SMILES/sequence keys for TDA cache lookup.

    Mirrors data.build_dataset but preserves the raw key (SMILES for small
    molecules, sequence for peptides) so that precomputed TDA features can
    be attached later.
    """
    graphs = []

    for target_name in chembl_df["target"].unique():
        target_df = chembl_df[chembl_df["target"] == target_name].head(max_per_target)
        count = 0
        for _, row in target_df.iterrows():
            graph = smiles_to_graph(row["smiles"])
            if graph is not None:
                graph.y = torch.tensor([row["pIC50"]], dtype=torch.float)
                graph.target_name = target_name
                graph.mol_type = "small_molecule"
                graph._smiles = row["smiles"]
                graphs.append(graph)
                count += 1
        print(f"  {target_name}: {count} graphs")

    if peptide_list:
        pep_count = 0
        for pep in peptide_list:
            graph = peptide_to_graph(pep["sequence"], use_residue_level=True,
                                     esm_cache=esm_cache)
            if graph is not None:
                graph.y = torch.tensor([pep["pIC50"]], dtype=torch.float)
                graph.target_name = pep["target"]
                graph.mol_type = "peptide"
                graph._sequence = pep["sequence"]
                graphs.append(graph)
                pep_count += 1
        print(f"  food_peptides: {pep_count} graphs")

    return graphs


def prepare_splits(graphs, target_name):
    """Split into train/val/test for a given target."""
    target_graphs = [g for g in graphs if g.target_name == target_name]
    if len(target_graphs) < 10:
        return target_graphs, [], []
    train_val, test = train_test_split(target_graphs, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.125, random_state=42)
    return train, val, test


def train_model(model, train_graphs, val_graphs, target_name, device,
                epochs=150, lr=1e-3, batch_size=32, feature_dim=None):
    if feature_dim is None:
        dims = [g.x.shape[1] for g in train_graphs + val_graphs if g.x is not None]
        feature_dim = max(dims) if dims else 11
    """Train a model on one target."""
    train_g = [pad_features(g, feature_dim) for g in train_graphs]
    val_g = [pad_features(g, feature_dim) for g in val_graphs]

    train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_g, batch_size=batch_size) if val_g else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch)
            if target_name in preds:
                loss = F.mse_loss(preds[target_name], batch.y.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch)
                    if target_name in preds:
                        val_loss += F.mse_loss(preds[target_name], batch.y.squeeze()).item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 30:
                break

        if epoch % 30 == 0:
            print(f"    Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_evaluation(model, test_graphs, target_name, device, feature_dim=None):
    if feature_dim is None:
        dims = [g.x.shape[1] for g in test_graphs if g.x is not None]
        feature_dim = max(dims) if dims else 11
    """Run evaluation on test set."""
    test_g = [pad_features(g, feature_dim) for g in test_graphs]
    loader = DataLoader(test_g, batch_size=64)

    model.eval()
    preds_all, trues_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            if target_name in preds:
                preds_all.extend(preds[target_name].cpu().tolist())
                trues_all.extend(batch.y.squeeze().cpu().tolist())

    if len(preds_all) < 2:
        return {"R2": None, "RMSE": None, "n": 0}

    return {
        "R2": round(r2_score(trues_all, preds_all), 4),
        "RMSE": round(mean_squared_error(trues_all, preds_all) ** 0.5, 4),
        "n": len(preds_all),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Meal Shield GNN+TDA")
    parser.add_argument("--esm", action="store_true",
                        help="Use ESM-2 embeddings on peptides (larger feature dim)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.__version__ >= "2.4":
        # MPS scatter_reduce for GATConv is only available from 2.4+
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device} (PyTorch {torch.__version__})")

    # ---- Load data ----
    print("\n" + "=" * 60)
    print("STEP 1: Loading data + computing TDA features")
    print("=" * 60)

    chembl_df = download_all_data(data_dir="data")
    print(f"ChEMBL records: {len(chembl_df)}")

    peptides = load_food_peptides()
    allowed_targets = set(TARGETS.keys())
    before = len(peptides)
    peptides = [p for p in peptides if p["target"] in allowed_targets]
    print(f"Filtered peptides: {before} -> {len(peptides)} "
          f"(kept targets: {sorted(allowed_targets)})")

    esm_cache = None
    if args.esm:
        from esm_embeddings import compute_and_cache_embeddings
        seqs = [p["sequence"] for p in peptides]
        print(f"Loading ESM-2 embeddings for {len(set(seqs))} unique sequences...")
        esm_cache = compute_and_cache_embeddings(seqs)

    print("\nBuilding molecular graphs (with TDA key preservation)...")
    graphs = build_graphs_with_keys(chembl_df, peptides, esm_cache=esm_cache)
    print(f"Total graphs: {len(graphs)}")

    # Auto-detect feature dimension
    feature_dims = set(g.x.shape[1] for g in graphs if g.x is not None)
    feature_dim = max(feature_dims) if feature_dims else 11
    print(f"Feature dimensions in dataset: {feature_dims} -> using {feature_dim}")

    # Compute TDA features
    tda_cache_file = "data/tda_cache.pt"
    if os.path.exists(tda_cache_file):
        print("\nLoading cached TDA features...")
        tda_cache = torch.load(tda_cache_file, weights_only=False)
        print(f"  Loaded {len(tda_cache)} TDA features from cache")
    else:
        print("\nComputing persistent homology features (this takes a few minutes)...")
        tda_cache = compute_tda_for_dataset(chembl_df, peptides, method="statistics")
        torch.save(tda_cache, tda_cache_file)
        print(f"  Saved {len(tda_cache)} TDA features to cache")

    # Keep only targets that both (a) have enough graphs and (b) are in the
    # allowed target set (matches train.py behavior).
    target_counts = {}
    for g in graphs:
        t = g.target_name
        target_counts[t] = target_counts.get(t, 0) + 1
    targets = sorted(
        t for t, c in target_counts.items()
        if c >= 20 and t in allowed_targets
    )
    print(f"\nTargets: {targets}")

    # ---- Run comparison ----
    print("\n" + "=" * 60)
    print("STEP 2: Training GNN-only vs GNN+TDA")
    print("=" * 60)

    comparison = {"gnn_only": {}, "gnn_tda": {}}

    for target in targets:
        train, val, test = prepare_splits(graphs, target)
        if len(train) < 5 or len(test) < 2:
            print(f"\n  Skipping {target} (too few samples)")
            continue

        print(f"\n{'='*50}")
        print(f"  Target: {target}")
        print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        print(f"{'='*50}")

        # --- GNN-only ---
        print(f"\n  [A] GNN-only:")
        set_seed(42)
        gnn_model = MealShieldGNN(
            node_feature_dim=feature_dim, hidden_dim=args.hidden_dim, num_heads=4,
            num_layers=3, dropout=0.2, target_names=targets,
        ).to(device)

        gnn_model = train_model(
            gnn_model, train, val, target, device,
            epochs=args.epochs, lr=args.lr,
            batch_size=min(args.batch_size, len(train)),
            feature_dim=feature_dim,
        )
        gnn_metrics = run_evaluation(gnn_model, test, target, device,
                                     feature_dim=feature_dim)
        comparison["gnn_only"][target] = gnn_metrics
        print(f"    R2 = {gnn_metrics['R2']}  |  RMSE = {gnn_metrics['RMSE']}")

        # --- GNN + TDA ---
        print(f"\n  [B] GNN + TDA:")
        set_seed(42)

        train_tda, _ = attach_tda_features(train, tda_cache)
        val_tda, _ = attach_tda_features(val, tda_cache)
        test_tda, _ = attach_tda_features(test, tda_cache)

        tda_model = MealShieldGNN_TDA(
            node_feature_dim=feature_dim, tda_feature_dim=42,
            hidden_dim=args.hidden_dim, num_heads=4, num_layers=3,
            dropout=0.2, target_names=targets,
        ).to(device)

        tda_model = train_model(
            tda_model, train_tda, val_tda, target, device,
            epochs=args.epochs, lr=args.lr,
            batch_size=min(args.batch_size, len(train_tda)),
            feature_dim=feature_dim,
        )
        tda_metrics = run_evaluation(tda_model, test_tda, target, device,
                                     feature_dim=feature_dim)
        comparison["gnn_tda"][target] = tda_metrics
        print(f"    R2 = {tda_metrics['R2']}  |  RMSE = {tda_metrics['RMSE']}")

        # --- Delta ---
        if gnn_metrics['R2'] is not None and tda_metrics['R2'] is not None:
            delta = tda_metrics['R2'] - gnn_metrics['R2']
            label = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
            print(f"\n    Delta R2: {delta:+.4f} ({label})")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: GNN-only vs GNN+TDA")
    print("=" * 60)
    print(f"\n{'Target':<30} {'GNN R2':>8} {'GNN+TDA R2':>12} {'Delta':>8} {'Result':>10}")
    print("-" * 72)

    total_delta = 0
    n_targets = 0
    for target in targets:
        gnn_r2 = comparison["gnn_only"].get(target, {}).get("R2")
        tda_r2 = comparison["gnn_tda"].get(target, {}).get("R2")

        if gnn_r2 is not None and tda_r2 is not None:
            delta = tda_r2 - gnn_r2
            total_delta += delta
            n_targets += 1
            mark = "+" if delta > 0.01 else "-" if delta < -0.01 else "="
            print(f"{target:<30} {gnn_r2:>8.4f} {tda_r2:>12.4f} {delta:>+8.4f} {mark:>10}")

    if n_targets > 0:
        avg_delta = total_delta / n_targets
        print(f"\n{'Average':>30} {'':>8} {'':>12} {avg_delta:>+8.4f}")

        if avg_delta > 0:
            print(f"\nPersistent homology IMPROVED predictions by {abs(avg_delta)*100:.1f}% R2 on average.")
        else:
            print(f"\nPersistent homology did not improve predictions ({abs(avg_delta)*100:.1f}% R2 decrease).")
            print("This may improve with more data or persistence image features instead of statistics.")

    # Save
    os.makedirs("checkpoints", exist_ok=True)

    results_payload = {
        "config": {
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "esm": args.esm,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "tda_method": "statistics",
            "tda_feature_dim": 42,
        },
        "targets": targets,
        "comparison": comparison,
    }
    with open("checkpoints/tda_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    # Keep the legacy filename too so older tooling doesn't break.
    with open("checkpoints/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    torch.save({
        "model_state_dict": tda_model.state_dict(),
        "target_names": targets,
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "esm_enabled": args.esm,
        "comparison": comparison,
    }, "checkpoints/meal_shield_gnn_tda.pt")

    print(f"\nResults saved to checkpoints/tda_results.json")
    print(f"GNN+TDA model saved to checkpoints/meal_shield_gnn_tda.pt")


if __name__ == "__main__":
    main()
