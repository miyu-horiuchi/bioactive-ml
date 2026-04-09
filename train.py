"""
Meal Shield GNN — Training Script
Trains the multi-task GNN and evaluates on held-out data.
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

from model import MealShieldGNN, MealShieldGIN
from data import download_all_data, build_dataset, KNOWN_FOOD_PEPTIDES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(graphs, target_name, test_size=0.2, val_size=0.1):
    """Split graphs for a specific target into train/val/test."""
    target_graphs = [g for g in graphs if g.target_name == target_name]

    if len(target_graphs) < 10:
        print(f"  Warning: Only {len(target_graphs)} graphs for {target_name}")
        return target_graphs, [], []

    train_val, test = train_test_split(target_graphs, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=42)

    return train, val, test


def pad_features(graph, target_dim):
    """Pad node features to target dimension if needed."""
    if graph.x.shape[1] < target_dim:
        padding = torch.zeros(graph.x.shape[0], target_dim - graph.x.shape[1])
        graph.x = torch.cat([graph.x, padding], dim=1)
    elif graph.x.shape[1] > target_dim:
        graph.x = graph.x[:, :target_dim]
    # Remove non-tensor attributes to avoid collation issues with mixed data
    for attr in ['edge_attr', 'sequence', 'mol_type']:
        if hasattr(graph, attr):
            delattr(graph, attr)
    return graph


def collate_mixed(graphs, feature_dim=11):
    """Pad all graphs to same feature dimension for batching."""
    return [pad_features(g, feature_dim) for g in graphs]


def train_single_target(model, train_graphs, val_graphs, target_name,
                         device, epochs=100, lr=1e-3, batch_size=32):
    """Train the model on a single target's data."""
    feature_dim = 11

    train_graphs = collate_mixed(train_graphs, feature_dim)
    val_graphs = collate_mixed(val_graphs, feature_dim)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 30

    for epoch in range(epochs):
        # Train
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

        # Validate
        val_loss = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch)
                    if target_name in preds:
                        loss = F.mse_loss(preds[target_name], batch.y.squeeze())
                        val_loss += loss.item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_now:.2e}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def compute_metrics(all_trues, all_preds):
    """Compute R2 and RMSE from lists of true and predicted values."""
    if len(all_preds) < 2:
        return {"R2": None, "RMSE": None, "n": len(all_preds)}

    r2 = r2_score(all_trues, all_preds)
    rmse = mean_squared_error(all_trues, all_preds) ** 0.5

    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "n": len(all_preds),
    }


def evaluate_target(model, test_graphs, target_name, device, feature_dim=11):
    """Evaluate model on test set for a specific target."""
    test_graphs = collate_mixed(test_graphs, feature_dim)
    test_loader = DataLoader(test_graphs, batch_size=64)

    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            if target_name in preds:
                all_preds.extend(preds[target_name].cpu().tolist())
                all_trues.extend(batch.y.squeeze().cpu().tolist())

    return compute_metrics(all_trues, all_preds)


def predict_peptide(model, sequence, target_names, device, feature_dim=11):
    """Predict activity profile for a new peptide sequence."""
    from data import peptide_to_graph

    graph = peptide_to_graph(sequence, use_residue_level=True)
    if graph is None:
        return None

    graph = pad_features(graph, feature_dim)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    graph = graph.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(graph)

    results = {}
    for target in target_names:
        if target in predictions:
            pic50 = predictions[target].item()
            # pIC50 to IC50 in micromolar
            if pic50 > 0:
                ic50_nM = 10 ** (9 - pic50)
                ic50_uM = ic50_nM / 1000
            else:
                ic50_uM = float('inf')
            results[target] = {
                "pIC50": round(pic50, 3),
                "IC50_uM": round(ic50_uM, 2),
            }

    return results


def main():
    set_seed(42)

    # Device — MPS doesn't support scatter_reduce for GAT yet, use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    chembl_data = download_all_data(data_dir="data")
    graphs, stats = build_dataset(chembl_data, KNOWN_FOOD_PEPTIDES)

    # ---- Model ----
    print("\n" + "=" * 60)
    print("STEP 2: Building model")
    print("=" * 60)

    target_names = list(set(g.target_name for g in graphs))
    print(f"Targets: {target_names}")

    model = MealShieldGNN(
        node_feature_dim=11,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        target_names=target_names,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ---- Train per target ----
    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)

    all_metrics = {}
    for target in target_names:
        print(f"\n--- Training: {target} ---")
        train, val, test = prepare_data(graphs, target)
        print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

        if len(train) < 5:
            print(f"  Skipping {target} -- too few samples")
            continue

        model = train_single_target(
            model, train, val, target,
            device=device,
            epochs=150,
            lr=1e-3,
            batch_size=min(32, len(train)),
        )

        if test:
            metrics = evaluate_target(model, test, target, device)
            all_metrics[target] = metrics
            r2_str = str(metrics['R2']) if metrics['R2'] is not None else 'N/A'
            rmse_str = str(metrics['RMSE']) if metrics['RMSE'] is not None else 'N/A'
            print(f"  Test R2: {r2_str} | RMSE: {rmse_str} | N: {metrics['n']}")

    # ---- Save model ----
    print("\n" + "=" * 60)
    print("STEP 4: Saving model")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "target_names": target_names,
        "metrics": all_metrics,
    }, "checkpoints/meal_shield_gnn.pt")

    # ---- Predict on food peptides ----
    print("\n" + "=" * 60)
    print("STEP 5: Predictions on food peptides")
    print("=" * 60)

    test_peptides = [
        ("IPP",   "ACE inhibitor from Calpis"),
        ("VPP",   "ACE inhibitor from Calpis"),
        ("IPAVF", "Alpha-glucosidase inhibitor from bean"),
        ("KLPGF", "Alpha-glucosidase inhibitor from silk"),
        ("LPYPY", "DPP-4 inhibitor from Gouda cheese"),
        ("GGGG",  "Control -- simple glycine chain"),
        ("AAAA",  "Control -- simple alanine chain"),
        ("FWKL",  "Hypothetical -- hydrophobic + charged"),
        ("DEEK",  "Hypothetical -- all charged/acidic"),
        ("WWWW",  "Hypothetical -- all aromatic"),
    ]

    print(f"\n{'Peptide':<12} {'Best Target':<28} {'pIC50':>8} {'IC50 uM':>10}")
    print("-" * 62)

    for seq, desc in test_peptides:
        results = predict_peptide(model, seq, target_names, device)
        if results:
            best_target = min(results.items(), key=lambda x: x[1]['IC50_uM'])
            print(f"{seq:<12} {best_target[0]:<28} {best_target[1]['pIC50']:>8.2f} {best_target[1]['IC50_uM']:>10.1f}  ({desc})")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for target, metrics in all_metrics.items():
        r2 = metrics.get("R2")
        status = "GOOD" if r2 and r2 > 0.3 else "WEAK"
        r2_str = str(r2) if r2 is not None else "N/A"
        rmse_str = str(metrics.get("RMSE", "N/A"))
        print(f"  [{status}] {target}: R2={r2_str}, RMSE={rmse_str}, N={metrics.get('n', 0)}")

    print(f"\nModel saved to: checkpoints/meal_shield_gnn.pt")
    print(f"Data saved to:  data/meal_shield_graphs.pt")
    print("\nNext steps:")
    print("  1. Run visualize.py to see molecular graphs")
    print("  2. Add persistent homology features")
    print("  3. Generate candidates with Tang's STGFlow")


if __name__ == "__main__":
    main()
