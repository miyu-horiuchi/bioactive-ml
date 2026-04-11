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
from data import download_all_data, build_dataset, KNOWN_FOOD_PEPTIDES, load_food_peptides, TARGETS

NUM_WORKERS = int(os.environ.get("DATALOADER_WORKERS", min(4, os.cpu_count() or 1)))


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
    for attr in ['edge_attr', 'sequence', 'mol_type', 'smiles']:
        if hasattr(graph, attr):
            delattr(graph, attr)
    # Ensure target_idx is a tensor for batching
    if hasattr(graph, 'target_idx') and not isinstance(graph.target_idx, torch.Tensor):
        graph.target_idx = torch.tensor([graph.target_idx], dtype=torch.long)
    return graph


def collate_mixed(graphs, feature_dim=11):
    """Pad all graphs to same feature dimension for batching."""
    return [pad_features(g, feature_dim) for g in graphs]


def train_single_target(model, train_graphs, val_graphs, target_name,
                         device, epochs=100, lr=1e-3, batch_size=32,
                         feature_dim=None):
    """Train the model on a single target's data."""
    if feature_dim is None:
        # Detect max feature dimension from the graphs
        dims = [g.x.shape[1] for g in train_graphs + val_graphs if g.x is not None]
        feature_dim = max(dims) if dims else 11

    train_graphs = collate_mixed(train_graphs, feature_dim)
    val_graphs = collate_mixed(val_graphs, feature_dim)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0) if val_graphs else None

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
    test_loader = DataLoader(test_graphs, batch_size=64,
                             num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)

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


def train_multitask(model, graphs, target_names, device,
                    epochs=150, lr=1e-3, batch_size=32, feature_dim=None):
    """
    Multi-task training with masked loss.

    Instead of training each target independently, this trains all targets
    simultaneously. Each graph contributes loss only for its labeled target,
    but the shared GNN backbone learns from all data.
    """
    if feature_dim is None:
        dims = [g.x.shape[1] for g in graphs if g.x is not None]
        feature_dim = max(dims) if dims else 11

    # Assign target index to each graph
    target_to_idx = {t: i for i, t in enumerate(target_names)}
    for g in graphs:
        g.target_idx = target_to_idx.get(g.target_name, -1)

    valid_graphs = [g for g in graphs if g.target_idx >= 0]

    # Split: stratified by target
    from collections import defaultdict
    by_target = defaultdict(list)
    for g in valid_graphs:
        by_target[g.target_name].append(g)

    train_graphs, val_graphs, test_graphs_raw = [], [], []
    for target, tg in by_target.items():
        if len(tg) < 5:
            train_graphs.extend(tg)
            continue
        tr_val, te = train_test_split(tg, test_size=0.2, random_state=42)
        tr, va = train_test_split(tr_val, test_size=0.1 / 0.8, random_state=42)
        train_graphs.extend(tr)
        val_graphs.extend(va)
        test_graphs_raw.extend(te)

    train_graphs = collate_mixed(train_graphs, feature_dim)
    val_graphs = collate_mixed(val_graphs, feature_dim)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0) if val_graphs else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 30

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(batch)

            # Masked loss: each graph contributes only to its labeled target
            total_loss = torch.tensor(0.0, device=device)
            n_terms = 0
            for tname in target_names:
                if tname not in preds:
                    continue
                mask = (batch.target_idx == target_to_idx[tname])
                if mask.sum() == 0:
                    continue
                pred_masked = preds[tname][mask]
                true_masked = batch.y.squeeze()[mask]
                total_loss = total_loss + F.mse_loss(pred_masked, true_masked)
                n_terms += 1

            if n_terms > 0:
                total_loss = total_loss / n_terms
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += total_loss.item()
                n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        val_loss = 0
        if val_loader:
            model.eval()
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch)
                    batch_loss = torch.tensor(0.0, device=device)
                    terms = 0
                    for tname in target_names:
                        if tname not in preds:
                            continue
                        mask = (batch.target_idx == target_to_idx[tname])
                        if mask.sum() == 0:
                            continue
                        batch_loss = batch_loss + F.mse_loss(
                            preds[tname][mask], batch.y.squeeze()[mask]
                        )
                        terms += 1
                    if terms > 0:
                        val_loss += (batch_loss / terms).item()
                        n_val += 1
            val_loss /= max(n_val, 1)
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
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.2e}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Per-target evaluation on test set
    all_metrics = {}
    for target in target_names:
        target_test = [g for g in test_graphs_raw if g.target_name == target]
        if len(target_test) < 2:
            continue
        metrics = evaluate_target(model, target_test, target, device,
                                  feature_dim=feature_dim)
        all_metrics[target] = metrics

    return model, all_metrics


def cross_validate(graphs, target_names, device, n_folds=5,
                    node_feature_dim=11, hidden_dim=128, epochs=100):
    """
    Stratified k-fold cross-validation with per-target metrics.

    Returns dict: target -> {"R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "folds"}
    """
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict

    # Assign fold-stratification labels based on target
    target_labels = [g.target_name for g in graphs]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Collect per-target metrics across folds
    fold_metrics = defaultdict(lambda: {"R2": [], "RMSE": []})

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(graphs, target_labels)):
        print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        # Further split train into train+val
        train_targets = [g.target_name for g in train_graphs]
        try:
            tr, va = train_test_split(train_graphs, test_size=0.1,
                                      stratify=train_targets, random_state=42)
        except ValueError:
            # If a class has too few samples for stratification
            tr, va = train_test_split(train_graphs, test_size=0.1, random_state=42)

        model = MealShieldGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=3,
            dropout=0.2,
            target_names=target_names,
        ).to(device)

        # Train on all targets simultaneously
        feature_dim = node_feature_dim
        target_to_idx = {t: i for i, t in enumerate(target_names)}
        for g in tr + va + test_graphs:
            g.target_idx = target_to_idx.get(g.target_name, -1)

        tr_padded = collate_mixed(tr, feature_dim)
        va_padded = collate_mixed(va, feature_dim)

        train_loader = DataLoader(tr_padded, batch_size=32, shuffle=True,
                                  num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
        val_loader = DataLoader(va_padded, batch_size=32,
                                num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0) if va_padded else None

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
        )

        best_val_loss = float('inf')
        best_state = None
        patience = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = model(batch)
                loss = torch.tensor(0.0, device=device)
                n = 0
                for tname in target_names:
                    if tname not in preds:
                        continue
                    mask = (batch.target_idx == target_to_idx[tname])
                    if mask.sum() == 0:
                        continue
                    loss = loss + F.mse_loss(preds[tname][mask], batch.y.squeeze()[mask])
                    n += 1
                if n > 0:
                    (loss / n).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            if val_loader:
                model.eval()
                vl = 0
                nv = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        preds = model(batch)
                        bl = torch.tensor(0.0, device=device)
                        bt = 0
                        for tname in target_names:
                            if tname not in preds:
                                continue
                            mask = (batch.target_idx == target_to_idx[tname])
                            if mask.sum() == 0:
                                continue
                            bl = bl + F.mse_loss(preds[tname][mask], batch.y.squeeze()[mask])
                            bt += 1
                        if bt > 0:
                            vl += (bl / bt).item()
                            nv += 1
                vl /= max(nv, 1)
                scheduler.step(vl)
                if vl < best_val_loss:
                    best_val_loss = vl
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 30:
                    break

        if best_state:
            model.load_state_dict(best_state)

        # Evaluate per target
        for target in target_names:
            target_test = [g for g in test_graphs if g.target_name == target]
            if len(target_test) < 2:
                continue
            metrics = evaluate_target(model, target_test, target, device,
                                      feature_dim=feature_dim)
            if metrics["R2"] is not None:
                fold_metrics[target]["R2"].append(metrics["R2"])
                fold_metrics[target]["RMSE"].append(metrics["RMSE"])

    # Aggregate
    results = {}
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({n_folds}-fold)")
    print(f"{'='*60}")
    print(f"{'Target':<28} {'R2':>12} {'RMSE':>12}")
    print("-" * 54)

    for target in target_names:
        fm = fold_metrics.get(target)
        if not fm or not fm["R2"]:
            continue
        r2s = np.array(fm["R2"])
        rmses = np.array(fm["RMSE"])
        results[target] = {
            "R2_mean": round(float(r2s.mean()), 4),
            "R2_std": round(float(r2s.std()), 4),
            "RMSE_mean": round(float(rmses.mean()), 4),
            "RMSE_std": round(float(rmses.std()), 4),
            "folds": len(r2s),
        }
        print(f"{target:<28} {r2s.mean():.4f}+/-{r2s.std():.4f}  {rmses.mean():.4f}+/-{rmses.std():.4f}")

    return results


def predict_peptide(model, sequence, target_names, device, feature_dim=11,
                    use_esm=None, n_samples=1):
    """Predict activity profile for a new peptide sequence.

    If ``use_esm`` is None, it's inferred from feature_dim (>11 implies the
    model was trained with ESM-2 embeddings). When enabled, the ESM-2
    per-residue embedding is computed on the fly so inference matches training.

    If ``n_samples > 1``, MC Dropout is used for uncertainty quantification:
    dropout layers are kept active and ``n_samples`` forward passes are run.
    The returned dict then includes ``pIC50_std`` and a 95% confidence
    interval in addition to the point prediction.
    """
    from data import peptide_to_graph

    if use_esm is None:
        use_esm = feature_dim > 11

    esm_cache = None
    if use_esm:
        from esm_embeddings import get_esm_embedding
        emb = get_esm_embedding(sequence)
        if emb is not None:
            esm_cache = {sequence: emb}

    graph = peptide_to_graph(sequence, use_residue_level=True, esm_cache=esm_cache)
    if graph is None:
        return None

    graph = pad_features(graph, feature_dim)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    graph = graph.to(device)

    if n_samples > 1:
        # MC Dropout: put the whole model in inference mode (train(False)),
        # then re-enable only the dropout layers so we get stochastic forward
        # passes for uncertainty estimation.
        model.train(False)
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train(True)

        samples = {t: [] for t in target_names}
        with torch.no_grad():
            for _ in range(n_samples):
                preds = model(graph)
                for t in target_names:
                    if t in preds:
                        samples[t].append(preds[t].item())

        results = {}
        for target in target_names:
            vals = samples[target]
            if not vals:
                continue
            arr = np.array(vals)
            mean = float(arr.mean())
            std = float(arr.std())
            lo = mean - 1.96 * std
            hi = mean + 1.96 * std
            if mean > 0:
                ic50_uM = (10 ** (9 - mean)) / 1000
            else:
                ic50_uM = float("inf")
            results[target] = {
                "pIC50": round(mean, 3),
                "pIC50_std": round(std, 3),
                "pIC50_ci95": [round(lo, 3), round(hi, 3)],
                "IC50_uM": round(ic50_uM, 2),
                "n_samples": n_samples,
            }
        # Restore all modules to inference mode before returning.
        model.train(False)
        return results

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
    import argparse

    parser = argparse.ArgumentParser(description="Train Meal Shield GNN")
    parser.add_argument("--esm", action="store_true",
                        help="Use ESM-2 embeddings (328-dim node features)")
    parser.add_argument("--multitask", action="store_true",
                        help="Use masked multi-task training (all targets simultaneously)")
    parser.add_argument("--cv", type=int, default=0,
                        help="Run k-fold cross-validation (0=off, 5=recommended)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.__version__ >= "2.4":
        # MPS scatter_reduce works reliably from PyTorch 2.4+
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} (PyTorch {torch.__version__})")

    # ---- Data ----
    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    chembl_data = download_all_data(data_dir="data")
    peptides = load_food_peptides()

    # Keep only peptides whose target is one of the 6 ChEMBL-backed targets.
    # Drops antioxidant/bile_acid_binding/inactive/mineral_binding heads
    # which have too few samples to train meaningful regressors.
    allowed_targets = set(TARGETS.keys())
    before = len(peptides)
    peptides = [p for p in peptides if p["target"] in allowed_targets]
    print(f"Filtered peptides: {before} -> {len(peptides)} "
          f"(kept targets: {sorted(allowed_targets)})")

    # ESM-2 embeddings (optional, adds 320-dim per-residue features)
    esm_cache = None
    if args.esm:
        from esm_embeddings import compute_and_cache_embeddings
        peptide_seqs = [p["sequence"] for p in peptides]
        print(f"Loading ESM-2 embeddings for {len(set(peptide_seqs))} unique sequences...")
        esm_cache = compute_and_cache_embeddings(peptide_seqs)

    graphs, stats = build_dataset(chembl_data, peptides, esm_cache=esm_cache)

    # Auto-detect feature dimension from data
    feature_dims = set(g.x.shape[1] for g in graphs if g.x is not None)
    feature_dim = max(feature_dims) if feature_dims else 11
    print(f"Feature dimensions in dataset: {feature_dims} -> using {feature_dim}")

    # ---- Model ----
    print("\n" + "=" * 60)
    print("STEP 2: Building model")
    print("=" * 60)

    target_names = sorted(set(g.target_name for g in graphs))
    print(f"Targets ({len(target_names)}): {target_names}")

    model = MealShieldGNN(
        node_feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=4,
        num_layers=3,
        dropout=0.2,
        target_names=target_names,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # torch.compile for ~30% speedup (PyTorch 2.x). Set TORCH_COMPILE=0 to
    # disable (useful on CPU where it can be slower or trigger crashes).
    if hasattr(torch, "compile") and os.environ.get("TORCH_COMPILE", "1") != "0":
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() skipped: {e}")

    # ---- Cross-validation mode ----
    if args.cv > 0:
        print(f"\n" + "=" * 60)
        print(f"RUNNING {args.cv}-FOLD CROSS-VALIDATION")
        print("=" * 60)
        cv_results = cross_validate(
            graphs, target_names, device,
            n_folds=args.cv,
            node_feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
        )
        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"\nSaved CV results to checkpoints/cv_results.json")
        return

    # ---- Training ----
    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)

    if args.multitask:
        print("Mode: Multi-task (masked loss, all targets simultaneously)")
        model, all_metrics = train_multitask(
            model, graphs, target_names, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            feature_dim=feature_dim,
        )
    else:
        print("Mode: Per-target sequential training")
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
                epochs=args.epochs,
                lr=args.lr,
                batch_size=min(args.batch_size, len(train)),
                feature_dim=feature_dim,
            )

            if test:
                metrics = evaluate_target(model, test, target, device,
                                          feature_dim=feature_dim)
                all_metrics[target] = metrics
                r2_str = str(metrics['R2']) if metrics['R2'] is not None else 'N/A'
                rmse_str = str(metrics['RMSE']) if metrics['RMSE'] is not None else 'N/A'
                print(f"  Test R2: {r2_str} | RMSE: {rmse_str} | N: {metrics['n']}")

    # ---- Save model ----
    print("\n" + "=" * 60)
    print("STEP 4: Saving model")
    print("=" * 60)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "target_names": target_names,
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "esm_enabled": args.esm,
        "multitask": args.multitask,
        "metrics": all_metrics,
    }
    torch.save(checkpoint, "checkpoints/meal_shield_gnn.pt")

    results_payload = {
        "config": {
            "feature_dim": feature_dim,
            "hidden_dim": args.hidden_dim,
            "esm": args.esm,
            "multitask": args.multitask,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "targets": target_names,
        "metrics": all_metrics,
    }
    with open("checkpoints/results.json", "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved metrics to checkpoints/results.json")

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
        results = predict_peptide(model, seq, target_names, device,
                                  feature_dim=feature_dim)
        if results:
            best_target = min(results.items(), key=lambda x: x[1]['IC50_uM'])
            print(f"{seq:<12} {best_target[0]:<28} {best_target[1]['pIC50']:>8.2f} {best_target[1]['IC50_uM']:>10.1f}  ({desc})")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for target, metrics in sorted(all_metrics.items()):
        r2 = metrics.get("R2")
        status = "GOOD" if r2 and r2 > 0.3 else "WEAK"
        r2_str = str(r2) if r2 is not None else "N/A"
        rmse_str = str(metrics.get("RMSE", "N/A"))
        print(f"  [{status}] {target}: R2={r2_str}, RMSE={rmse_str}, N={metrics.get('n', 0)}")

    esm_tag = "+ESM" if args.esm else ""
    mt_tag = "+multitask" if args.multitask else ""
    print(f"\nModel saved to: checkpoints/meal_shield_gnn.pt")
    print(f"Config: feature_dim={feature_dim}, hidden={args.hidden_dim}{esm_tag}{mt_tag}")
    print(f"\nUsage examples:")
    print(f"  python train.py                          # Basic per-target training")
    print(f"  python train.py --esm --multitask        # ESM + multi-task (recommended)")
    print(f"  python train.py --esm --cv 5             # 5-fold cross-validation")
    print(f"  python train.py --esm --multitask --epochs 200 --lr 5e-4")


if __name__ == "__main__":
    main()
