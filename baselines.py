"""
Meal Shield — Baseline Models

Random Forest and Ridge Regression baselines using molecular fingerprints
and amino acid composition features. Provides a comparison floor for
the GNN models.

Usage:
    python baselines.py                   # Run all baselines
    python baselines.py --model rf        # Only Random Forest
    python baselines.py --model ridge     # Only Ridge Regression
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Amino acid physicochemical descriptors (same order as data.py AA_FEATURES)
AA_PROPS = {
    "A": [89.1,  1.8,  0.0, 0, 0, 71.8,  6.01, 0],
    "R": [174.2, -4.5, 1.0, 0, 5, 148.0, 10.76, 0],
    "N": [132.1, -3.5, 0.0, 0, 2, 114.0, 5.41, 0],
    "D": [133.1, -3.5, -1.0, 0, 1, 111.0, 2.77, 1],
    "C": [121.2, 2.5,  0.0, 0, 1, 108.5, 5.07, 0],
    "E": [147.1, -3.5, -1.0, 0, 1, 138.4, 3.22, 1],
    "Q": [146.2, -3.5, 0.0, 0, 2, 143.8, 5.65, 0],
    "G": [75.0,  -0.4, 0.0, 0, 0, 60.1,  5.97, 0],
    "H": [155.2, -3.2, 0.0, 1, 2, 153.2, 7.59, 0],
    "I": [131.2, 4.5,  0.0, 0, 0, 166.7, 6.02, 0],
    "L": [131.2, 3.8,  0.0, 0, 0, 166.7, 5.98, 0],
    "K": [146.2, -3.9, 1.0, 0, 2, 168.6, 9.74, 0],
    "M": [149.2, 1.9,  0.0, 0, 0, 162.9, 5.74, 0],
    "F": [165.2, 2.8,  0.0, 1, 0, 189.9, 5.48, 0],
    "P": [115.1, -1.6, 0.0, 0, 0, 129.0, 6.30, 1],
    "S": [105.1, -0.8, 0.0, 0, 1, 89.0,  5.68, 0],
    "T": [119.1, -0.7, 0.0, 0, 1, 116.1, 5.60, 0],
    "W": [204.2, -0.9, 0.0, 1, 1, 227.8, 5.89, 0],
    "Y": [181.2, -1.3, 0.0, 1, 1, 193.6, 5.66, 0],
    "V": [117.1, 4.2,  0.0, 0, 0, 140.0, 5.97, 0],
}

STANDARD_AA = sorted(AA_PROPS.keys())


# ============================================================
# Feature Extraction
# ============================================================

def aa_composition(sequence):
    """
    Amino acid composition vector (20-dim, normalized).
    """
    counts = {aa: 0 for aa in STANDARD_AA}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
    n = max(len(sequence), 1)
    return np.array([counts[aa] / n for aa in STANDARD_AA])


def dipeptide_composition(sequence):
    """
    Dipeptide composition (400-dim, normalized).
    Captures local sequence patterns.
    """
    dipeptides = [a + b for a in STANDARD_AA for b in STANDARD_AA]
    dp_idx = {dp: i for i, dp in enumerate(dipeptides)}

    counts = np.zeros(400)
    for i in range(len(sequence) - 1):
        dp = sequence[i:i+2]
        if dp in dp_idx:
            counts[dp_idx[dp]] += 1

    n = max(len(sequence) - 1, 1)
    return counts / n


def physicochemical_features(sequence):
    """
    Aggregate physicochemical descriptors across residues.
    Returns mean, std, min, max of each property (8 props x 4 stats = 32 features).
    """
    if not sequence or not all(aa in AA_PROPS for aa in sequence):
        return np.zeros(32)

    props = np.array([AA_PROPS[aa] for aa in sequence])

    features = []
    for i in range(props.shape[1]):
        col = props[:, i]
        features.extend([col.mean(), col.std(), col.min(), col.max()])

    return np.array(features)


def global_features(sequence):
    """
    Global peptide features: length, MW, net charge, hydrophobicity, etc.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(8)

    props = np.array([AA_PROPS.get(aa, [0]*8) for aa in sequence])

    total_mw = props[:, 0].sum() - (n - 1) * 18.015  # Subtract water for peptide bonds
    avg_hydrophobicity = props[:, 1].mean()
    net_charge = props[:, 2].sum()
    n_aromatic = props[:, 3].sum()
    n_hbond = props[:, 4].sum()
    total_volume = props[:, 5].sum()
    avg_pi = props[:, 6].mean()
    n_rigid = props[:, 7].sum()

    return np.array([
        n, total_mw, avg_hydrophobicity, net_charge,
        n_aromatic, n_hbond, total_volume / n, n_rigid / n,
    ])


def peptide_features(sequence):
    """
    Full feature vector for a peptide sequence.
    Concatenates: AA composition (20) + dipeptide (400) + physicochemical (32) + global (8) = 460 dims.
    """
    return np.concatenate([
        aa_composition(sequence),           # 20
        dipeptide_composition(sequence),    # 400
        physicochemical_features(sequence), # 32
        global_features(sequence),          # 8
    ])


def smiles_fingerprint(smiles, nbits=2048):
    """
    Morgan fingerprint (ECFP4) for small molecules.
    Falls back to zeros if RDKit fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        return np.array(fp)
    except Exception:
        return np.zeros(nbits)


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_baseline_data(chembl_df, peptides, target_name, max_per_target=2000):
    """
    Build feature matrix X and label vector y for a specific target.
    Combines small molecules (fingerprints) and peptides (sequence features).
    """
    X_list = []
    y_list = []

    # Small molecules from ChEMBL
    target_df = chembl_df[chembl_df["target"] == target_name].head(max_per_target)
    for _, row in target_df.iterrows():
        fp = smiles_fingerprint(row["smiles"])
        # Pad to match peptide feature dim (or vice versa)
        # Use fingerprint directly for molecules
        X_list.append(fp)
        y_list.append(row["pIC50"])

    # Peptides
    for pep in peptides:
        if pep.get("target") != target_name:
            continue
        seq = pep.get("sequence", "")
        if not seq:
            continue
        # Use peptide features, padded/truncated to same dim as fingerprints
        feat = peptide_features(seq)
        if len(feat) < 2048:
            feat = np.concatenate([feat, np.zeros(2048 - len(feat))])
        else:
            feat = feat[:2048]
        X_list.append(feat)
        y_list.append(pep["pIC50"])

    if not X_list:
        return None, None, None, None

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# Models
# ============================================================

def train_random_forest(X_train, y_train, n_estimators=500, max_depth=20):
    """Train a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, alpha=1.0):
    """Train a Ridge Regression model with feature scaling."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate_baseline(model, X_test, y_test, scaler=None):
    """Evaluate a baseline model."""
    X = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X)

    if len(y_test) < 2:
        return {"R2": None, "RMSE": None, "n": len(y_test)}

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "n": len(y_test),
    }


def cross_validate_baselines(peptides, n_folds=5, include_inactive=False):
    """
    Run baseline models with stratified k-fold CV on peptide data only.
    Returns dict: model_name -> target -> {R2_mean, R2_std, RMSE_mean, RMSE_std}
    """
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict

    if not include_inactive:
        peptides = [p for p in peptides if p["target"] != "inactive"]

    X_list, y_list, t_list = [], [], []
    for pep in peptides:
        X_list.append(peptide_features(pep["sequence"]))
        y_list.append(pep["pIC50"])
        t_list.append(pep["target"])

    X = np.array(X_list)
    y = np.array(y_list)
    targets = np.array(t_list)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    unique_targets = sorted(set(targets))

    for model_name, model_template in models.items():
        print(f"\n--- {model_name} (k-fold CV) ---")
        target_metrics = defaultdict(lambda: {"R2": [], "RMSE": []})

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, targets)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            t_test = targets[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = type(model_template)(**model_template.get_params())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for target in unique_targets:
                mask = t_test == target
                if mask.sum() < 2:
                    continue
                r2 = r2_score(y_test[mask], y_pred[mask])
                rmse = mean_squared_error(y_test[mask], y_pred[mask]) ** 0.5
                target_metrics[target]["R2"].append(r2)
                target_metrics[target]["RMSE"].append(rmse)

        results[model_name] = {}
        for target in unique_targets:
            tm = target_metrics.get(target)
            if not tm or not tm["R2"]:
                continue
            r2s = np.array(tm["R2"])
            rmses = np.array(tm["RMSE"])
            results[model_name][target] = {
                "R2_mean": round(float(r2s.mean()), 4),
                "R2_std": round(float(r2s.std()), 4),
                "RMSE_mean": round(float(rmses.mean()), 4),
                "RMSE_std": round(float(rmses.std()), 4),
            }
            print(f"  {target:<28} R2={r2s.mean():.4f}+/-{r2s.std():.4f}  "
                  f"RMSE={rmses.mean():.4f}+/-{rmses.std():.4f}")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline models for comparison")
    parser.add_argument("--model", choices=["rf", "ridge", "all"], default="all",
                        help="Which baseline to run")
    parser.add_argument("--cv", action="store_true",
                        help="Run k-fold cross-validation on peptides")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", default="checkpoints/baseline_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("MEAL SHIELD — BASELINE MODELS")
    print("=" * 60)

    from data import download_all_data, TARGETS, load_food_peptides
    peptides = load_food_peptides()

    if args.cv:
        print(f"\nRunning {args.folds}-fold cross-validation on peptide data...")
        cv_results = cross_validate_baselines(peptides, n_folds=args.folds)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"\nSaved to {args.output}")
        return

    # Load data
    chembl_df = download_all_data()

    target_names = list(TARGETS.keys())
    results = {}

    for target in target_names:
        print(f"\n--- {target} ---")
        X_train, X_test, y_train, y_test = prepare_baseline_data(
            chembl_df, peptides, target
        )

        if X_train is None:
            print(f"  No data for {target}")
            continue

        print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")
        results[target] = {}

        # Random Forest
        if args.model in ("rf", "all"):
            rf = train_random_forest(X_train, y_train)
            rf_metrics = evaluate_baseline(rf, X_test, y_test)
            results[target]["random_forest"] = rf_metrics
            print(f"  RF:    R2={rf_metrics['R2']}, RMSE={rf_metrics['RMSE']}")

        # Ridge Regression
        if args.model in ("ridge", "all"):
            ridge, scaler = train_ridge(X_train, y_train)
            ridge_metrics = evaluate_baseline(ridge, X_test, y_test, scaler)
            results[target]["ridge"] = ridge_metrics
            print(f"  Ridge: R2={ridge_metrics['R2']}, RMSE={ridge_metrics['RMSE']}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':<30} {'RF R2':>8} {'Ridge R2':>10} {'RF RMSE':>10}")
    print("-" * 62)
    for target, m in sorted(results.items()):
        rf_r2 = m.get("random_forest", {}).get("R2")
        ridge_r2 = m.get("ridge", {}).get("R2")
        rf_rmse = m.get("random_forest", {}).get("RMSE")
        rf_str = f"{rf_r2:.4f}" if rf_r2 is not None else "N/A"
        ri_str = f"{ridge_r2:.4f}" if ridge_r2 is not None else "N/A"
        rm_str = f"{rf_rmse:.4f}" if rf_rmse is not None else "N/A"
        print(f"{target:<30} {rf_str:>8} {ri_str:>10} {rm_str:>10}")

    print(f"\nSaved to {args.output}")
    print("\nThese baselines establish the floor. The GNN should beat them")
    print("by learning structural features that fingerprints/composition miss.")


if __name__ == "__main__":
    main()
