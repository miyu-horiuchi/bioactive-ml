"""
Meal Shield GNN — Data Pipeline
Pulls enzyme inhibitor data from ChEMBL and builds molecular graphs.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# ============================================================
# 1. Pull data from ChEMBL REST API
# ============================================================

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

# Target ChEMBL IDs for our 4 meal-shield targets
TARGETS = {
    "alpha_glucosidase": {
        "chembl_id": "CHEMBL4979",       # Human maltase-glucoamylase
        "alt_ids": ["CHEMBL6640"],        # Alpha-glucosidase (yeast, commonly used)
        "description": "Alpha-glucosidase — blocks starch→glucose"
    },
    "lipase": {
        "chembl_id": "CHEMBL4822",       # Human pancreatic lipase
        "alt_ids": [],
        "description": "Pancreatic lipase — blocks fat digestion"
    },
    "bile_acid_receptor": {
        "chembl_id": "CHEMBL4829",       # FXR (farnesoid X receptor) — bile acid signaling
        "alt_ids": ["CHEMBL2056"],       # TGR5 (bile acid receptor)
        "description": "Bile acid receptor — modulates bile acid signaling"
    },
    "sodium_hydrogen_exchanger": {
        "chembl_id": "CHEMBL4145",       # NHE3 (Sodium-hydrogen exchanger 3)
        "alt_ids": [],
        "description": "NHE3 — sodium absorption in gut"
    },
    "ace_inhibitor": {
        "chembl_id": "CHEMBL1808",       # Angiotensin-converting enzyme (ACE)
        "alt_ids": ["CHEMBL4525"],       # ACE-2
        "description": "ACE — angiotensin-converting enzyme (blood pressure)"
    },
    "dpp4_inhibitor": {
        "chembl_id": "CHEMBL284",        # Dipeptidyl peptidase IV (DPP-4)
        "alt_ids": [],
        "description": "DPP-4 — dipeptidyl peptidase IV (blood sugar)"
    },
}


def fetch_chembl_activities(target_chembl_id, max_records=5000):
    """Fetch bioactivity data from ChEMBL for a given target."""
    url = f"{CHEMBL_API}/activity.json"
    all_results = []
    offset = 0
    limit = 1000

    while offset < max_records:
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type__in": "IC50,Ki,Kd",
            "standard_relation": "=",
            "limit": limit,
            "offset": offset,
            "format": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            activities = data.get("activities", [])
            if not activities:
                break
            all_results.extend(activities)
            offset += limit
            print(f"  Fetched {len(all_results)} records for {target_chembl_id}...")
        except Exception as e:
            print(f"  Error fetching {target_chembl_id}: {e}")
            break

    return all_results


def _convert_to_nM(value, units):
    """Convert a bioactivity value to nM. Returns None if units unrecognized."""
    conversions = {
        "nM": 1.0,
        "uM": 1e3,
        "mM": 1e6,
        "pM": 1e-3,
        "M": 1e9,
        "ug.mL-1": None,  # Needs MW — skip
        "ug/mL": None,
    }
    factor = conversions.get(units)
    if factor is None:
        return None
    return value * factor


def process_chembl_data(activities, target_name):
    """Convert raw ChEMBL activities to a clean DataFrame."""
    records = []
    skipped_units = {}
    for act in activities:
        smiles = act.get("canonical_smiles")
        value = act.get("standard_value")
        units = act.get("standard_units")
        std_type = act.get("standard_type")

        if not (smiles and value):
            continue

        try:
            raw_value = float(value)
        except (ValueError, TypeError):
            continue

        if raw_value <= 0:
            continue

        value_nM = _convert_to_nM(raw_value, units)
        if value_nM is None:
            skipped_units[units] = skipped_units.get(units, 0) + 1
            continue

        pIC50 = 9 - np.log10(value_nM)  # Convert to pIC50
        records.append({
            "smiles": smiles,
            "value_nM": value_nM,
            "pIC50": pIC50,
            "target": target_name,
            "type": std_type,
            "chembl_id": act.get("molecule_chembl_id", ""),
        })

    if skipped_units:
        print(f"  Skipped units for {target_name}: {skipped_units}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        # Deduplicate: keep median value per SMILES per target
        df = df.groupby(["smiles", "target"]).agg({
            "pIC50": "median",
            "value_nM": "median",
            "chembl_id": "first",
            "type": "first",
        }).reset_index()

    return df


def download_all_data(data_dir="data"):
    """Download and process data for all targets."""
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, "chembl_combined.csv")

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)

    all_dfs = []
    for target_name, info in TARGETS.items():
        print(f"\nFetching {target_name} ({info['description']})...")

        # Fetch from primary target ID
        all_ids = [info["chembl_id"]] + info.get("alt_ids", [])
        activities = []
        for tid in all_ids:
            activities.extend(fetch_chembl_activities(tid))

        df = process_chembl_data(activities, target_name)
        print(f"  → {len(df)} unique compounds for {target_name}")
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(cache_file, index=False)
    print(f"\nTotal: {len(combined)} activity records saved to {cache_file}")
    return combined


# ============================================================
# 2. Food peptide data — loaded from CSV
# ============================================================

def load_food_peptides(csv_path="data/food_peptides.csv", include_inactive=True):
    """
    Load food peptide data from the curated/scraped CSV.
    Returns list of dicts compatible with build_dataset().

    Args:
        csv_path: path to CSV file
        include_inactive: if True, keep inactive peptides as negative examples
                          (they get high IC50 / low pIC50 labels, which teaches
                          the model what inactivity looks like)
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Run fetch_peptides.py first.")
        return []

    df = pd.read_csv(csv_path)
    peptides = []
    n_inactive = 0
    for _, row in df.iterrows():
        if pd.isna(row.get("pIC50")) or pd.isna(row.get("sequence")):
            continue
        activity = row["activity"]
        if activity == "inactive":
            if not include_inactive:
                continue
            n_inactive += 1
        peptides.append({
            "sequence": row["sequence"],
            "target": activity,
            "pIC50": float(row["pIC50"]),
            "source": row.get("source", "unknown"),
        })

    inactive_note = f" (including {n_inactive} inactive)" if n_inactive else ""
    print(f"Loaded {len(peptides)} food peptides from {csv_path}{inactive_note}")
    activities = {}
    for p in peptides:
        activities[p["target"]] = activities.get(p["target"], 0) + 1
    for act, cnt in sorted(activities.items()):
        print(f"  {act}: {cnt}")
    return peptides


# Backwards compatibility: load from CSV at import time if available
KNOWN_FOOD_PEPTIDES = load_food_peptides()


# ============================================================
# 3. Molecular graph construction
# ============================================================

# Amino acid one-letter to SMILES mapping (for converting peptides to molecular graphs)
AA_SMILES = {
    'A': 'C(C(=O)O)N',                       # Alanine
    'R': 'C(CC/N=C(\\N)N)C(C(=O)O)N',       # Arginine
    'N': 'C(C(=O)N)C(C(=O)O)N',             # Asparagine
    'D': 'C(C(=O)O)C(C(=O)O)N',             # Aspartate
    'C': 'C(CS)C(C(=O)O)N',                  # Cysteine
    'E': 'C(CC(=O)O)C(C(=O)O)N',            # Glutamate
    'Q': 'C(CC(=O)N)C(C(=O)O)N',            # Glutamine
    'G': 'C(C(=O)O)N',                       # Glycine
    'H': 'c1c[nH]cn1CC(C(=O)O)N',           # Histidine
    'I': 'CCC(C)C(C(=O)O)N',                # Isoleucine
    'L': 'CC(C)CC(C(=O)O)N',                # Leucine
    'K': 'C(CCN)CC(C(=O)O)N',               # Lysine
    'M': 'CSCCC(C(=O)O)N',                  # Methionine
    'F': 'c1ccc(CC(C(=O)O)N)cc1',           # Phenylalanine
    'P': 'C1CC(NC1)C(=O)O',                 # Proline
    'S': 'C(C(C(=O)O)N)O',                  # Serine
    'T': 'CC(C(C(=O)O)N)O',                 # Threonine
    'W': 'c1ccc2c(c1)c(c[nH]2)CC(C(=O)O)N', # Tryptophan
    'Y': 'c1cc(ccc1CC(C(=O)O)N)O',          # Tyrosine
    'V': 'CC(C)C(C(=O)O)N',                 # Valine
}

# Amino acid physicochemical features (for residue-level graphs)
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
    'P': [115.1, -1.6,  0.0,  0, 0, 129.0, 6.30, 1],  # proline = rigid
    'S': [105.1, -0.8,  0.0,  0, 1, 89.0,  5.68, 0],
    'T': [119.1, -0.7,  0.0,  0, 1, 116.1, 5.60, 0],
    'W': [204.2, -0.9,  0.0,  1, 1, 227.8, 5.89, 0],
    'Y': [181.2, -1.3,  0.0,  1, 1, 193.6, 5.66, 0],
    'V': [117.1,  4.2,  0.0,  0, 0, 140.0, 5.97, 0],
}
# Features: [MW, hydrophobicity(Kyte-Doolittle), charge_pH7, aromatic, H-bond_donors, volume, pI, is_rigid]


def smiles_to_graph(smiles):
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (per atom)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            int(atom.GetAtomicNum() == 6),   # C
            int(atom.GetAtomicNum() == 7),   # N
            int(atom.GetAtomicNum() == 8),   # O
            int(atom.GetAtomicNum() == 16),  # S
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index (bonds, undirected)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

        bond_feat = [
            int(bond.GetBondType() == Chem.BondType.SINGLE),
            int(bond.GetBondType() == Chem.BondType.DOUBLE),
            int(bond.GetBondType() == Chem.BondType.TRIPLE),
            int(bond.GetBondType() == Chem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        edge_attrs.extend([bond_feat, bond_feat])

    if len(edge_indices) == 0:
        return None

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def peptide_to_graph(sequence, use_residue_level=True, esm_cache=None):
    """
    Convert a peptide sequence to a molecular graph.

    Two modes:
    - residue_level=True:  Each amino acid = one node (fast, good for short peptides)
    - residue_level=False: Each atom = one node (detailed, uses RDKit)

    If esm_cache is provided, residue-level graphs get ESM-2 embeddings
    concatenated to the physicochemical features (8 + 320 = 328 dim).
    """
    if use_residue_level:
        return _peptide_residue_graph(sequence, esm_cache=esm_cache)
    else:
        return _peptide_atom_graph(sequence)


def _peptide_residue_graph(sequence, esm_cache=None):
    """
    Residue-level graph: nodes = amino acids, edges = sequential + proximity.

    If esm_cache is provided and contains the sequence, the 8-dim physicochemical
    features are concatenated with 320-dim ESM-2 embeddings per residue (total: 328).
    """
    if not all(aa in AA_FEATURES for aa in sequence):
        return None

    # Node features: physicochemical (8-dim)
    node_feats = []
    for aa in sequence:
        node_feats.append(AA_FEATURES[aa])

    x = torch.tensor(node_feats, dtype=torch.float)

    # Concatenate ESM-2 embeddings if available (8 + 320 = 328 dim)
    if esm_cache is not None and sequence in esm_cache:
        esm_emb = esm_cache[sequence]  # (seq_len, 320)
        if esm_emb.shape[0] == len(sequence):
            x = torch.cat([x, esm_emb], dim=-1)

    # Edges: sequential bonds + skip connections (k=3)
    edge_indices = []
    for i in range(len(sequence) - 1):
        edge_indices.extend([[i, i + 1], [i + 1, i]])

    k = 3
    for i in range(len(sequence)):
        for j in range(i + 2, min(i + k + 1, len(sequence))):
            edge_indices.extend([[i, j], [j, i]])

    if len(edge_indices) == 0:
        # Single amino acid — self-loop
        edge_indices = [[0, 0]]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def _peptide_atom_graph(sequence):
    """Atom-level graph: build full molecular structure of the peptide."""
    # Build peptide SMILES by joining amino acids
    # (simplified — uses individual AA SMILES, not true peptide bonds)
    smiles_parts = []
    for aa in sequence:
        if aa in AA_SMILES:
            smiles_parts.append(AA_SMILES[aa])
    if not smiles_parts:
        return None

    # For short peptides, concatenate and let RDKit parse
    combined = ".".join(smiles_parts)
    return smiles_to_graph(combined)


# ============================================================
# 4. Build dataset
# ============================================================

def build_dataset(chembl_df, peptide_list=None, max_per_target=2000, esm_cache=None):
    """
    Build a combined dataset of molecular graphs with activity labels.

    Args:
        chembl_df: DataFrame of ChEMBL bioactivity data.
        peptide_list: List of peptide dicts with 'sequence', 'target', 'pIC50'.
        max_per_target: Max compounds per ChEMBL target.
        esm_cache: Dict mapping sequence -> ESM-2 embedding tensor.
                   If provided, peptide node features are augmented (8 + 320 = 328 dim).

    Returns:
        (graphs, stats) tuple.
    """
    graphs = []
    stats = {}

    # Process ChEMBL small molecules
    for target_name in chembl_df["target"].unique():
        target_df = chembl_df[chembl_df["target"] == target_name].head(max_per_target)
        count = 0
        for _, row in target_df.iterrows():
            graph = smiles_to_graph(row["smiles"])
            if graph is not None:
                graph.y = torch.tensor([row["pIC50"]], dtype=torch.float)
                graph.target_name = target_name
                graph.mol_type = "small_molecule"
                graphs.append(graph)
                count += 1
        stats[target_name] = count
        print(f"  {target_name}: {count} molecular graphs")

    # Process food peptides (with optional ESM-2 embeddings)
    if peptide_list:
        pep_count = 0
        for pep in peptide_list:
            graph = peptide_to_graph(pep["sequence"], use_residue_level=True,
                                    esm_cache=esm_cache)
            if graph is not None:
                graph.y = torch.tensor([pep["pIC50"]], dtype=torch.float)
                graph.target_name = pep["target"]
                graph.mol_type = "peptide"
                graph.sequence = pep["sequence"]
                graphs.append(graph)
                pep_count += 1
        stats["food_peptides"] = pep_count
        print(f"  food_peptides: {pep_count} peptide graphs")
        if esm_cache:
            print(f"  (ESM-2 embeddings: {sum(1 for p in peptide_list if p['sequence'] in esm_cache)}/{len(peptide_list)} sequences)")

    print(f"\nTotal dataset: {len(graphs)} graphs")
    return graphs, stats


if __name__ == "__main__":
    print("=" * 60)
    print("MEAL SHIELD GNN — Data Pipeline")
    print("=" * 60)

    # Step 1: Download ChEMBL data
    print("\n[1/3] Downloading ChEMBL bioactivity data...")
    chembl_data = download_all_data(data_dir="data")
    print(f"\nChEMBL data shape: {chembl_data.shape}")
    print(chembl_data["target"].value_counts())

    # Step 2: Build graphs
    print("\n[2/3] Building molecular graphs...")
    graphs, stats = build_dataset(chembl_data, KNOWN_FOOD_PEPTIDES)

    # Step 3: Save
    print("\n[3/3] Saving dataset...")
    os.makedirs("data", exist_ok=True)
    torch.save(graphs, "data/meal_shield_graphs.pt")
    with open("data/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n✓ Dataset saved to data/meal_shield_graphs.pt")
    print(f"✓ Stats: {stats}")
