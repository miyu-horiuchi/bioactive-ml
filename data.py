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


def process_chembl_data(activities, target_name):
    """Convert raw ChEMBL activities to a clean DataFrame."""
    records = []
    for act in activities:
        smiles = act.get("canonical_smiles")
        value = act.get("standard_value")
        units = act.get("standard_units")
        std_type = act.get("standard_type")

        if smiles and value and units == "nM":
            try:
                value_nM = float(value)
                if value_nM > 0:
                    pIC50 = 9 - np.log10(value_nM)  # Convert to pIC50
                    records.append({
                        "smiles": smiles,
                        "value_nM": value_nM,
                        "pIC50": pIC50,
                        "target": target_name,
                        "type": std_type,
                        "chembl_id": act.get("molecule_chembl_id", ""),
                    })
            except (ValueError, TypeError):
                continue

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
# 2. Food peptide data (BIOPEP-style curated set)
# ============================================================

# Curated set of known bioactive food peptides with activities
# (From BIOPEP-UWM and published literature)
KNOWN_FOOD_PEPTIDES = [
    # Alpha-glucosidase inhibitors (from food sources)
    {"sequence": "IPAVF",   "target": "alpha_glucosidase", "pIC50": 4.5, "source": "black bean"},
    {"sequence": "AKSPLF",  "target": "alpha_glucosidase", "pIC50": 4.2, "source": "wheat gluten"},
    {"sequence": "PPYIL",   "target": "alpha_glucosidase", "pIC50": 4.0, "source": "quinoa"},
    {"sequence": "LRSELAAWSR", "target": "alpha_glucosidase", "pIC50": 3.8, "source": "rice"},
    {"sequence": "GGSK",    "target": "alpha_glucosidase", "pIC50": 3.5, "source": "soybean"},
    {"sequence": "EAK",     "target": "alpha_glucosidase", "pIC50": 3.3, "source": "wheat"},
    {"sequence": "KLPGF",   "target": "alpha_glucosidase", "pIC50": 4.8, "source": "silk"},
    {"sequence": "SVPA",    "target": "alpha_glucosidase", "pIC50": 3.9, "source": "egg"},

    # Lipase inhibitors (from food sources)
    {"sequence": "PAGNFLPP", "target": "lipase", "pIC50": 4.1, "source": "soybean"},
    {"sequence": "GPVRGPFPIIV", "target": "lipase", "pIC50": 3.6, "source": "casein"},
    {"sequence": "VFPS",    "target": "lipase", "pIC50": 3.4, "source": "tuna"},
    {"sequence": "YALPHA",  "target": "lipase", "pIC50": 3.2, "source": "whey"},

    # ACE inhibitors (well-studied, useful for transfer learning)
    {"sequence": "IPP",     "target": "ace_inhibitor", "pIC50": 5.1, "source": "fermented milk (Calpis)"},
    {"sequence": "VPP",     "target": "ace_inhibitor", "pIC50": 4.9, "source": "fermented milk (Calpis)"},
    {"sequence": "LKP",     "target": "ace_inhibitor", "pIC50": 5.5, "source": "bonito"},
    {"sequence": "IKP",     "target": "ace_inhibitor", "pIC50": 5.3, "source": "chicken"},
    {"sequence": "VY",      "target": "ace_inhibitor", "pIC50": 4.6, "source": "sardine"},
    {"sequence": "IY",      "target": "ace_inhibitor", "pIC50": 4.4, "source": "wheat"},
    {"sequence": "LKPNM",   "target": "ace_inhibitor", "pIC50": 5.0, "source": "bonito"},
    {"sequence": "FQKVVA",  "target": "ace_inhibitor", "pIC50": 4.7, "source": "chicken"},

    # DPP-4 inhibitors (blood sugar regulation)
    {"sequence": "LPYPY",   "target": "dpp4_inhibitor", "pIC50": 4.3, "source": "gouda cheese"},
    {"sequence": "IPAVFK",  "target": "dpp4_inhibitor", "pIC50": 4.0, "source": "milk"},
    {"sequence": "WR",      "target": "dpp4_inhibitor", "pIC50": 4.5, "source": "various"},
    {"sequence": "VAGTWY",  "target": "dpp4_inhibitor", "pIC50": 3.8, "source": "tuna"},
    {"sequence": "FLQP",    "target": "dpp4_inhibitor", "pIC50": 4.1, "source": "wheat"},
]


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


def peptide_to_graph(sequence, use_residue_level=True):
    """
    Convert a peptide sequence to a molecular graph.

    Two modes:
    - residue_level=True:  Each amino acid = one node (fast, good for short peptides)
    - residue_level=False: Each atom = one node (detailed, uses RDKit)
    """
    if use_residue_level:
        return _peptide_residue_graph(sequence)
    else:
        return _peptide_atom_graph(sequence)


def _peptide_residue_graph(sequence):
    """Residue-level graph: nodes = amino acids, edges = sequential + proximity."""
    if not all(aa in AA_FEATURES for aa in sequence):
        return None

    # Node features
    node_feats = []
    for aa in sequence:
        node_feats.append(AA_FEATURES[aa])

    x = torch.tensor(node_feats, dtype=torch.float)

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

def build_dataset(chembl_df, peptide_list=None, max_per_target=2000):
    """
    Build a combined dataset of molecular graphs with activity labels.
    Returns list of Data objects.
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

    # Process food peptides
    if peptide_list:
        pep_count = 0
        for pep in peptide_list:
            graph = peptide_to_graph(pep["sequence"], use_residue_level=True)
            if graph is not None:
                graph.y = torch.tensor([pep["pIC50"]], dtype=torch.float)
                graph.target_name = pep["target"]
                graph.mol_type = "peptide"
                graph.sequence = pep["sequence"]
                graphs.append(graph)
                pep_count += 1
        stats["food_peptides"] = pep_count
        print(f"  food_peptides: {pep_count} peptide graphs")

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
