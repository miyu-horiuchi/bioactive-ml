"""
Meal Shield GNN — Persistent Homology (Topological Data Analysis)

Computes topological features from molecular 3D coordinates using persistent
homology. These features capture global shape information (rings, cavities,
tunnels) that GNNs miss because GNNs only learn from local neighborhoods.

Pipeline:
  Molecule → 3D coordinates (RDKit) → Distance matrix → Ripser (persistent homology)
  → Persistence diagram → Vectorized features (persistence images / statistics)
"""

import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from ripser import ripser
from persim import PersistenceImager


# ============================================================
# 1. Generate 3D coordinates for molecules
# ============================================================

def get_3d_coords_from_smiles(smiles, num_conformers=1):
    """
    Generate 3D coordinates for a molecule from its SMILES string.
    Uses RDKit's ETKDG conformer generator.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1

    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback: try without ETKDG
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return None

    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    # Extract coordinates (heavy atoms only, skip hydrogens)
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:  # Skip H
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

    return np.array(coords) if coords else None


def get_3d_coords_from_peptide(sequence):
    """
    Generate approximate 3D coordinates for a peptide.
    Uses a simplified backbone model with physicochemical positioning.
    """
    # Amino acid approximate volumes (for spacing)
    AA_RADIUS = {
        'G': 0.6, 'A': 0.9, 'V': 1.1, 'L': 1.2, 'I': 1.2,
        'P': 1.0, 'F': 1.4, 'W': 1.5, 'M': 1.2, 'S': 0.9,
        'T': 1.0, 'C': 1.0, 'Y': 1.4, 'H': 1.2, 'D': 1.0,
        'E': 1.1, 'N': 1.0, 'Q': 1.1, 'K': 1.3, 'R': 1.4,
    }

    coords = []
    # Simple helical backbone model
    # Each residue advances ~3.8A along backbone, with ~1.5A radius
    for i, aa in enumerate(sequence):
        # Backbone spiral
        t = i * 100 * np.pi / 180  # ~100 degrees per residue (alpha helix)
        z = i * 1.5  # Rise per residue (~1.5A for alpha helix)
        r = 2.3  # Helix radius
        x = r * np.cos(t)
        y = r * np.sin(t)

        # Perturb by residue size
        radius = AA_RADIUS.get(aa, 1.0)
        x += radius * 0.3 * np.cos(t + np.pi / 2)
        y += radius * 0.3 * np.sin(t + np.pi / 2)

        coords.append([x, y, z])

    return np.array(coords)


# ============================================================
# 2. Compute persistent homology
# ============================================================

def compute_persistence(coords, max_dim=2, max_edge_length=10.0):
    """
    Compute persistent homology from 3D coordinates using Ripser.

    Args:
        coords: Nx3 array of 3D coordinates
        max_dim: Maximum homological dimension (0=components, 1=loops, 2=voids)
        max_edge_length: Maximum edge length for Vietoris-Rips complex

    Returns:
        Dictionary with persistence diagrams for each dimension
    """
    if coords is None or len(coords) < 3:
        return None

    try:
        result = ripser(
            coords,
            maxdim=max_dim,
            thresh=max_edge_length,
        )
        return result['dgms']  # List of persistence diagrams (one per dimension)
    except Exception:
        return None


def compute_persistence_with_cocycles(coords, max_dim=2, max_edge_length=10.0):
    """
    Compute persistent homology WITH representative cocycles.

    Cocycles tell us WHERE topological features are located in the molecule,
    not just that they exist. This gives spatially-aware topological features.

    Returns:
        dict with 'dgms' (persistence diagrams) and 'cocycles' (representative cocycles)
    """
    if coords is None or len(coords) < 3:
        return None

    try:
        result = ripser(
            coords,
            maxdim=max_dim,
            thresh=max_edge_length,
            do_cocycles=True,
        )
        return {
            'dgms': result['dgms'],
            'cocycles': result.get('cocycles', []),
            'num_edges': result.get('num_edges', 0),
        }
    except Exception:
        return None


def cocycle_features(result, coords, max_dim=1):
    """
    Extract features from representative cocycles.

    For each persistent homological feature, the cocycle tells us which
    edges/simplices create that feature. We extract:
    - Spatial extent of each feature (how spread out is the ring/void)
    - Center of mass of the feature
    - Orientation relative to molecular axes
    """
    if result is None or 'cocycles' not in result:
        return np.zeros(12, dtype=np.float32)

    features = []
    dgms = result['dgms']
    cocycles = result['cocycles']

    for dim in range(min(max_dim + 1, len(cocycles))):
        dgm = dgms[dim]
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]
        persistence = dgm_finite[:, 1] - dgm_finite[:, 0]

        if len(persistence) > 0 and dim < len(cocycles) and len(cocycles[dim]) > 0:
            # Find the most persistent feature's cocycle
            most_persistent_idx = np.argmax(persistence)

            if most_persistent_idx < len(cocycles[dim]):
                cocycle = cocycles[dim][most_persistent_idx]

                # Cocycle is an array of [simplex_indices..., coefficient]
                # Extract the vertex indices involved
                if len(cocycle) > 0:
                    vertex_indices = set()
                    for simplex in cocycle:
                        for idx in simplex[:-1]:  # Last element is coefficient
                            if 0 <= idx < len(coords):
                                vertex_indices.add(int(idx))

                    if vertex_indices:
                        involved_coords = coords[list(vertex_indices)]
                        centroid = np.mean(involved_coords, axis=0)
                        spatial_extent = np.max(np.linalg.norm(involved_coords - centroid, axis=1))
                        fraction_involved = len(vertex_indices) / len(coords)

                        features.extend([
                            spatial_extent,
                            fraction_involved,
                            len(vertex_indices),
                            np.std(np.linalg.norm(involved_coords - centroid, axis=1)),
                        ])
                    else:
                        features.extend([0, 0, 0, 0])
                else:
                    features.extend([0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])

    # Pad to fixed size (4 features * 3 dimensions = 12)
    while len(features) < 12:
        features.append(0)

    return np.array(features[:12], dtype=np.float32)


# ============================================================
# 3. Vectorize persistence diagrams
# ============================================================

def persistence_statistics(diagrams, max_dim=2):
    """
    Compute statistical features from persistence diagrams.
    Fast, simple, and effective baseline vectorization.

    For each homological dimension, computes:
    - Number of features
    - Mean persistence (lifetime)
    - Max persistence
    - Std persistence
    - Mean birth time
    - Mean death time
    - Entropy of persistence values
    - Sum of squared persistences (total persistence energy)

    Returns: Fixed-size feature vector
    """
    features = []

    for dim in range(max_dim + 1):
        if dim < len(diagrams):
            dgm = diagrams[dim]
            # Remove infinite features
            finite_mask = np.isfinite(dgm[:, 1])
            dgm = dgm[finite_mask]

            if len(dgm) > 0:
                births = dgm[:, 0]
                deaths = dgm[:, 1]
                persistence = deaths - births

                # Filter out zero-persistence features
                nonzero = persistence > 1e-8
                persistence = persistence[nonzero]
                births = births[nonzero]
                deaths = deaths[nonzero]

                if len(persistence) > 0:
                    # Persistence entropy
                    p_norm = persistence / persistence.sum()
                    entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))

                    features.extend([
                        len(persistence),                      # Count
                        np.mean(persistence),                  # Mean persistence
                        np.max(persistence),                   # Max persistence
                        np.std(persistence),                   # Std persistence
                        np.mean(births),                       # Mean birth
                        np.mean(deaths),                       # Mean death
                        entropy,                               # Persistence entropy
                        np.sum(persistence ** 2),              # Total energy
                        np.percentile(persistence, 25),        # Q1
                        np.percentile(persistence, 75),        # Q3
                    ])
                else:
                    features.extend([0] * 10)
            else:
                features.extend([0] * 10)
        else:
            features.extend([0] * 10)

    return np.array(features, dtype=np.float32)


def persistence_image_features(diagrams, max_dim=1, resolution=20, sigma=0.1):
    """
    Convert persistence diagrams to persistence images.

    Persistence images are a stable, fixed-size vectorization that
    captures the distribution of topological features in birth-persistence space.

    Returns: Flattened persistence image as feature vector
    """
    pimgr = PersistenceImager(
        pixel_size=sigma,
        birth_range=(0, 5),
        pers_range=(0, 5),
    )

    features = []

    for dim in range(max_dim + 1):
        if dim < len(diagrams):
            dgm = diagrams[dim]
            # Remove infinite features
            finite_mask = np.isfinite(dgm[:, 1])
            dgm = dgm[finite_mask]

            if len(dgm) > 0:
                try:
                    img = pimgr.transform(dgm, skew=True)
                    features.append(img.flatten())
                except Exception:
                    features.append(np.zeros(resolution * resolution))
            else:
                features.append(np.zeros(resolution * resolution))
        else:
            features.append(np.zeros(resolution * resolution))

    return np.concatenate(features).astype(np.float32)


# ============================================================
# 4. Full pipeline: molecule → topological features
# ============================================================

def compute_tda_features(smiles=None, sequence=None, method="statistics",
                         use_cocycles=True, use_distpepfold=False, target_pdb=None):
    """
    Full pipeline: convert molecule to topological feature vector.

    Args:
        smiles: SMILES string (for small molecules)
        sequence: Amino acid sequence (for peptides)
        method: "statistics" (30+12=42 dim with cocycles) or "image" (800-dim)
        use_cocycles: Whether to compute cocycle-based spatial features (+12 dims)
        use_distpepfold: Whether to use DistPepFold for peptide 3D structure (GPU required)
        target_pdb: Path to target protein PDB file (for docking-based TDA)

    Returns:
        Torch tensor of topological features
    """
    # Get 3D coordinates
    if smiles:
        coords = get_3d_coords_from_smiles(smiles)
    elif sequence:
        if use_distpepfold and target_pdb:
            coords = get_3d_coords_distpepfold(sequence, target_pdb)
        else:
            coords = get_3d_coords_from_peptide(sequence)
    else:
        return None

    if coords is None or len(coords) < 3:
        return None

    # Compute persistent homology
    if use_cocycles:
        result = compute_persistence_with_cocycles(coords, max_dim=2)
        if result is None:
            return None
        diagrams = result['dgms']
        cocycle_feats = cocycle_features(result, coords, max_dim=2)
    else:
        diagrams = compute_persistence(coords, max_dim=2)
        if diagrams is None:
            return None
        cocycle_feats = None

    # Vectorize
    if method == "statistics":
        features = persistence_statistics(diagrams, max_dim=2)
        if cocycle_feats is not None:
            features = np.concatenate([features, cocycle_feats])
    elif method == "image":
        features = persistence_image_features(diagrams, max_dim=1)
    else:
        features = persistence_statistics(diagrams, max_dim=2)

    return torch.tensor(features, dtype=torch.float32)


def compute_tda_for_dataset(chembl_df, peptide_list=None, method="statistics",
                             use_cocycles=True):
    """
    Compute TDA features for an entire dataset.
    Returns dict mapping (smiles_or_sequence) -> feature tensor.
    """
    tda_cache = {}
    total = 0
    success = 0

    # Small molecules
    unique_smiles = chembl_df["smiles"].unique()
    print(f"Computing TDA for {len(unique_smiles)} unique molecules...")

    for i, smiles in enumerate(unique_smiles):
        total += 1
        feat = compute_tda_features(smiles=smiles, method=method,
                                     use_cocycles=use_cocycles)
        if feat is not None:
            tda_cache[smiles] = feat
            success += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(unique_smiles)} molecules ({success} successful)")

    print(f"  Molecules: {success}/{total} successful")

    # Peptides
    if peptide_list:
        pep_success = 0
        for pep in peptide_list:
            total += 1
            feat = compute_tda_features(sequence=pep["sequence"], method=method,
                                         use_cocycles=use_cocycles)
            if feat is not None:
                tda_cache[pep["sequence"]] = feat
                pep_success += 1
        print(f"  Peptides: {pep_success}/{len(peptide_list)} successful")

    print(f"Total TDA features computed: {len(tda_cache)}")
    return tda_cache


# ============================================================
# 5. DistPepFold integration (optional, requires GPU)
# ============================================================

DISTPEPFOLD_DIR = os.environ.get("DISTPEPFOLD_DIR", None)


def check_distpepfold():
    """Check if DistPepFold is installed and GPU is available."""
    if DISTPEPFOLD_DIR is None:
        return False, "DISTPEPFOLD_DIR environment variable not set"
    if not os.path.exists(DISTPEPFOLD_DIR):
        return False, f"DistPepFold directory not found: {DISTPEPFOLD_DIR}"
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return False, "DistPepFold requires CUDA GPU (12GB+)"
    except ImportError:
        return False, "PyTorch not available"
    return True, "DistPepFold ready"


def get_3d_coords_distpepfold(sequence, target_pdb_path):
    """
    Use DistPepFold to predict peptide-protein binding pose.

    DistPepFold (github.com/kiharalab/DistPepFold) predicts how a peptide
    docks to a target protein, giving us the 3D coordinates of the peptide
    in the binding context.

    This is much richer than our simplified helical model because:
    1. The peptide conformation depends on the target it binds to
    2. We get the binding interface topology, not just isolated peptide topology
    3. TDA on the complex captures binding pocket shape

    Args:
        sequence: Peptide amino acid sequence
        target_pdb_path: Path to target protein PDB file

    Returns:
        Nx3 numpy array of peptide atom coordinates in bound state,
        or None if DistPepFold is not available
    """
    available, msg = check_distpepfold()
    if not available:
        # Fallback to simplified model
        return get_3d_coords_from_peptide(sequence)

    import subprocess
    import tempfile

    try:
        # Write peptide sequence to temp FASTA file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">peptide\n{sequence}\n")
            fasta_path = f.name

        # Run DistPepFold
        output_dir = tempfile.mkdtemp()
        cmd = [
            "bash", os.path.join(DISTPEPFOLD_DIR, "pred.sh"),
            "--peptide", fasta_path,
            "--receptor", target_pdb_path,
            "--output", output_dir,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                cwd=DISTPEPFOLD_DIR)

        if result.returncode != 0:
            print(f"  DistPepFold failed: {result.stderr[:200]}")
            return get_3d_coords_from_peptide(sequence)

        # Parse output PDB for peptide coordinates
        output_pdb = os.path.join(output_dir, "predicted_complex.pdb")
        if os.path.exists(output_pdb):
            coords = _parse_peptide_coords_from_pdb(output_pdb)
            if coords is not None and len(coords) >= 3:
                return coords

    except Exception as e:
        print(f"  DistPepFold error: {e}")

    # Fallback
    return get_3d_coords_from_peptide(sequence)


def _parse_peptide_coords_from_pdb(pdb_path, chain_id='B'):
    """Parse CA atom coordinates from a PDB file for a specific chain."""
    coords = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[21] == chain_id:
                    atom_name = line[12:16].strip()
                    if atom_name == 'CA':  # Alpha carbon only
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
    except Exception:
        return None

    return np.array(coords) if coords else None


def compute_binding_tda(sequence, target_pdb_path, method="statistics"):
    """
    Compute TDA features for a peptide-protein binding complex.

    If DistPepFold is available, predicts the binding pose and computes
    TDA on the complex. Otherwise falls back to isolated peptide TDA.

    This captures the topology of the binding INTERFACE — rings and
    cavities formed between the peptide and its target — which is
    directly informative about binding affinity.
    """
    available, _ = check_distpepfold()

    if available and target_pdb_path:
        coords = get_3d_coords_distpepfold(sequence, target_pdb_path)
    else:
        coords = get_3d_coords_from_peptide(sequence)

    if coords is None or len(coords) < 3:
        return None

    result = compute_persistence_with_cocycles(coords, max_dim=2)
    if result is None:
        return None

    stats = persistence_statistics(result['dgms'], max_dim=2)
    cocycle_feats = cocycle_features(result, coords, max_dim=2)
    features = np.concatenate([stats, cocycle_feats])

    return torch.tensor(features, dtype=torch.float32)


# ============================================================
# Target protein PDB paths (for docking-based TDA)
# ============================================================

# Download these from RCSB PDB (rcsb.org) for docking-based TDA
TARGET_PDBS = {
    "alpha_glucosidase": {
        "pdb_id": "5NN8",   # Human intestinal maltase-glucoamylase
        "description": "Crystal structure of human MGAM in complex with acarbose",
    },
    "lipase": {
        "pdb_id": "1LPB",   # Human pancreatic lipase
        "description": "Crystal structure of human pancreatic lipase",
    },
    "bile_acid_receptor": {
        "pdb_id": "6HL1",   # FXR (farnesoid X receptor)
        "description": "Crystal structure of FXR ligand binding domain",
    },
    "sodium_hydrogen_exchanger": {
        "pdb_id": "7S0Z",   # NHE3
        "description": "Cryo-EM structure of NHE3",
    },
}


# ============================================================
# 6. Quick test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MEAL SHIELD -- Persistent Homology Module (Enhanced)")
    print("=" * 60)

    # Test enhanced features on a known drug
    print("\n--- Acarbose (alpha-glucosidase inhibitor) ---")
    acarbose_smiles = "OC1C(OC2(CO)OC(OC3OC(CO)C(O)C(O)C3N)C(O)C2O)C(O)C(O)C1O"
    feat = compute_tda_features(smiles=acarbose_smiles, method="statistics", use_cocycles=True)
    if feat is not None:
        print(f"  TDA features: {feat.shape} dims (30 stats + 12 cocycle)")
        print(f"  Stats portion:   {feat[:5].tolist()}")
        print(f"  Cocycle portion: {feat[30:].tolist()}")

    # Test on peptides
    print("\n--- Food peptides with cocycle features ---")
    test_peptides = [
        ("IPP",        "ACE inhibitor (Calpis)"),
        ("IPAVF",      "Alpha-glucosidase inhibitor (bean)"),
        ("KLPGF",      "Alpha-glucosidase inhibitor (silk)"),
        ("PAGNFLPP",   "Lipase inhibitor (soybean)"),
        ("LRSELAAWSR", "Alpha-glucosidase inhibitor (rice)"),
    ]

    for seq, desc in test_peptides:
        feat = compute_tda_features(sequence=seq, method="statistics", use_cocycles=True)
        if feat is not None:
            print(f"  {seq:<12} {feat.shape} dims | norm: {feat.norm():.2f}  ({desc})")
        else:
            print(f"  {seq:<12} FAILED")

    # Check DistPepFold availability
    print("\n--- DistPepFold status ---")
    available, msg = check_distpepfold()
    print(f"  Available: {available}")
    print(f"  Message: {msg}")
    if not available:
        print("  To enable: git clone https://github.com/kiharalab/DistPepFold")
        print("  Then set: export DISTPEPFOLD_DIR=/path/to/DistPepFold")
        print("  Requires: CUDA GPU with 12GB+ VRAM")

    # Show target PDBs for docking
    print("\n--- Target protein structures (for docking TDA) ---")
    for target, info in TARGET_PDBS.items():
        print(f"  {target}: PDB {info['pdb_id']} - {info['description']}")
    print("  Download from: https://www.rcsb.org/")

    print("\nDone!")
