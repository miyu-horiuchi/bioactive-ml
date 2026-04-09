"""
Meal Shield GNN — Persistent Homology (Topological Data Analysis)

Computes topological features from molecular 3D coordinates using persistent
homology. These features capture global shape information (rings, cavities,
tunnels) that GNNs miss because GNNs only learn from local neighborhoods.

Pipeline:
  Molecule → 3D coordinates (RDKit) → Distance matrix → Ripser (persistent homology)
  → Persistence diagram → Vectorized features (persistence images / statistics)
"""

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

def compute_tda_features(smiles=None, sequence=None, method="statistics"):
    """
    Full pipeline: convert molecule to topological feature vector.

    Args:
        smiles: SMILES string (for small molecules)
        sequence: Amino acid sequence (for peptides)
        method: "statistics" (30-dim) or "image" (800-dim)

    Returns:
        Torch tensor of topological features
    """
    # Get 3D coordinates
    if smiles:
        coords = get_3d_coords_from_smiles(smiles)
    elif sequence:
        coords = get_3d_coords_from_peptide(sequence)
    else:
        return None

    if coords is None or len(coords) < 3:
        return None

    # Compute persistent homology
    diagrams = compute_persistence(coords, max_dim=2)
    if diagrams is None:
        return None

    # Vectorize
    if method == "statistics":
        features = persistence_statistics(diagrams, max_dim=2)
    elif method == "image":
        features = persistence_image_features(diagrams, max_dim=1)
    else:
        features = persistence_statistics(diagrams, max_dim=2)

    return torch.tensor(features, dtype=torch.float32)


def compute_tda_for_dataset(chembl_df, peptide_list=None, method="statistics"):
    """
    Compute TDA features for an entire dataset.
    Returns dict mapping (smiles_or_sequence) → feature tensor.
    """
    tda_cache = {}
    total = 0
    success = 0

    # Small molecules
    unique_smiles = chembl_df["smiles"].unique()
    print(f"Computing TDA for {len(unique_smiles)} unique molecules...")

    for i, smiles in enumerate(unique_smiles):
        total += 1
        feat = compute_tda_features(smiles=smiles, method=method)
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
            feat = compute_tda_features(sequence=pep["sequence"], method=method)
            if feat is not None:
                tda_cache[pep["sequence"]] = feat
                pep_success += 1
        print(f"  Peptides: {pep_success}/{len(peptide_list)} successful")

    print(f"Total TDA features computed: {len(tda_cache)}")
    return tda_cache


# ============================================================
# 5. Quick test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MEAL SHIELD — Persistent Homology Module")
    print("=" * 60)

    # Test on a known drug
    print("\n--- Acarbose (alpha-glucosidase inhibitor) ---")
    acarbose_smiles = "OC1C(OC2(CO)OC(OC3OC(CO)C(O)C(O)C3N)C(O)C2O)C(O)C(O)C1O"
    coords = get_3d_coords_from_smiles(acarbose_smiles)
    if coords is not None:
        print(f"  3D coords: {coords.shape} atoms")
        dgms = compute_persistence(coords)
        if dgms:
            for dim, dgm in enumerate(dgms):
                finite = dgm[np.isfinite(dgm[:, 1])]
                print(f"  H{dim}: {len(finite)} features")

            stats = persistence_statistics(dgms)
            print(f"  Statistical features ({len(stats)} dims): {stats[:5]}...")

    # Test on a peptide
    print("\n--- IPAVF (alpha-glucosidase inhibitor from bean) ---")
    coords = get_3d_coords_from_peptide("IPAVF")
    if coords is not None:
        print(f"  3D coords: {coords.shape} residues")
        dgms = compute_persistence(coords)
        if dgms:
            for dim, dgm in enumerate(dgms):
                finite = dgm[np.isfinite(dgm[:, 1])]
                print(f"  H{dim}: {len(finite)} features")

            stats = persistence_statistics(dgms)
            print(f"  Statistical features ({len(stats)} dims): {stats[:5]}...")

    # Test feature computation
    print("\n--- Full pipeline test ---")
    test_molecules = [
        ("Orlistat", "CCCCCCCCCCC(CC1OC(=O)C1CCCCCC)OC(=O)C(CC(C)C)NC=O", None),
        ("IPP", None, "IPP"),
        ("KLPGF", None, "KLPGF"),
        ("LRSELAAWSR", None, "LRSELAAWSR"),
    ]

    for name, smiles, seq in test_molecules:
        feat = compute_tda_features(smiles=smiles, sequence=seq, method="statistics")
        if feat is not None:
            print(f"  {name:<15} TDA features: {feat.shape} | norm: {feat.norm():.2f}")
        else:
            print(f"  {name:<15} FAILED")

    print("\nDone!")
