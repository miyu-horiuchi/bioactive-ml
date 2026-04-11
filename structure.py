"""
Meal Shield GNN — ESMFold 3D Structure Prediction for Peptides

Predicts 3D protein/peptide structures from sequence alone using ESMFold,
with PDB output, per-residue confidence (pLDDT) scoring, and contact maps.

ESMFold is a large model (~700MB). This module uses lazy loading with a
global model cache so it is only loaded once per process, and falls back
gracefully on systems without enough RAM or GPU memory.

Predicted structures are cached to disk under data/structures/ to avoid
redundant inference.

Usage:
    python structure.py --sequence LIWKL --output structures/LIWKL.pdb
    python structure.py --csv data/generated_peptides.csv --output-dir structures/
"""

import argparse
import hashlib
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================
# Global model cache (loaded once per process via lazy loading)
# ============================================================

_esmfold_model = None
_esmfold_available: Optional[bool] = None

STRUCTURE_CACHE_DIR = os.path.join("data", "structures")


def _get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_esmfold():
    """
    Lazy-load ESMFold model with global caching.

    The model is only loaded on the first call. Subsequent calls return
    the cached instance. Falls back gracefully if the model cannot be
    loaded (e.g. insufficient RAM).

    Returns:
        The ESMFold model, or None if loading failed.
    """
    global _esmfold_model, _esmfold_available

    if _esmfold_available is False:
        return None

    if _esmfold_model is not None:
        return _esmfold_model

    try:
        import esm

        logger.info("Loading ESMFold (esmfold_v1) — this may take a moment...")
        model = esm.pretrained.esmfold_v1()
        model.eval()

        device = _get_device()

        # For short peptides, half precision saves memory and is fast enough
        if device.type == "cuda":
            model = model.half()
            logger.info("ESMFold loaded on GPU (float16)")
        else:
            logger.info("ESMFold loaded on CPU (float32)")

        model = model.to(device)
        _esmfold_model = model
        _esmfold_available = True
        return _esmfold_model

    except ImportError:
        logger.error(
            "ESMFold not available — install fair-esm: pip install fair-esm"
        )
        _esmfold_available = False
        return None
    except RuntimeError as e:
        logger.error(
            "ESMFold failed to load (likely insufficient memory): %s", e
        )
        _esmfold_available = False
        return None
    except Exception as e:
        logger.error("Unexpected error loading ESMFold: %s", e)
        _esmfold_available = False
        return None


def _cache_key(sequence: str) -> str:
    """Deterministic filename-safe cache key for a sequence."""
    seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:12]
    return f"{sequence[:20]}_{seq_hash}"


def _cached_pdb_path(sequence: str) -> str:
    """Return the disk-cache path for a sequence's PDB file."""
    return os.path.join(STRUCTURE_CACHE_DIR, f"{_cache_key(sequence)}.pdb")


def _read_cached_pdb(sequence: str) -> Optional[str]:
    """Read a cached PDB string from disk, or None if not cached."""
    path = _cached_pdb_path(sequence)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def _write_cached_pdb(sequence: str, pdb_string: str) -> None:
    """Write a PDB string to disk cache."""
    os.makedirs(STRUCTURE_CACHE_DIR, exist_ok=True)
    path = _cached_pdb_path(sequence)
    with open(path, "w") as f:
        f.write(pdb_string)


# ============================================================
# 1. predict_structure
# ============================================================

def predict_structure(sequence: str) -> Optional[Dict]:
    """
    Predict the 3D structure of a peptide using ESMFold.

    Args:
        sequence: Amino acid sequence (e.g. "LIWKL").

    Returns:
        Dictionary with keys:
            - pdb_string: The full PDB-format coordinate string.
            - plddt_per_residue: np.ndarray of per-residue pLDDT confidence
              scores (0-100 scale, higher is better).
            - plddt_mean: Overall mean pLDDT confidence score.
            - sequence: The input sequence (echoed back for convenience).
        Returns None if ESMFold is unavailable or inference fails.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        logger.warning("Empty sequence provided")
        return None

    model = _load_esmfold()
    if model is None:
        return None

    # --- PDB string (use disk cache when available) ---
    pdb_string = _read_cached_pdb(sequence)
    if pdb_string is None:
        try:
            with torch.no_grad():
                pdb_string = model.infer_pdb(sequence)
            _write_cached_pdb(sequence, pdb_string)
        except Exception as e:
            logger.error("ESMFold infer_pdb failed for '%s': %s", sequence, e)
            return None

    # --- pLDDT scores (always run infer for these) ---
    try:
        with torch.no_grad():
            output = model.infer(sequence)

        # output.plddt shape: (1, seq_len, 1) or (1, seq_len)
        plddt_raw = output["plddt"]
        if isinstance(plddt_raw, torch.Tensor):
            plddt_np = plddt_raw.detach().cpu().float().numpy().squeeze()
        else:
            plddt_np = np.asarray(plddt_raw).squeeze()

        # Ensure 1-D array of per-residue scores
        if plddt_np.ndim == 0:
            plddt_np = np.array([float(plddt_np)])

        plddt_mean = float(np.mean(plddt_np))

    except Exception as e:
        logger.warning(
            "Could not extract pLDDT for '%s', falling back to PDB B-factors: %s",
            sequence, e,
        )
        plddt_np, plddt_mean = _plddt_from_pdb(pdb_string, len(sequence))

    return {
        "pdb_string": pdb_string,
        "plddt_per_residue": plddt_np,
        "plddt_mean": plddt_mean,
        "sequence": sequence,
    }


def _plddt_from_pdb(pdb_string: str, seq_len: int):
    """
    Extract per-residue pLDDT from PDB B-factor column as a fallback.

    ESMFold writes pLDDT into the B-factor field of the PDB output.
    We average over atoms belonging to each residue.
    """
    residue_bfactors: Dict[int, List[float]] = {}
    for line in pdb_string.splitlines():
        if line.startswith("ATOM"):
            try:
                res_seq = int(line[22:26].strip())
                bfactor = float(line[60:66].strip())
                residue_bfactors.setdefault(res_seq, []).append(bfactor)
            except (ValueError, IndexError):
                continue

    if not residue_bfactors:
        plddt_np = np.full(seq_len, np.nan, dtype=np.float32)
        return plddt_np, float("nan")

    # Build per-residue mean B-factor array (sorted by residue number)
    sorted_keys = sorted(residue_bfactors.keys())
    plddt_np = np.array(
        [np.mean(residue_bfactors[k]) for k in sorted_keys], dtype=np.float32
    )
    plddt_mean = float(np.mean(plddt_np))
    return plddt_np, plddt_mean


# ============================================================
# 2. save_pdb
# ============================================================

def save_pdb(sequence: str, output_path: str) -> Optional[str]:
    """
    Predict the 3D structure of a peptide and save it as a PDB file.

    Args:
        sequence: Amino acid sequence.
        output_path: Destination file path (e.g. "structures/LIWKL.pdb").

    Returns:
        The output_path on success, or None on failure.
    """
    result = predict_structure(sequence)
    if result is None:
        logger.error("Structure prediction failed for '%s'", sequence)
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(result["pdb_string"])

    logger.info(
        "Saved PDB for %s -> %s  (mean pLDDT=%.1f)",
        sequence, output_path, result["plddt_mean"],
    )
    return output_path


# ============================================================
# 3. batch_predict
# ============================================================

def batch_predict(
    sequences: List[str],
    output_dir: str,
) -> List[Dict]:
    """
    Predict structures for multiple peptides and save each as a PDB file.

    Args:
        sequences: List of amino acid sequences.
        output_dir: Directory in which to write PDB files. Each file is
            named ``<SEQUENCE>.pdb``.

    Returns:
        List of result dicts (one per sequence), each containing the keys
        from predict_structure() plus ``output_path``. Failed predictions
        are included with ``"error"`` set to a message string.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict] = []

    for i, seq in enumerate(sequences):
        seq = seq.strip().upper()
        if not seq:
            continue

        logger.info(
            "[%d/%d] Predicting structure for %s ...", i + 1, len(sequences), seq
        )

        result = predict_structure(seq)
        if result is None:
            results.append({"sequence": seq, "error": "prediction failed"})
            continue

        out_path = os.path.join(output_dir, f"{seq}.pdb")
        with open(out_path, "w") as f:
            f.write(result["pdb_string"])

        result["output_path"] = out_path
        results.append(result)

        logger.info(
            "  -> %s  (mean pLDDT=%.1f)", out_path, result["plddt_mean"]
        )

    succeeded = sum(1 for r in results if "error" not in r)
    logger.info(
        "Batch complete: %d/%d succeeded, output dir: %s",
        succeeded, len(results), output_dir,
    )
    return results


# ============================================================
# 4. get_contact_map
# ============================================================

def get_contact_map(sequence: str) -> Optional[np.ndarray]:
    """
    Return the predicted inter-residue contact map for a peptide.

    The contact map is a symmetric (L x L) matrix where L is the sequence
    length.  Each entry (i, j) is the predicted probability that residues
    i and j are in contact (C-beta distance < 8 A).

    Args:
        sequence: Amino acid sequence.

    Returns:
        np.ndarray of shape (L, L) with contact probabilities, or None
        if ESMFold is unavailable or inference fails.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        return None

    model = _load_esmfold()
    if model is None:
        return None

    try:
        with torch.no_grad():
            output = model.infer(sequence)

        # ESMFold output contains 'positions' — derive contacts from
        # predicted C-beta coordinates (or C-alpha for glycine).
        # The positions tensor shape is (1, L, atom_types, 3).
        positions = output["positions"]
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().float()

        # Index 1 corresponds to C-alpha in ESMFold atom ordering.
        # Use C-alpha for a robust contact map.
        ca_coords = positions[0, :, 1, :]  # (L, 3)

        # Pairwise distance matrix
        diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)  # (L, L, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (L, L)

        # Convert to contact probabilities using a soft threshold at 8 A
        contact_map = torch.sigmoid(5.0 * (8.0 - dist))
        return contact_map.numpy()

    except Exception as e:
        logger.error("Contact map extraction failed for '%s': %s", sequence, e)
        return None


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict 3D peptide structures with ESMFold",
    )
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        default=None,
        help="Single amino acid sequence to predict (e.g. LIWKL)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output PDB file path (used with --sequence)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Path to a CSV file containing a 'sequence' column. "
            "Predicts structures for every sequence in the file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="structures",
        help="Output directory for batch predictions (default: structures/)",
    )
    parser.add_argument(
        "--contact-map",
        action="store_true",
        help="Also compute and print the contact map (single-sequence mode)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Single sequence mode ---
    if args.sequence:
        seq = args.sequence.strip().upper()
        output_path = args.output or f"structures/{seq}.pdb"

        print(f"Predicting structure for: {seq}")
        result = predict_structure(seq)
        if result is None:
            print("ERROR: Structure prediction failed (see log above).")
            sys.exit(1)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(result["pdb_string"])

        print(f"  PDB saved to: {output_path}")
        print(f"  Mean pLDDT:   {result['plddt_mean']:.1f}")
        print(f"  Per-residue pLDDT: {result['plddt_per_residue']}")

        if args.contact_map:
            cmap = get_contact_map(seq)
            if cmap is not None:
                print(f"  Contact map shape: {cmap.shape}")
                print(f"  Contact map (top-left 5x5):")
                n = min(5, cmap.shape[0])
                for row in cmap[:n, :n]:
                    print("    " + "  ".join(f"{v:.2f}" for v in row))
            else:
                print("  Contact map: extraction failed")

        return

    # --- Batch CSV mode ---
    if args.csv:
        import pandas as pd

        if not os.path.exists(args.csv):
            print(f"ERROR: CSV file not found: {args.csv}")
            sys.exit(1)

        df = pd.read_csv(args.csv)
        if "sequence" not in df.columns:
            print(
                f"ERROR: CSV must contain a 'sequence' column. "
                f"Found: {list(df.columns)}"
            )
            sys.exit(1)

        sequences = df["sequence"].dropna().unique().tolist()
        print(f"Found {len(sequences)} unique sequences in {args.csv}")

        results = batch_predict(sequences, args.output_dir)

        succeeded = sum(1 for r in results if "error" not in r)
        print(f"\nDone: {succeeded}/{len(results)} structures predicted")
        print(f"PDB files saved to: {args.output_dir}/")

        if succeeded > 0:
            plddt_values = [
                r["plddt_mean"] for r in results if "error" not in r
            ]
            print(
                f"pLDDT — mean: {np.mean(plddt_values):.1f}, "
                f"min: {np.min(plddt_values):.1f}, "
                f"max: {np.max(plddt_values):.1f}"
            )
        return

    # --- No input provided ---
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
