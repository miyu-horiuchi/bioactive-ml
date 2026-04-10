"""
ESM-2 Protein Language Model Embeddings for Peptides

Generates per-residue embeddings using ESM-2 (8M parameter model),
which encode evolutionary, structural, and functional context far
richer than hand-crafted physicochemical features.

Embeddings are cached to disk to avoid re-computation.
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache (loaded once per process)
_esm_model = None
_esm_alphabet = None
_esm_batch_converter = None


def _load_esm():
    """Load ESM-2 model (cached globally)."""
    global _esm_model, _esm_alphabet, _esm_batch_converter

    if _esm_model is not None:
        return _esm_model, _esm_alphabet, _esm_batch_converter

    import esm
    logger.info("Loading ESM-2 (esm2_t6_8M_UR50D)...")
    _esm_model, _esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    _esm_batch_converter = _esm_alphabet.get_batch_converter()
    _esm_model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        _esm_model = _esm_model.cuda()

    logger.info(f"ESM-2 loaded (embed_dim={_esm_model.embed_dim})")
    return _esm_model, _esm_alphabet, _esm_batch_converter


def get_esm_embedding(sequence: str) -> Optional[torch.Tensor]:
    """
    Get per-residue ESM-2 embedding for a peptide sequence.

    Args:
        sequence: Amino acid sequence (e.g., "LIWKL")

    Returns:
        Tensor of shape (seq_len, 320) with per-residue embeddings,
        or None if the sequence can't be processed.
    """
    model, alphabet, batch_converter = _load_esm()

    data = [("peptide", sequence)]
    _, _, batch_tokens = batch_converter(data)

    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    # Extract last layer representations, strip BOS/EOS tokens
    embeddings = results["representations"][6]
    # Shape: (1, seq_len + 2, 320) -> strip special tokens -> (seq_len, 320)
    per_residue = embeddings[0, 1:len(sequence) + 1, :].cpu()

    return per_residue


def get_esm_embeddings_batch(sequences: List[str],
                             batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Get ESM-2 embeddings for a batch of sequences.

    Args:
        sequences: List of amino acid sequences
        batch_size: Number of sequences per forward pass

    Returns:
        Dict mapping sequence -> Tensor(seq_len, 320)
    """
    model, alphabet, batch_converter = _load_esm()

    results = {}
    unique_seqs = list(set(sequences))

    for i in range(0, len(unique_seqs), batch_size):
        batch_seqs = unique_seqs[i:i + batch_size]
        data = [(f"pep_{j}", seq) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(data)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[6], return_contacts=False)

        embeddings = out["representations"][6]

        for j, seq in enumerate(batch_seqs):
            per_residue = embeddings[j, 1:len(seq) + 1, :].cpu()
            results[seq] = per_residue

        if (i + batch_size) % 100 < batch_size:
            logger.info(f"  ESM-2: {min(i + batch_size, len(unique_seqs))}/{len(unique_seqs)} sequences")

    return results


def compute_and_cache_embeddings(sequences: List[str],
                                 cache_path: str = "data/esm_cache.pt",
                                 batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Compute ESM-2 embeddings with disk caching.

    Loads existing cache, computes missing embeddings, saves updated cache.
    """
    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded ESM cache: {len(cache)} sequences")

    # Find sequences that need computing
    unique_seqs = list(set(sequences))
    missing = [s for s in unique_seqs if s not in cache]

    if missing:
        logger.info(f"Computing ESM-2 embeddings for {len(missing)} new sequences...")
        new_embeddings = get_esm_embeddings_batch(missing, batch_size=batch_size)
        cache.update(new_embeddings)

        # Save updated cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
        logger.info(f"Updated ESM cache: {len(cache)} total sequences")
    else:
        logger.info(f"All {len(unique_seqs)} sequences found in cache")

    return cache


def get_embedding_dim() -> int:
    """Return the ESM-2 embedding dimension (320 for esm2_t6_8M)."""
    return 320
