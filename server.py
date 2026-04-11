"""
Meal Shield API — FastAPI backend for peptide bioactivity prediction.

Wraps the GNN prediction pipeline and TDA computation as REST endpoints.
Includes a demo mode that returns simulated predictions when no trained
model checkpoint is available.
"""

import os
import logging
from typing import Optional, List
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logger = logging.getLogger("meal-shield")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_NAMES = [
    "alpha_glucosidase",
    "lipase",
    "bile_acid_receptor",
    "sodium_hydrogen_exchanger",
    "ace_inhibitor",
    "dpp4_inhibitor",
]

TARGET_LABELS = {
    "alpha_glucosidase": "Alpha-glucosidase (starch to glucose blocker)",
    "lipase": "Pancreatic lipase (fat digestion blocker)",
    "bile_acid_receptor": "FXR bile acid receptor",
    "sodium_hydrogen_exchanger": "NHE3 sodium-hydrogen exchanger",
    "ace_inhibitor": "ACE inhibitor (blood pressure)",
    "dpp4_inhibitor": "DPP-4 inhibitor (blood sugar)",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT", "checkpoints/meal_shield_gnn.pt"
)
TDA_CHECKPOINT_PATH = os.environ.get(
    "TDA_MODEL_CHECKPOINT", "checkpoints/meal_shield_gnn_tda.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

state = {
    "model": None,
    "tda_model": None,
    "demo_mode": True,
}


def _load_models():
    """Load model checkpoints, reading config from checkpoint metadata."""
    from model import MealShieldGNN, MealShieldGNN_TDA

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

        # Support both wrapped (metadata dict) and bare (raw state_dict) formats.
        # For bare state dicts, infer feature_dim from the input projection layer
        # so a mismatched checkpoint never silently loads against the wrong model.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            feat_dim = checkpoint.get("feature_dim")
            hidden = checkpoint.get("hidden_dim", 128)
            targets = checkpoint.get("target_names", TARGET_NAMES)
            esm_enabled = checkpoint.get("esm_enabled")
        else:
            model_state = checkpoint
            feat_dim = None
            hidden = 128
            targets = TARGET_NAMES
            esm_enabled = None

        if feat_dim is None:
            ip = model_state.get("input_proj.weight")
            feat_dim = int(ip.shape[1]) if ip is not None else 11
        if esm_enabled is None:
            esm_enabled = feat_dim > 11

        model = MealShieldGNN(
            node_feature_dim=feat_dim,
            hidden_dim=hidden,
            num_heads=4,
            num_layers=3,
            target_names=targets,
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        state["model"] = model
        state["feature_dim"] = feat_dim
        state["target_names"] = targets
        state["esm_enabled"] = esm_enabled
        state["demo_mode"] = False
        logger.info("Loaded GNN (feature_dim=%d, esm=%s, targets=%s)",
                    feat_dim, esm_enabled, targets)

    if os.path.exists(TDA_CHECKPOINT_PATH):
        try:
            tda_ckpt = torch.load(TDA_CHECKPOINT_PATH, map_location=device, weights_only=False)
            if isinstance(tda_ckpt, dict) and "model_state_dict" in tda_ckpt:
                tda_state = tda_ckpt["model_state_dict"]
                tda_feat_dim = tda_ckpt.get("feature_dim")
                tda_targets = tda_ckpt.get("target_names", TARGET_NAMES)
                tda_hidden = tda_ckpt.get("hidden_dim", 128)
            else:
                tda_state = tda_ckpt
                tda_feat_dim = None
                tda_targets = TARGET_NAMES
                tda_hidden = 128

            if tda_feat_dim is None:
                ip = tda_state.get("input_proj.weight")
                tda_feat_dim = int(ip.shape[1]) if ip is not None else 11

            tda_model = MealShieldGNN_TDA(
                node_feature_dim=tda_feat_dim,
                tda_feature_dim=42,
                hidden_dim=tda_hidden,
                num_heads=4,
                num_layers=3,
                target_names=tda_targets,
            )
            tda_model.load_state_dict(tda_state)
            tda_model.to(device)
            tda_model.eval()
            state["tda_model"] = tda_model
            state["tda_feature_dim"] = tda_feat_dim
            logger.info("Loaded GNN+TDA model (feature_dim=%d)", tda_feat_dim)
        except Exception as e:
            logger.warning("Skipping TDA model load: %s", e)
            state["tda_model"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Meal Shield API",
    description="Predict food peptide bioactivity against digestive enzyme targets using GNN + TDA",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"(https://.*\.vercel\.app|http://(localhost|127\.0\.0\.1):(3000|3001|3002|3003|3004|3005|3006))",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PropertiesRequest(BaseModel):
    sequence: str = Field(
        ..., description="Peptide amino acid sequence", examples=["LIWKL"]
    )


class PropertiesResponse(BaseModel):
    sequence: str
    properties: dict


class GenerateRequest(BaseModel):
    target: str = Field(
        ..., description="Target name", examples=["ace_inhibitor"]
    )
    method: str = Field(
        default="genetic",
        description="Generation method: 'genetic', 'mc', or 'enumerate'",
    )
    length: int = Field(default=5, description="Peptide length")
    n_candidates: int = Field(default=50, description="Number of candidates to generate")
    top_k: int = Field(default=10, description="Number of top candidates to return")


class GenerateCandidate(BaseModel):
    sequence: str
    pIC50: float
    IC50_uM: float
    properties: dict


class GenerateResponse(BaseModel):
    target: str
    method: str
    candidates: list


class PredictRequest(BaseModel):
    sequence: str = Field(
        ..., description="Peptide amino acid sequence", examples=["IPAVF"]
    )
    n_samples: int = Field(
        default=1,
        ge=1,
        le=200,
        description="If >1, use MC Dropout to return per-target uncertainty "
                    "(mean, std, 95% CI) by running this many stochastic "
                    "forward passes.",
    )


class TargetPrediction(BaseModel):
    target: str
    label: str
    pIC50: float
    IC50_uM: float
    pIC50_std: Optional[float] = None
    pIC50_ci95: Optional[List[float]] = None
    n_samples: Optional[int] = None


class PredictResponse(BaseModel):
    sequence: str
    predictions: list[TargetPrediction]
    demo_mode: bool


class TDARequest(BaseModel):
    sequence: str = Field(
        ..., description="Peptide amino acid sequence", examples=["IPAVF"]
    )


class TDAFeature(BaseModel):
    name: str
    value: float


class TDAResponse(BaseModel):
    sequence: str
    features: list[TDAFeature]
    dimension_summary: dict


class StructureResponse(BaseModel):
    sequence: str
    pdb: str
    num_residues: int


class ExplainRequest(BaseModel):
    sequence: str = Field(
        ..., description="Peptide amino acid sequence", examples=["IPAVF"]
    )
    target: str = Field(
        ..., description="Target to explain", examples=["alpha_glucosidase"]
    )
    method: str = Field(
        default="attention",
        description="Attribution method: 'attention' or 'integrated_gradients'",
    )


class ResidueAttribution(BaseModel):
    residue: str
    position: int
    score: float


class ExplainResponse(BaseModel):
    sequence: str
    target: str
    method: str
    residues: list[str]
    scores: list[float]
    top_residues: list[ResidueAttribution]
    demo_mode: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_sequence(sequence: str) -> str:
    seq = sequence.upper().strip()
    if not seq:
        raise HTTPException(400, "Empty sequence")
    if len(seq) > 50:
        raise HTTPException(400, "Sequence too long (max 50 residues)")
    invalid = set(seq) - VALID_AA
    if invalid:
        raise HTTPException(
            400, f"Invalid amino acids: {', '.join(sorted(invalid))}"
        )
    return seq


def _demo_predict(sequence: str) -> list[TargetPrediction]:
    """Generate plausible demo predictions based on sequence properties."""
    rng = np.random.RandomState(hash(sequence) % (2**31))

    hydrophobic = sum(1 for aa in sequence if aa in "AVILMFWP")
    charged = sum(1 for aa in sequence if aa in "DEKRH")
    length_factor = min(len(sequence) / 5.0, 1.5)

    base = 4.0 + length_factor * 0.5
    predictions = []
    for target in TARGET_NAMES:
        if target == "lipase":
            pic50 = base + hydrophobic * 0.15 + rng.normal(0, 0.2)
        elif target == "alpha_glucosidase":
            pic50 = base + charged * 0.12 + rng.normal(0, 0.2)
        else:
            pic50 = base + rng.normal(0, 0.3)

        pic50 = max(2.0, min(9.0, pic50))
        ic50_uM = (10 ** (9 - pic50)) / 1000 if pic50 > 0 else float("inf")

        predictions.append(
            TargetPrediction(
                target=target,
                label=TARGET_LABELS[target],
                pIC50=round(pic50, 3),
                IC50_uM=round(ic50_uM, 2),
            )
        )
    return predictions


def _real_predict(sequence: str, n_samples: int = 1) -> list[TargetPrediction]:
    """Run actual model prediction."""
    from train import predict_peptide

    results = predict_peptide(
        state["model"], sequence,
        state.get("target_names", TARGET_NAMES),
        device,
        feature_dim=state.get("feature_dim", 11),
        use_esm=state.get("esm_enabled", False),
        n_samples=n_samples,
    )
    if results is None:
        raise HTTPException(422, "Could not build molecular graph for this sequence")

    predictions = []
    for target, vals in results.items():
        predictions.append(
            TargetPrediction(
                target=target,
                label=TARGET_LABELS.get(target, target),
                pIC50=vals["pIC50"],
                IC50_uM=vals["IC50_uM"],
                pIC50_std=vals.get("pIC50_std"),
                pIC50_ci95=vals.get("pIC50_ci95"),
                n_samples=vals.get("n_samples"),
            )
        )
    return predictions


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "tda_model_loaded": state["tda_model"] is not None,
        "demo_mode": state["demo_mode"],
        "device": str(device),
    }


@app.get("/api/targets")
async def list_targets():
    try:
        from data import TARGETS as DATA_TARGETS
        descriptions = {
            name: info.get("description", "")
            for name, info in DATA_TARGETS.items()
        }
    except ImportError:
        descriptions = {}

    return [
        {
            "name": name,
            "label": TARGET_LABELS[name],
            "description": descriptions.get(name, ""),
        }
        for name in TARGET_NAMES
    ]


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    sequence = _validate_sequence(req.sequence)

    if state["demo_mode"]:
        predictions = _demo_predict(sequence)
    else:
        predictions = _real_predict(sequence, n_samples=req.n_samples)

    return PredictResponse(
        sequence=sequence,
        predictions=predictions,
        demo_mode=state["demo_mode"],
    )


@app.post("/api/tda", response_model=TDAResponse)
async def compute_tda(req: TDARequest):
    sequence = _validate_sequence(req.sequence)

    try:
        from topology import compute_tda_features
    except ImportError:
        raise HTTPException(503, "TDA dependencies not installed (ripser, persim)")

    feats = compute_tda_features(sequence=sequence, method="statistics", use_cocycles=True)
    if feats is None:
        raise HTTPException(422, "Could not compute TDA features for this sequence")

    feats_np = feats.numpy()

    feature_names = []
    for dim in range(3):
        prefix = f"H{dim}"
        for stat in [
            "count", "mean_persistence", "max_persistence", "std_persistence",
            "mean_birth", "mean_death", "entropy", "energy", "q1", "q3",
        ]:
            feature_names.append(f"{prefix}_{stat}")
    if len(feats_np) > 30:
        for dim in range(3):
            prefix = f"H{dim}_cocycle"
            for stat in ["spatial_extent", "fraction_involved", "count_involved", "std_distance"]:
                feature_names.append(f"{prefix}_{stat}")

    features = [
        TDAFeature(name=name, value=round(float(val), 6))
        for name, val in zip(feature_names, feats_np)
    ]

    dim_summary = {}
    for dim in range(3):
        start = dim * 10
        dim_summary[f"H{dim}"] = {
            "count": int(feats_np[start]),
            "mean_persistence": round(float(feats_np[start + 1]), 4),
            "max_persistence": round(float(feats_np[start + 2]), 4),
            "interpretation": [
                "Connected components (molecule count)",
                "Loops and rings (aromatic systems, cyclic structures)",
                "Cavities and voids (binding pockets)",
            ][dim],
        }

    return TDAResponse(
        sequence=sequence, features=features, dimension_summary=dim_summary
    )


@app.post("/api/structure", response_model=StructureResponse)
async def get_structure(req: PredictRequest):
    """Generate a simple PDB-format 3D structure for 3Dmol.js visualization."""
    sequence = _validate_sequence(req.sequence)

    try:
        from topology import get_3d_coords_from_peptide
    except ImportError:
        raise HTTPException(503, "RDKit not installed")

    coords = get_3d_coords_from_peptide(sequence)
    if coords is None or len(coords) == 0:
        raise HTTPException(422, "Could not generate 3D coordinates")

    aa_codes = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }

    pdb_lines = []
    for i, (aa, coord) in enumerate(zip(sequence, coords)):
        resname = aa_codes.get(aa, "UNK")
        pdb_lines.append(
            f"ATOM  {i+1:5d}  CA  {resname} A{i+1:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
            f"  1.00  0.00           C"
        )
    for i in range(len(sequence) - 1):
        pdb_lines.append(f"CONECT{i+1:5d}{i+2:5d}")
    pdb_lines.append("END")

    pdb_str = "\n".join(pdb_lines)

    return StructureResponse(
        sequence=sequence, pdb=pdb_str, num_residues=len(sequence)
    )


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):
    """Explain which residues drive the prediction for a given target."""
    sequence = _validate_sequence(req.sequence)

    if req.target not in TARGET_NAMES:
        raise HTTPException(400, f"Unknown target: {req.target}. Available: {TARGET_NAMES}")

    if req.method not in ("attention", "integrated_gradients"):
        raise HTTPException(400, "method must be 'attention' or 'integrated_gradients'")

    if state["demo_mode"]:
        from interpret import demo_explain
        result = demo_explain(sequence, req.target)
    else:
        from interpret import explain_prediction
        result = explain_prediction(
            state["model"], sequence, req.target, device, method=req.method
        )

    if result is None:
        raise HTTPException(422, "Could not explain prediction for this sequence")

    return ExplainResponse(
        sequence=sequence,
        target=req.target,
        method=result["method"],
        residues=result["residues"],
        scores=result["scores"],
        top_residues=[
            ResidueAttribution(**r) for r in result["top_residues"]
        ],
        demo_mode=state["demo_mode"],
    )


@app.post("/api/properties", response_model=PropertiesResponse)
async def score_properties(req: PropertiesRequest):
    """Score a peptide on developability properties (toxicity, solubility, etc.)."""
    sequence = _validate_sequence(req.sequence)

    try:
        from properties import score_peptide
    except ImportError:
        raise HTTPException(503, "Properties module not installed")

    props = score_peptide(sequence)
    if props is None:
        raise HTTPException(422, "Could not score properties for this sequence")

    return PropertiesResponse(sequence=sequence, properties=props)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_peptides(req: GenerateRequest):
    """Generate candidate peptides for a given target."""
    if req.target not in TARGET_NAMES:
        raise HTTPException(400, f"Unknown target: {req.target}. Available: {TARGET_NAMES}")

    valid_methods = ("genetic", "mc", "enumerate")
    if req.method not in valid_methods:
        raise HTTPException(400, f"Invalid method: {req.method}. Available: {list(valid_methods)}")

    if req.length < 2 or req.length > 15:
        raise HTTPException(400, "length must be between 2 and 15")

    if req.n_candidates < 1 or req.n_candidates > 500:
        raise HTTPException(400, "n_candidates must be between 1 and 500")

    if req.top_k < 1 or req.top_k > req.n_candidates:
        raise HTTPException(400, f"top_k must be between 1 and {req.n_candidates}")

    try:
        from generate import generate_mc, generate_genetic, generate_enumerate
    except ImportError:
        raise HTTPException(503, "Generate module not installed")

    model = state.get("model")
    target_names = state.get("target_names")
    feature_dim = state.get("feature_dim")
    if model is None or target_names is None or feature_dim is None:
        raise HTTPException(503, "Model not loaded")

    generators = {
        "genetic": generate_genetic,
        "mc": generate_mc,
        "enumerate": generate_enumerate,
    }
    generate_fn = generators[req.method]

    # generate_* returns List[Tuple[str, float]] — (sequence, pIC50)
    try:
        ranked = generate_fn(
            model=model,
            target=req.target,
            target_names=target_names,
            device=device,
            feature_dim=feature_dim,
            length=req.length,
            n_candidates=req.n_candidates,
        )
    except Exception as e:
        logger.exception("generate_%s failed", req.method)
        raise HTTPException(500, f"Generation failed: {e}")

    if not ranked:
        raise HTTPException(422, "Could not generate candidates for this target")

    # Score developability properties for top_k candidates so the response
    # matches GenerateCandidate's schema.
    from properties import score_peptide
    top = ranked[: req.top_k]
    candidates = []
    for seq, pic50 in top:
        try:
            props = score_peptide(seq) or {}
        except Exception:
            props = {}
        candidates.append({
            "sequence": seq,
            "pIC50": float(pic50),
            "IC50_uM": float(10 ** (6 - pic50)) if pic50 else float("nan"),
            "properties": props,
        })

    return GenerateResponse(
        target=req.target,
        method=req.method,
        candidates=candidates,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
