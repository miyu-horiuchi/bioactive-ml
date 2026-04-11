"""
Meal Shield API — FastAPI backend for peptide bioactivity prediction.

Wraps the GNN prediction pipeline and TDA computation as REST endpoints.
Includes a demo mode that returns simulated predictions when no trained
model checkpoint is available.
"""

import os
import logging
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

        # Support both old (raw state_dict) and new (metadata dict) checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            feat_dim = checkpoint.get("feature_dim", 11)
            hidden = checkpoint.get("hidden_dim", 128)
            targets = checkpoint.get("target_names", TARGET_NAMES)
        else:
            model_state = checkpoint
            feat_dim = 11
            hidden = 128
            targets = TARGET_NAMES

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
        state["demo_mode"] = False
        logger.info("Loaded GNN (feature_dim=%d, targets=%s)", feat_dim, targets)

    if os.path.exists(TDA_CHECKPOINT_PATH):
        feat_dim = state.get("feature_dim", 11)
        targets = state.get("target_names", TARGET_NAMES)

        tda_model = MealShieldGNN_TDA(
            node_feature_dim=feat_dim,
            tda_feature_dim=42,
            hidden_dim=128,
            num_heads=4,
            num_layers=3,
            target_names=targets,
        )
        tda_ckpt = torch.load(TDA_CHECKPOINT_PATH, map_location=device, weights_only=False)
        if isinstance(tda_ckpt, dict) and "model_state_dict" in tda_ckpt:
            tda_model.load_state_dict(tda_ckpt["model_state_dict"])
        else:
            tda_model.load_state_dict(tda_ckpt)
        tda_model.to(device)
        tda_model.eval()
        state["tda_model"] = tda_model
        logger.info("Loaded GNN+TDA model")


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
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    sequence: str = Field(
        ..., description="Peptide amino acid sequence", examples=["IPAVF"]
    )


class TargetPrediction(BaseModel):
    target: str
    label: str
    pIC50: float
    IC50_uM: float


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


def _real_predict(sequence: str) -> list[TargetPrediction]:
    """Run actual model prediction."""
    from train import predict_peptide

    results = predict_peptide(
        state["model"], sequence,
        state.get("target_names", TARGET_NAMES),
        device,
        feature_dim=state.get("feature_dim", 11)
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
    return [
        {"name": name, "label": TARGET_LABELS[name]} for name in TARGET_NAMES
    ]


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    sequence = _validate_sequence(req.sequence)

    if state["demo_mode"]:
        predictions = _demo_predict(sequence)
    else:
        predictions = _real_predict(sequence)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
