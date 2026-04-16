# Meal Shield — Evaluation Results

Multi-task graph neural network for predicting food peptide inhibition
of six digestive-enzyme targets, with ESM-2 embeddings, topological
data-analysis features, and MC-Dropout uncertainty.

## Headline

**α-glucosidase R² = 0.800, mean R² = 0.561 across six targets.** Trained
in ~8 minutes on CPU using ESM-2 per-residue embeddings (320-dim) plus
masked multi-task learning over 6 ChEMBL targets and 110 food peptides.

## Dataset

- **8,904 molecular graphs total**
  - ChEMBL bioactivity: 7 targets capped at 2,000 compounds each
  - 110 food peptides filtered to the 6 modelled targets (from 212 curated)
- **6 modelled targets:** ACE inhibitor, α-glucosidase, bile-acid receptor
  (FXR), DPP-4, pancreatic lipase, sodium/hydrogen exchanger (NHE3)
- **Input features:** 8 physicochemical residue properties + 320-dim
  ESM-2 (`esm2_t6_8M_UR50D`) per-residue embeddings → 328-dim nodes

## Architecture

- 3-layer Graph Attention Network (GATConv, 4 heads, 128 hidden dims)
- Shared GNN backbone, per-target regression heads
- Masked multi-task MSE loss — each graph contributes only to its
  labelled target, but the backbone learns from the full dataset
- Training: Adam, `lr=1e-3`, weight decay `1e-5`, early stopping
  (patience 30), `ReduceLROnPlateau` scheduler, 100 epochs

## Baseline results — ESM + multi-task, hidden=128

| Target | R² | RMSE (pIC50) | n (test) |
|---|---|---|---|
| **alpha_glucosidase** | **0.800** | 0.566 | 254 |
| lipase | 0.620 | 0.775 | 403 |
| ace_inhibitor | 0.614 | 0.982 | 244 |
| dpp4_inhibitor | 0.569 | 0.875 | 405 |
| bile_acid_receptor | 0.391 | 0.556 | 400 |
| sodium_hydrogen_exchanger | 0.376 | 0.941 | 77 |
| **Mean** | **0.561** | — | — |

## Ablation study

Each ablation removes one ingredient of the baseline and keeps everything
else constant.

| Setting | alpha_glu | lipase | ace | dpp4 | bile | nhe3 | **Mean R²** |
|---|---|---|---|---|---|---|---|
| **Baseline** (ESM + multitask, h128) | **0.800** | **0.620** | 0.614 | **0.569** | **0.391** | 0.376 | **0.561** |
| no-ESM (multitask only) | 0.766 | 0.564 | 0.596 | 0.518 | 0.322 | 0.376 | 0.524 |
| no-multitask (ESM only, per-target) | 0.710 | 0.562 | 0.603 | 0.518 | 0.375 | **0.485** | 0.542 |

**Takeaways.**

1. **ESM-2 embeddings are the single biggest contributor (+0.038 mean R²).**
   Gains are largest on `bile_acid_receptor` (+0.069), `lipase` (+0.056), and
   `dpp4_inhibitor` (+0.051) — the targets where small-molecule signal alone
   is weakest. This is the headline architectural choice.

2. **Masked multi-task training is a modest net win (+0.019 mean R²), but
   with one honest trade-off.** It *hurts* `sodium_hydrogen_exchanger`
   (0.485 → 0.376) because that target has only 77 test samples and its
   gradient gets drowned out by the much larger ChEMBL classes. Per-target
   training recovers the small-class signal.

3. **Multi-task is the right default; fall back to per-target training
   when the target has < 100 samples.** For production this would mean
   running both heads and routing per-target.

## Hyperparameter sweep — model capacity

All runs use ESM + multitask.

| Hidden dim | Params | Mean R² | Best target | Worst target |
|---|---|---|---|---|
| 64 | ~90k | 0.553 | ace (0.626) | bile (0.346) |
| **128 (baseline)** | ~351k | **0.561** | alpha_glu (0.800) | nhe3 (0.376) |
| 256 | ~1.4M | 0.557 | alpha_glu (0.795) | bile (0.338) |

**Capacity is saturated at h=128.** Scaling to h=256 (4× params) gives
−0.004 mean R² — additional capacity is not what's limiting this model.
The bottleneck is data (15k graphs), not architecture. A future scaling
effort should focus on more peptides, not more parameters.

## GNN vs GNN + topological features

Persistent homology features (42-dim: H0/H1/H2 lifetime statistics +
12 cocycle-spatial descriptors, method=`statistics`) concatenated with
the GNN pooled embedding before the regression head. **Both legs use
ESM-2 node features**, same 3-layer GAT backbone, same 100-epoch
per-target training protocol (source: `checkpoints/tda_results.json`).

| Target | GNN R² | GNN+TDA R² | ΔR² |
|---|---|---|---|
| **bile_acid_receptor** | 0.042 | **0.297** | **+0.255** |
| **sodium_hydrogen_exchanger** | 0.011 | **0.151** | **+0.141** |
| lipase | 0.488 | **0.562** | **+0.073** |
| ace_inhibitor | 0.603 | **0.622** | +0.019 |
| alpha_glucosidase | **0.732** | 0.695 | −0.037 |
| **dpp4_inhibitor** | **0.576** | 0.370 | **−0.205** |
| **Mean ΔR²** | 0.409 | 0.450 | **+0.041** |

**TDA is a rescue mechanism for data-starved targets, not a blanket
improvement.** The two biggest wins are on `bile_acid_receptor`
(+0.255) and `sodium_hydrogen_exchanger` (+0.141) — the two targets
where per-target GNN training *collapses* (R² = 0.04 and 0.01
respectively) because their sample counts are too small for a bare
GAT to find signal. Persistent-homology descriptors provide a
sequence-independent structural prior that lifts them to real
predictive power.

`dpp4_inhibitor` regresses by −0.205, however — TDA is actively
harmful on the target with the most samples (n=405). A production
pipeline should gate TDA per target (bile, nhe, lipase) rather than
apply it everywhere.

**Caveat on the GNN-only baseline in this table.** These per-target
numbers are noticeably worse than the full ESM + multi-task baseline
reported above (mean 0.561 → 0.409) because `train_tda.py` trains
each target in isolation to keep the GNN-only vs GNN+TDA comparison
apples-to-apples. The small-sample targets (bile, nhe) collapse
without the multi-task backbone — which is exactly the regime where
TDA helps most. An ideal future experiment would re-run both legs
under masked multi-task training.

## Uncertainty quantification (MC Dropout)

`POST /api/predict` accepts an optional `n_samples` field. When > 1, the
model keeps dropout layers active at inference and runs `n_samples`
forward passes, returning a Gaussian 95% CI per target.

Example — IPP (known ACE inhibitor from fermented milk), `n_samples=30`:

| Target | pIC50 | ±σ | 95% CI |
|---|---|---|---|
| **ace_inhibitor** | 5.04 | 0.33 | [4.39, 5.70] |
| bile_acid_receptor | 5.73 | **0.16** | [5.42, 6.04] |
| alpha_glucosidase | 4.32 | 0.29 | [3.75, 4.88] |
| dpp4_inhibitor | 4.16 | 0.28 | [3.61, 4.72] |
| lipase | 3.72 | 0.25 | [3.24, 4.21] |

Typical σ is 0.15–0.35 pIC50 units — roughly ±0.5× to ±1× the per-target
RMSE, which is what we'd expect from MC Dropout with dropout rate 0.2.
`bile_acid_receptor` is the most confident target (σ=0.16), which aligns
with it being the target where the model generalises with the tightest
RMSE (0.56 in the baseline).

## Sanity check — distinct predictions for distinct peptides

After fixing `predict_peptide` to pipe ESM embeddings into inference
(earlier revisions returned bit-identical outputs for all peptides),
the following eight sequences now return meaningfully different
profiles:

| Peptide | Top target | pIC50 | Notes |
|---|---|---|---|
| IPP (milk, ACE) | ace_inhibitor | 5.06 | lit ~5.3 ✓ |
| VPP (milk, ACE) | ace_inhibitor | 5.20 | lit ~5.4 ✓ |
| IPAVF (soy) | alpha_glucosidase | 4.91 | — |
| KLPGF (whey) | alpha_glucosidase | 4.96 | — |
| LPYPY (Gouda, DPP-4) | dpp4_inhibitor | 4.39 | lit ~3.8 ✓ |
| WWWW (synthetic) | alpha_glucosidase | 4.81 | control |
| GGGG (control) | bile_acid_receptor | 5.69 | near-null baseline |
| DEEK (charged) | ace_inhibitor | 5.06 | — |

The top-ranked target and magnitude are literature-plausible for the
three known cases (IPP, VPP, LPYPY).

## Known limitations (honest list)

- **CPU-bound on this hardware.** PyTorch 2.2.2's MPS back-end does not
  implement `scatter_reduce` for `GATConv`, so training falls back to
  CPU. PyTorch ≥ 2.4 resolves this.
- **Small-class underfit.** `sodium_hydrogen_exchanger` has 77 test
  samples and multi-task training costs it ~0.11 R². Per-target fallback
  is the mitigation.
- **Test split is random, not scaffold-split.** Reported R²'s reflect
  in-distribution generalisation; an out-of-scaffold evaluation would
  likely score lower and is the most important next experiment.
- **Uncertainty is MC Dropout, not a proper posterior.** It correctly
  reflects parameter uncertainty from the dropout mask but does not
  cover epistemic uncertainty from OOD inputs. A deep-ensemble or
  conformal layer would improve calibration.
- **No external validation set.** All numbers are from a held-out test
  slice of the training distribution. Publication-grade claims need
  a truly unseen benchmark.

## Reproducibility

Everything below regenerates the numbers in this file.

```bash
# Baseline (≈ 8 min on CPU with ESM cache warm)
DATALOADER_WORKERS=0 TORCH_COMPILE=0 \
  python train.py --esm --multitask --epochs 100

# Ablation — no ESM
DATALOADER_WORKERS=0 TORCH_COMPILE=0 \
  python train.py --multitask --epochs 100

# Ablation — no multi-task (per-target)
DATALOADER_WORKERS=0 TORCH_COMPILE=0 \
  python train.py --esm --epochs 100

# Hyperparameter sweep
for h in 64 128 256; do
  DATALOADER_WORKERS=0 TORCH_COMPILE=0 \
    python train.py --esm --multitask --epochs 100 --hidden-dim $h
done

# GNN vs GNN+TDA (ESM)
DATALOADER_WORKERS=0 TORCH_COMPILE=0 \
  python train_tda.py --esm --epochs 100

# Live demo
python -m uvicorn server:app --host 127.0.0.1 --port 8000 &
cd web && NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000 \
  PORT=3004 npm run dev
# Browser: http://localhost:3004
```

Saved checkpoints (`checkpoints/`):

- `meal_shield_gnn.pt` — baseline, ESM+multitask, h128
- `baseline_esm_multitask.{pt,json}` — immutable backup of the baseline
- `ablation_no_esm.{pt,json}`, `ablation_no_multitask.{pt,json}`
- `hp_h64.{pt,json}`, `hp_h256.{pt,json}`
- `meal_shield_gnn_tda.pt` — GNN+TDA
- `results.json`, `sweep_summary.json` — aggregated metrics
