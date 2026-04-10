"use client";

import { useState } from "react";
import PeptideInput from "@/components/PeptideInput";
import PredictionResults from "@/components/PredictionResults";
import MoleculeViewer from "@/components/MoleculeViewer";
import AttributionViewer from "@/components/AttributionViewer";
import {
  predictPeptide,
  explainPrediction,
  type PredictResponse,
  type ExplainResponse,
} from "@/lib/api";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState("");
  const [activeSequence, setActiveSequence] = useState("");

  const handlePredict = async (sequence: string) => {
    setLoading(true);
    setError("");
    setActiveSequence(sequence);
    setExplanation(null);

    try {
      const data = await predictPeptide(sequence);
      setResult(data);

      // Auto-explain for the top-predicted target
      if (data.predictions.length > 0) {
        const topTarget = [...data.predictions].sort(
          (a, b) => b.pIC50 - a.pIC50
        )[0];
        const explainData = await explainPrediction(
          sequence,
          topTarget.target
        );
        setExplanation(explainData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleExplainTarget = async (target: string) => {
    if (!activeSequence) return;
    try {
      const data = await explainPrediction(activeSequence, target);
      setExplanation(data);
    } catch {
      // Silently fail — explanation is supplementary
    }
  };

  return (
    <div className="space-y-8">
      {/* Hero */}
      <section>
        <h2 className="text-2xl font-bold">Peptide Bioactivity Prediction</h2>
        <p className="mt-2 max-w-2xl text-sm text-[var(--color-text-muted)]">
          Enter a food peptide sequence to predict its inhibitory activity
          against digestive enzyme targets. See which amino acids drive each
          prediction with attention-based explainability.
        </p>
      </section>

      {/* Input */}
      <section>
        <PeptideInput onSubmit={handlePredict} loading={loading} />
      </section>

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 p-4">
          <p className="text-sm text-[var(--color-danger)]">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Predictions + 3D viewer */}
          <div className="grid gap-8 lg:grid-cols-2">
            <section>
              <PredictionResults
                predictions={result.predictions}
                demoMode={result.demo_mode}
                onExplainTarget={handleExplainTarget}
                activeExplainTarget={explanation?.target}
              />
            </section>
            <section>
              <MoleculeViewer
                sequence={activeSequence}
                attributionScores={explanation?.scores}
              />
            </section>
          </div>

          {/* Attribution viewer */}
          {explanation && (
            <section>
              <AttributionViewer explanation={explanation} />
            </section>
          )}
        </>
      )}

      {/* Architecture info (shown before first prediction) */}
      {!result && !error && (
        <section className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
            <h3 className="font-semibold">Graph Neural Network</h3>
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">
              3-layer Graph Attention Network converts peptide residues into a
              molecular graph. Each amino acid becomes a node with 8 physicochemical
              features. Attention heads learn which residue interactions matter most.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
            <h3 className="font-semibold">Topological Data Analysis</h3>
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">
              Persistent homology captures global molecular shape — rings, cavities,
              and binding pockets — that local GNN neighborhoods miss. 42-dimensional
              feature vector from Ripser + cocycle analysis.
            </p>
          </div>
          <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
            <h3 className="font-semibold">Explainable Predictions</h3>
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">
              See exactly which amino acids drive each prediction. GAT attention
              weights and integrated gradients reveal per-residue importance,
              visualized as a heatmap on the 3D molecular structure.
            </p>
          </div>
        </section>
      )}
    </div>
  );
}
