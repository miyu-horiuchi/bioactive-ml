"use client";

import { useState } from "react";
import {
  generatePeptides,
  predictPeptide,
  type DesignCandidate,
  type TargetPrediction,
  formatIC50,
} from "@/lib/api";
import MoleculeViewer from "./MoleculeViewer";

const TARGETS = [
  {
    id: "ace_inhibitor",
    label: "Blood Pressure Regulation",
    desc: "ACE inhibitor — blocks angiotensin-converting enzyme",
    icon: "heart",
  },
  {
    id: "alpha_glucosidase",
    label: "Sugar Absorption Blocker",
    desc: "Alpha-glucosidase inhibitor — slows starch-to-glucose conversion",
    icon: "sugar",
  },
  {
    id: "dpp4_inhibitor",
    label: "Blood Sugar Control",
    desc: "DPP-4 inhibitor — helps regulate insulin signaling",
    icon: "glucose",
  },
  {
    id: "lipase",
    label: "Fat Absorption Blocker",
    desc: "Pancreatic lipase inhibitor — reduces dietary fat digestion",
    icon: "fat",
  },
  {
    id: "bile_acid_receptor",
    label: "Cholesterol Modulation",
    desc: "FXR bile acid receptor — modulates bile acid signaling",
    icon: "bile",
  },
  {
    id: "sodium_hydrogen_exchanger",
    label: "Gut Ion Transport",
    desc: "NHE3 — regulates sodium absorption in the gut",
    icon: "ion",
  },
];

const METHODS = [
  { id: "genetic", label: "Genetic Algorithm", desc: "Evolutionary search" },
  { id: "mc", label: "Monte Carlo", desc: "Simulated annealing" },
  { id: "enumerate", label: "Enumerate All", desc: "Exhaustive (short peptides only)" },
];

interface DesignPanelProps {}

export default function DesignPanel({}: DesignPanelProps) {
  const [selectedTarget, setSelectedTarget] = useState("");
  const [method, setMethod] = useState("genetic");
  const [length, setLength] = useState(5);
  const [loading, setLoading] = useState(false);
  const [candidates, setCandidates] = useState<DesignCandidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<DesignCandidate | null>(null);
  const [candidatePredictions, setCandidatePredictions] = useState<TargetPrediction[] | null>(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const handleDesign = async () => {
    if (!selectedTarget) {
      setError("Select a target function first");
      return;
    }
    setLoading(true);
    setError("");
    setCandidates([]);
    setSelectedCandidate(null);
    setCandidatePredictions(null);
    setStatus("Generating candidate peptides...");

    try {
      const result = await generatePeptides(
        selectedTarget,
        method,
        length,
        50,
        10,
      );
      setCandidates(result.candidates);
      setStatus(`Found ${result.candidates.length} candidates`);

      if (result.candidates.length > 0) {
        handleSelectCandidate(result.candidates[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Design failed");
      setStatus("");
    } finally {
      setLoading(false);
    }
  };

  const handleSelectCandidate = async (candidate: DesignCandidate) => {
    setSelectedCandidate(candidate);
    try {
      const pred = await predictPeptide(candidate.sequence);
      setCandidatePredictions(pred.predictions);
    } catch {
      setCandidatePredictions(null);
    }
  };

  const propertyBar = (label: string, value: number | undefined, inverted = false) => {
    const v = value ?? 0;
    const displayV = inverted ? 1 - v : v;
    const color = displayV > 0.7 ? "bg-emerald-500" : displayV > 0.4 ? "bg-amber-500" : "bg-red-500";
    return (
      <div className="flex items-center gap-2 text-xs">
        <span className="w-20 text-[var(--color-text-muted)]">{label}</span>
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-bg)]">
          <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${displayV * 100}%` }} />
        </div>
        <span className="w-8 text-right font-mono">{(v * 100).toFixed(0)}%</span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Step 1: Select function */}
      <div>
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
          Step 1: What should the peptide do?
        </h3>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {TARGETS.map((t) => (
            <button
              key={t.id}
              onClick={() => setSelectedTarget(t.id)}
              className={`rounded-lg border p-3 text-left transition ${
                selectedTarget === t.id
                  ? "border-[var(--color-accent)] bg-[var(--color-accent)]/10"
                  : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/50"
              }`}
            >
              <p className="text-sm font-medium">{t.label}</p>
              <p className="mt-0.5 text-[11px] text-[var(--color-text-muted)]">{t.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Step 2: Design parameters */}
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
          >
            {METHODS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[var(--color-text-muted)]">Length</label>
          <select
            value={length}
            onChange={(e) => setLength(Number(e.target.value))}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-2 text-sm"
          >
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n} value={n}>{n} residues</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleDesign}
          disabled={loading || !selectedTarget}
          className="rounded-lg bg-[var(--color-accent)] px-6 py-2 text-sm font-medium text-white transition hover:bg-[var(--color-accent-light)] disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
              Designing...
            </span>
          ) : (
            "Design Peptides"
          )}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 p-3">
          <p className="text-sm text-[var(--color-danger)]">{error}</p>
        </div>
      )}

      {status && !loading && candidates.length > 0 && (
        <p className="text-sm text-[var(--color-text-muted)]">{status}</p>
      )}

      {/* Results */}
      {candidates.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Candidate list */}
          <div>
            <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
              Designed Candidates
            </h3>
            <div className="space-y-2">
              {candidates.map((c, i) => (
                <button
                  key={c.sequence}
                  onClick={() => handleSelectCandidate(c)}
                  className={`w-full rounded-lg border p-3 text-left transition ${
                    selectedCandidate?.sequence === c.sequence
                      ? "border-[var(--color-accent)] bg-[var(--color-accent)]/5"
                      : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/50"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-[var(--color-accent)]/20 text-xs font-bold text-[var(--color-accent)]">
                        {i + 1}
                      </span>
                      <span className="font-mono text-lg tracking-widest">{c.sequence}</span>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-sm font-semibold">
                        {formatIC50(c.IC50_uM)}
                      </p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">
                        pIC50 = {c.pIC50.toFixed(2)}
                      </p>
                    </div>
                  </div>
                  {/* Property mini-bars */}
                  <div className="mt-2 space-y-1">
                    {propertyBar("Solubility", c.solubility)}
                    {propertyBar("Toxicity", c.toxicity, true)}
                    {propertyBar("Bitterness", c.bitterness, true)}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Selected candidate detail */}
          {selectedCandidate && (
            <div className="space-y-4">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
                3D Structure: {selectedCandidate.sequence}
              </h3>
              <MoleculeViewer sequence={selectedCandidate.sequence} />

              {candidatePredictions && (
                <div>
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
                    Activity Profile
                  </h4>
                  <div className="space-y-1">
                    {[...candidatePredictions]
                      .sort((a, b) => b.pIC50 - a.pIC50)
                      .map((pred) => (
                        <div key={pred.target} className="flex items-center gap-2 text-xs">
                          <span className="w-28 truncate text-[var(--color-text-muted)]">
                            {pred.label.split("(")[0].trim()}
                          </span>
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-bg)]">
                            <div
                              className="h-full rounded-full bg-[var(--color-accent)] transition-all"
                              style={{ width: `${Math.max((pred.pIC50 / 8) * 100, 5)}%` }}
                            />
                          </div>
                          <span className="w-16 text-right font-mono">
                            {formatIC50(pred.IC50_uM)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Properties detail */}
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-text-muted)]">
                  Developability Properties
                </h4>
                <div className="space-y-1">
                  {propertyBar("Solubility", selectedCandidate.solubility)}
                  {propertyBar("Stability", selectedCandidate.stability)}
                  {propertyBar("Permeability", selectedCandidate.hemolysis)}
                  {propertyBar("Toxicity", selectedCandidate.toxicity, true)}
                  {propertyBar("Hemolysis", selectedCandidate.hemolysis, true)}
                  {propertyBar("Bitterness", selectedCandidate.bitterness, true)}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
