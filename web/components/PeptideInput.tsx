"use client";

import { useState } from "react";

const EXAMPLES = [
  { sequence: "IPP", label: "IPP (milk, ACE inhibitor)" },
  { sequence: "VPP", label: "VPP (milk, ACE inhibitor)" },
  { sequence: "IPAVF", label: "IPAVF (soy-derived)" },
  { sequence: "KLPGF", label: "KLPGF (whey peptide)" },
  { sequence: "LKP", label: "LKP (bonito, ACE inhibitor)" },
  { sequence: "AYFYPEL", label: "AYFYPEL (alpha-lactalbumin)" },
];

interface PeptideInputProps {
  onSubmit: (sequence: string) => void;
  loading: boolean;
}

export default function PeptideInput({ onSubmit, loading }: PeptideInputProps) {
  const [sequence, setSequence] = useState("");
  const [error, setError] = useState("");

  const validate = (seq: string): string | null => {
    const clean = seq.toUpperCase().trim();
    if (!clean) return "Enter a peptide sequence";
    if (clean.length > 50) return "Max 50 residues";
    const invalid = [...clean].filter(
      (c) => !"ACDEFGHIKLMNPQRSTVWY".includes(c)
    );
    if (invalid.length > 0)
      return `Invalid amino acids: ${[...new Set(invalid)].join(", ")}`;
    return null;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const err = validate(sequence);
    if (err) {
      setError(err);
      return;
    }
    setError("");
    onSubmit(sequence.toUpperCase().trim());
  };

  const handleExample = (seq: string) => {
    setSequence(seq);
    setError("");
    onSubmit(seq);
  };

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <div className="flex-1">
          <input
            type="text"
            value={sequence}
            onChange={(e) => {
              setSequence(e.target.value.toUpperCase());
              setError("");
            }}
            placeholder="Enter peptide sequence (e.g. IPAVF)"
            className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] px-4 py-3 text-lg font-mono tracking-wider text-[var(--color-text)] placeholder:text-[var(--color-text-muted)] focus:border-[var(--color-accent)] focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
            disabled={loading}
            spellCheck={false}
            autoComplete="off"
          />
          {error && (
            <p className="mt-1.5 text-sm text-[var(--color-danger)]">
              {error}
            </p>
          )}
        </div>
        <button
          type="submit"
          disabled={loading || !sequence.trim()}
          className="rounded-lg bg-[var(--color-accent)] px-6 py-3 font-medium text-white transition hover:bg-[var(--color-accent-light)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg
                className="h-4 w-4 animate-spin"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="3"
                  className="opacity-25"
                />
                <path
                  d="M4 12a8 8 0 018-8"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
              </svg>
              Predicting
            </span>
          ) : (
            "Predict"
          )}
        </button>
      </form>

      <div className="mt-4">
        <p className="mb-2 text-xs text-[var(--color-text-muted)]">
          Try an example:
        </p>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.sequence}
              onClick={() => handleExample(ex.sequence)}
              disabled={loading}
              className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] px-3 py-1.5 text-xs font-mono text-[var(--color-text-muted)] transition hover:border-[var(--color-accent)] hover:text-[var(--color-text)] disabled:opacity-50"
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
