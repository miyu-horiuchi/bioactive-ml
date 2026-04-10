"use client";

import { type ExplainResponse } from "@/lib/api";

interface AttributionViewerProps {
  explanation: ExplainResponse;
}

function scoreToColor(score: number): string {
  // Red (high importance) -> Yellow (medium) -> Blue (low)
  if (score > 0.66) {
    const t = (score - 0.66) / 0.34;
    const r = 239;
    const g = Math.round(68 + (1 - t) * 90);
    const b = 68;
    return `rgb(${r},${g},${b})`;
  }
  if (score > 0.33) {
    const t = (score - 0.33) / 0.33;
    const r = Math.round(245 * t + 100 * (1 - t));
    const g = Math.round(158 * t + 130 * (1 - t));
    const b = Math.round(11 * t + 180 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }
  const t = score / 0.33;
  const r = Math.round(100 * t + 60 * (1 - t));
  const g = Math.round(130 * t + 90 * (1 - t));
  const b = Math.round(180 * t + 160 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

export default function AttributionViewer({
  explanation,
}: AttributionViewerProps) {
  const { residues, scores, top_residues, method } = explanation;

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Residue Attribution</h2>
        <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1 text-[10px] text-[var(--color-text-muted)]">
          {method}
        </span>
      </div>

      {/* Sequence heatmap */}
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <p className="mb-2 text-xs text-[var(--color-text-muted)]">
          Which amino acids drive the prediction (brighter = more important):
        </p>
        <div className="flex items-end gap-0.5">
          {residues.map((aa, i) => (
            <div key={i} className="flex flex-col items-center">
              {/* Bar */}
              <div
                className="w-8 rounded-t transition-all"
                style={{
                  height: `${Math.max(scores[i] * 60, 4)}px`,
                  backgroundColor: scoreToColor(scores[i]),
                  opacity: 0.4 + scores[i] * 0.6,
                }}
              />
              {/* Residue letter */}
              <div
                className="flex h-8 w-8 items-center justify-center rounded-b font-mono text-sm font-bold"
                style={{
                  backgroundColor: scoreToColor(scores[i]),
                  color: scores[i] > 0.5 ? "#fff" : "#ddd",
                }}
              >
                {aa}
              </div>
              {/* Position */}
              <span className="mt-0.5 text-[9px] text-[var(--color-text-muted)]">
                {i + 1}
              </span>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-3 flex items-center justify-between text-[10px] text-[var(--color-text-muted)]">
          <div className="flex items-center gap-1.5">
            <div
              className="h-2.5 w-10 rounded-sm"
              style={{
                background:
                  "linear-gradient(to right, rgb(60,90,160), rgb(100,130,180), rgb(245,158,11), rgb(239,68,68))",
              }}
            />
            <span>Low</span>
            <span className="ml-4">High</span>
          </div>
          <span>importance score</span>
        </div>
      </div>

      {/* Top residues table */}
      <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
        <p className="mb-2 text-xs font-medium text-[var(--color-text-muted)]">
          Top contributing residues:
        </p>
        <div className="space-y-1">
          {top_residues.slice(0, 5).map((r, i) => (
            <div
              key={i}
              className="flex items-center justify-between rounded px-2 py-1 text-sm"
              style={{
                backgroundColor: `${scoreToColor(r.score)}15`,
              }}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold" style={{ color: scoreToColor(r.score) }}>
                  {r.residue}
                </span>
                <span className="text-xs text-[var(--color-text-muted)]">
                  position {r.position + 1}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-16 overflow-hidden rounded-full bg-[var(--color-bg)]">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${r.score * 100}%`,
                      backgroundColor: scoreToColor(r.score),
                    }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-xs text-[var(--color-text-muted)]">
                  {(r.score * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {explanation.demo_mode && (
        <p className="mt-2 text-[10px] text-amber-400/70">
          Demo mode — attributions are heuristic. Train the model for real attention-based explanations.
        </p>
      )}
    </div>
  );
}

export { scoreToColor };
