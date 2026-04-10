"use client";

import { type TargetPrediction, activityLevel, formatIC50 } from "@/lib/api";

const TARGET_ICONS: Record<string, string> = {
  alpha_glucosidase: "Starch Blocker",
  lipase: "Fat Blocker",
  bile_acid_receptor: "Bile Acid",
  sodium_hydrogen_exchanger: "Na+/H+ Exchange",
};

const LEVEL_COLORS = {
  high: { bar: "bg-emerald-500", text: "text-emerald-400", label: "Strong" },
  medium: { bar: "bg-amber-500", text: "text-amber-400", label: "Moderate" },
  low: { bar: "bg-zinc-600", text: "text-zinc-400", label: "Weak" },
};

interface PredictionResultsProps {
  predictions: TargetPrediction[];
  demoMode: boolean;
  onExplainTarget?: (target: string) => void;
  activeExplainTarget?: string;
}

export default function PredictionResults({
  predictions,
  demoMode,
  onExplainTarget,
  activeExplainTarget,
}: PredictionResultsProps) {
  const sorted = [...predictions].sort((a, b) => b.pIC50 - a.pIC50);
  const maxPIC50 = Math.max(...sorted.map((p) => p.pIC50), 9);

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Bioactivity Predictions</h2>
        {demoMode && (
          <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs text-amber-400">
            Demo mode — train model for real predictions
          </span>
        )}
      </div>

      <div className="space-y-3">
        {sorted.map((pred) => {
          const level = activityLevel(pred.pIC50);
          const colors = LEVEL_COLORS[level];
          const barWidth = Math.max((pred.pIC50 / maxPIC50) * 100, 5);

          const isExplaining = activeExplainTarget === pred.target;

          return (
            <div
              key={pred.target}
              className={`rounded-lg border bg-[var(--color-surface)] p-4 ${
                isExplaining
                  ? "border-[var(--color-accent)]"
                  : "border-[var(--color-border)]"
              }`}
            >
              <div className="mb-2 flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-[var(--color-text)]">
                      {TARGET_ICONS[pred.target] || pred.target}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${colors.text} border ${
                        level === "high"
                          ? "border-emerald-500/30 bg-emerald-500/10"
                          : level === "medium"
                            ? "border-amber-500/30 bg-amber-500/10"
                            : "border-zinc-500/30 bg-zinc-500/10"
                      }`}
                    >
                      {colors.label}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
                    {pred.label}
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-mono text-sm font-semibold">
                    pIC50 = {pred.pIC50.toFixed(2)}
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)]">
                    IC50 = {formatIC50(pred.IC50_uM)}
                  </p>
                </div>
              </div>

              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-bg)]">
                <div
                  className={`h-full rounded-full ${colors.bar} transition-all duration-500`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>

              {onExplainTarget && (
                <button
                  onClick={() => onExplainTarget(pred.target)}
                  className={`mt-2 text-xs transition ${
                    isExplaining
                      ? "text-[var(--color-accent-light)] font-medium"
                      : "text-[var(--color-text-muted)] hover:text-[var(--color-accent)]"
                  }`}
                >
                  {isExplaining ? "Showing attribution below" : "Explain this prediction"}
                </button>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-3">
        <p className="text-xs text-[var(--color-text-muted)]">
          <strong>Reading the results:</strong> pIC50 is -log10(IC50 in M). Higher
          = stronger inhibition. Strong (&ge;6.0) means sub-micromolar activity.
          Moderate (4.5-6.0) is relevant for food-derived peptides. Values from
          multi-task GNN trained on ChEMBL bioactivity data + 212 curated food
          peptides.
        </p>
      </div>
    </div>
  );
}
