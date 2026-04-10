"use client";

import { useEffect, useRef, useState } from "react";
import { getStructure } from "@/lib/api";

interface MoleculeViewerProps {
  sequence: string;
  attributionScores?: number[];
}

// Amino acid property colors (hydrophobic = warm, polar = cool)
const AA_COLORS: Record<string, string> = {
  A: "#e67e22", V: "#d35400", I: "#c0392b", L: "#e74c3c",
  M: "#f39c12", F: "#e74c3c", W: "#c0392b", P: "#e67e22",
  G: "#95a5a6", S: "#1abc9c", T: "#16a085", C: "#f1c40f",
  Y: "#e91e63", N: "#3498db", Q: "#2980b9",
  D: "#2471a3", E: "#1a5276", K: "#6c3483", R: "#7d3c98", H: "#8e44ad",
};

function attributionColor(score: number): string {
  // Blue (low) -> Yellow (mid) -> Red (high)
  if (score > 0.5) {
    const t = (score - 0.5) * 2;
    const r = 239;
    const g = Math.round(180 * (1 - t) + 68 * t);
    const b = Math.round(20 * (1 - t));
    return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
  }
  const t = score * 2;
  const r = Math.round(60 + t * 180);
  const g = Math.round(90 + t * 90);
  const b = Math.round(200 * (1 - t) + 20 * t);
  return `0x${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

export default function MoleculeViewer({ sequence, attributionScores }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hasAttribution = attributionScores && attributionScores.length === sequence.length;

  useEffect(() => {
    if (!sequence || !containerRef.current) return;

    let cancelled = false;

    async function loadStructure() {
      setLoading(true);
      setError("");

      try {
        const data = await getStructure(sequence);
        const $3Dmol = await import("3dmol");

        if (cancelled || !containerRef.current) return;

        while (containerRef.current.firstChild) {
          containerRef.current.removeChild(containerRef.current.firstChild);
        }

        const viewer = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: "0x12121a",
        });

        viewer.addModel(data.pdb, "pdb");

        // Color by attribution if available, otherwise by amino acid properties
        for (let i = 0; i < sequence.length; i++) {
          let color: string;
          let sphereRadius: number;

          if (hasAttribution) {
            const score = attributionScores[i];
            color = attributionColor(score);
            // Important residues get bigger spheres
            sphereRadius = 0.5 + score * 0.5;
          } else {
            color = AA_COLORS[sequence[i]] || "#95a5a6";
            sphereRadius = 0.6;
          }

          viewer.setStyle(
            { resi: i + 1 },
            {
              stick: { radius: 0.3, color },
              sphere: { radius: sphereRadius, color },
            }
          );
        }

        for (let i = 0; i < sequence.length; i++) {
          const label = hasAttribution
            ? `${sequence[i]} ${Math.round((attributionScores[i]) * 100)}%`
            : sequence[i];

          viewer.addLabel(label, {
            position: { resi: i + 1 },
            fontSize: hasAttribution ? 10 : 12,
            fontColor: "white",
            backgroundColor: hasAttribution ? "0x222233" : "0x333344",
            backgroundOpacity: 0.8,
            showBackground: true,
          });
        }

        viewer.zoomTo();
        viewer.render();
        viewer.spin("y", 0.5);
        viewerRef.current = viewer;
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Failed to load structure"
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadStructure();

    return () => {
      cancelled = true;
    };
  }, [sequence, attributionScores, hasAttribution]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-lg font-semibold">3D Structure</h2>
        <div className="flex items-center gap-4 text-xs text-[var(--color-text-muted)]">
          {hasAttribution ? (
            <>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#3c5aa0]" />
                Low importance
              </span>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#f59e0b]" />
                Medium
              </span>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#ef4444]" />
                High importance
              </span>
            </>
          ) : (
            <>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#e74c3c]" />
                Hydrophobic
              </span>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#3498db]" />
                Polar
              </span>
              <span>
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-[#8e44ad]" />
                Charged
              </span>
            </>
          )}
        </div>
      </div>

      <div className="relative overflow-hidden rounded-lg border border-[var(--color-border)]">
        <div
          ref={containerRef}
          className="h-[350px] w-full bg-[var(--color-surface)]"
        />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-surface)]">
            <div className="flex items-center gap-2 text-sm text-[var(--color-text-muted)]">
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
              Generating 3D structure...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--color-surface)]">
            <p className="text-sm text-[var(--color-danger)]">{error}</p>
          </div>
        )}
      </div>

      <div className="mt-2 flex justify-center">
        <p className="font-mono text-sm tracking-[0.3em] text-[var(--color-text-muted)]">
          {sequence.split("").join(" ")}
        </p>
      </div>
    </div>
  );
}
