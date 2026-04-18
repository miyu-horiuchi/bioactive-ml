export interface TargetPrediction {
  target: string;
  label: string;
  pIC50: number;
  IC50_uM: number;
}

export interface PredictResponse {
  sequence: string;
  predictions: TargetPrediction[];
  demo_mode: boolean;
}

export interface TDAFeature {
  name: string;
  value: number;
}

export interface TDAResponse {
  sequence: string;
  features: TDAFeature[];
  dimension_summary: Record<
    string,
    {
      count: number;
      mean_persistence: number;
      max_persistence: number;
      interpretation: string;
    }
  >;
}

export interface StructureResponse {
  sequence: string;
  pdb: string;
  num_residues: number;
}

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "";

export async function predictPeptide(
  sequence: string
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export async function computeTDA(sequence: string): Promise<TDAResponse> {
  const res = await fetch(`${BASE}/api/tda`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "TDA computation failed");
  }
  return res.json();
}

export async function getStructure(
  sequence: string
): Promise<StructureResponse> {
  const res = await fetch(`${BASE}/api/structure`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Structure generation failed");
  }
  return res.json();
}

export interface ResidueAttribution {
  residue: string;
  position: number;
  score: number;
}

export interface ExplainResponse {
  sequence: string;
  target: string;
  method: string;
  residues: string[];
  scores: number[];
  top_residues: ResidueAttribution[];
  demo_mode: boolean;
}

export async function explainPrediction(
  sequence: string,
  target: string,
  method: string = "attention"
): Promise<ExplainResponse> {
  const res = await fetch(`${BASE}/api/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence, target, method }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Explanation failed");
  }
  return res.json();
}

// ── Design pipeline types ──────────────────────────────────

export interface DesignCandidate {
  sequence: string;
  pIC50: number;
  IC50_uM: number;
  solubility?: number;
  toxicity?: number;
  bitterness?: number;
  hemolysis?: number;
  stability?: number;
  developability?: number;
}

export interface GenerateResponse {
  target: string;
  method: string;
  candidates: DesignCandidate[];
}

export interface PropertiesResponse {
  sequence: string;
  properties: Record<string, number>;
}

export async function generatePeptides(
  target: string,
  method: string = "genetic",
  length: number = 5,
  n_candidates: number = 50,
  top_k: number = 10,
): Promise<GenerateResponse> {
  const res = await fetch(`${BASE}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target, method, length, n_candidates, top_k }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Generation failed");
  }
  return res.json();
}

export async function scoreProperties(
  sequence: string,
): Promise<PropertiesResponse> {
  const res = await fetch(`${BASE}/api/properties`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Property scoring failed");
  }
  return res.json();
}

export function activityLevel(pIC50: number): "high" | "medium" | "low" {
  if (pIC50 >= 6.0) return "high";
  if (pIC50 >= 4.5) return "medium";
  return "low";
}

export function formatIC50(ic50: number): string {
  if (ic50 >= 10000) return ">10 mM";
  if (ic50 >= 1000) return `${(ic50 / 1000).toFixed(1)} mM`;
  if (ic50 >= 1) return `${ic50.toFixed(1)} uM`;
  return `${(ic50 * 1000).toFixed(0)} nM`;
}
