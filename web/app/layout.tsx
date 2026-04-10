import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Meal Shield — Peptide Bioactivity Prediction",
  description:
    "Predict food peptide bioactivity against digestive enzyme targets using GNN + Topological Data Analysis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[var(--color-border)] px-6 py-4">
          <div className="mx-auto flex max-w-5xl items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-accent)] text-sm font-bold text-white">
              MS
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">
                Meal Shield
              </h1>
              <p className="text-xs text-[var(--color-text-muted)]">
                GNN + TDA peptide bioactivity prediction
              </p>
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
