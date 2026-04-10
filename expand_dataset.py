"""
Meal Shield — Dataset Expansion Pipeline

Scrapes BIOPEP-UWM for thousands of bioactive peptides, merges with
existing food_peptides.csv, and adds new prediction targets (ACE, DPP-4).

Usage:
    python expand_dataset.py                    # Full scrape + merge
    python expand_dataset.py --quick            # Quick test (50 details per activity)
    python expand_dataset.py --activities ace_inhibitor dpp4_inhibitor
"""

import os
import logging
import pandas as pd
import numpy as np

from scrape_biopep import scrape_all, ACTIVITY_MAP

logger = logging.getLogger(__name__)

# New ChEMBL targets to add to data.py
NEW_CHEMBL_TARGETS = {
    "ace_inhibitor": {
        "chembl_id": "CHEMBL1808",
        "alt_ids": ["CHEMBL4525"],
        "description": "ACE — angiotensin-converting enzyme (blood pressure)",
    },
    "dpp4_inhibitor": {
        "chembl_id": "CHEMBL284",
        "alt_ids": [],
        "description": "DPP-4 — dipeptidyl peptidase IV (blood sugar)",
    },
}

# Map BIOPEP activity labels to our model target names
BIOPEP_TO_TARGET = {
    "ace_inhibitor": "ace_inhibitor",
    "dpp4_inhibitor": "dpp4_inhibitor",
    "alpha_glucosidase": "alpha_glucosidase",
    "lipase": "lipase",
    "antioxidant": "antioxidant",
    "antimicrobial": "antimicrobial",
    "anti_inflammatory": "anti_inflammatory",
    "anticancer": "anticancer",
    "renin_inhibitor": "renin_inhibitor",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_existing_peptides(path="data/food_peptides.csv"):
    """Load the existing curated food peptide dataset."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Existing dataset: {len(df)} peptides")
    return df


def scrape_biopep(activities=None, max_details=None):
    """Run the BIOPEP-UWM scraper."""
    if activities is None:
        activities = list(BIOPEP_TO_TARGET.keys())

    logger.info(f"Scraping BIOPEP for {len(activities)} activities...")
    df = scrape_all(
        activities=activities,
        fetch_details=True,
        max_details_per_activity=max_details,
    )

    if df.empty:
        logger.warning("No peptides scraped from BIOPEP")
        return df

    # Map activities to our target names
    df["target"] = df["activity"].map(BIOPEP_TO_TARGET)
    df = df[df["target"].notna()].copy()

    # Filter to valid amino acid sequences only
    df = df[df["sequence"].apply(
        lambda s: bool(s) and 2 <= len(s) <= 50 and all(aa in VALID_AA for aa in s)
    )].copy()

    logger.info(f"BIOPEP scraped: {len(df)} valid peptides")
    return df


def merge_datasets(existing_df, biopep_df):
    """Merge existing food peptides with BIOPEP-scraped peptides."""
    records = []

    # Existing peptides
    if not existing_df.empty:
        for _, row in existing_df.iterrows():
            records.append({
                "sequence": row["sequence"],
                "activity": row.get("activity", "unknown"),
                "ic50_uM": row.get("ic50_uM"),
                "pIC50": row.get("pIC50"),
                "source": row.get("source", "curated"),
            })

    # BIOPEP peptides
    if not biopep_df.empty:
        for _, row in biopep_df.iterrows():
            ic50 = row.get("ic50_uM")
            pic50 = None
            if pd.notna(ic50) and ic50 > 0:
                pic50 = 9 - np.log10(ic50 * 1000)  # IC50 uM -> pIC50

            records.append({
                "sequence": row["sequence"],
                "activity": row.get("target", row.get("activity", "unknown")),
                "ic50_uM": ic50 if pd.notna(ic50) else None,
                "pIC50": pic50,
                "source": row.get("source", "BIOPEP-UWM"),
            })

    merged = pd.DataFrame(records)

    # Deduplicate: same sequence + activity -> keep the one with IC50
    merged = merged.sort_values("pIC50", ascending=False, na_position="last")
    merged = merged.drop_duplicates(subset=["sequence", "activity"], keep="first")

    # For peptides without IC50, assign a moderate default pIC50
    # (enables them to contribute to training as weakly active)
    mask_no_pic50 = merged["pIC50"].isna()
    merged.loc[mask_no_pic50, "pIC50"] = 4.0  # ~100 uM assumed
    merged.loc[mask_no_pic50, "ic50_uM"] = 100.0

    return merged


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Expand peptide dataset with BIOPEP-UWM")
    parser.add_argument("--activities", nargs="+", default=None,
                        help="Activity labels to scrape (default: all mapped)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: limit to 50 detail pages per activity")
    parser.add_argument("--output", default="data/food_peptides_expanded.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    max_details = 50 if args.quick else None

    # 1. Load existing
    existing = load_existing_peptides()

    # 2. Scrape BIOPEP
    biopep = scrape_biopep(activities=args.activities, max_details=max_details)

    # 3. Merge
    merged = merge_datasets(existing, biopep)

    # 4. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

    # 5. Report
    print(f"\n{'='*60}")
    print("DATASET EXPANSION SUMMARY")
    print(f"{'='*60}")
    print(f"Existing peptides: {len(existing)}")
    print(f"BIOPEP scraped:    {len(biopep)}")
    print(f"Merged total:      {len(merged)} unique (sequence, activity) pairs")
    print(f"With real IC50:    {(merged['source'] != 'BIOPEP-UWM').sum() + merged['ic50_uM'].notna().sum()}")
    print(f"\nBy activity:")
    for activity, count in merged["activity"].value_counts().items():
        with_ic50 = merged[(merged["activity"] == activity) & (merged["pIC50"] != 4.0)].shape[0]
        print(f"  {activity:<25} {count:>5} peptides ({with_ic50} with measured IC50)")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
