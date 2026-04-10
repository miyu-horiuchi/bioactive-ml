"""
External Database Integration — Peptipedia + DFBP

Downloads and parses bioactive peptide data from:
  1. Peptipedia v2.0 — REST API (103K labeled peptides, 213 activities)
  2. DFBP — HTML scraping (7,058 food-derived peptides with IC50, 31 categories)

Both are integrated into the fetch_peptides.py pipeline via --external flag.
"""

import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Dict

import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/external_cache")

# ============================================================
# Shared utilities
# ============================================================

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    safe = re.sub(r"[^\w\-]", "_", key)[:50]
    return CACHE_DIR / f"{safe}_{h}.json"


def _cached_request(url: str, method: str = "GET", delay: float = 1.0,
                    encoding: str = "utf-8", **kwargs) -> str:
    cache_key = method + url + json.dumps(kwargs.get("params", {}), sort_keys=True)
    cp = _cache_path(cache_key)

    if cp.exists():
        return cp.read_text(encoding="utf-8")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    time.sleep(delay)
    logger.info(f"Fetching: {url}")

    resp = requests.request(method, url, timeout=60,
                            headers={"User-Agent": "BioactivePeptideResearch/1.0"},
                            **kwargs)
    resp.encoding = encoding
    text = resp.text
    cp.write_text(text, encoding="utf-8")
    return text


# ============================================================
# 1. Peptipedia v2.0
# ============================================================

PEPTIPEDIA_API = "https://api.app.peptipedia.cl/api"

# Activity IDs relevant to food bioactive peptide research.
# Obtained from /api/get_count_activities_table/
PEPTIPEDIA_ACTIVITIES = {
    "ace_inhibitor": {"id": 40, "name": "ACE inhibitor"},
    "dpp4_inhibitor": {"id": 51, "name": "DPP-IV inhibitor"},
    "antioxidant": {"id": 14, "name": "antioxidant"},
    "antimicrobial": {"id": 1, "name": "antimicrobial"},
    "anticancer": {"id": 5, "name": "anticancer"},
    "anti_inflammatory": {"id": 7, "name": "anti-inflammatory"},
    "antihypertensive": {"id": 39, "name": "antihypertensive"},
    "antifungal": {"id": 3, "name": "antifungal"},
    "antiviral": {"id": 4, "name": "antiviral"},
    "immunomodulating": {"id": 10, "name": "immunomodulatory"},
    "antithrombotic": {"id": 42, "name": "antithrombotic"},
    "opioid": {"id": 48, "name": "opioid"},
    "bitter": {"id": 56, "name": "bitter"},
    "umami": {"id": 55, "name": "umami"},
}


def fetch_peptipedia_activity_list() -> List[dict]:
    """Get all activities and their peptide counts from Peptipedia."""
    text = _cached_request(f"{PEPTIPEDIA_API}/get_count_activities_table/")
    data = json.loads(text)
    return data


def fetch_peptipedia_peptide(peptide_id: int) -> Optional[dict]:
    """Fetch a single peptide profile from Peptipedia."""
    text = _cached_request(f"{PEPTIPEDIA_API}/get_peptide/{peptide_id}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def download_peptipedia_dump(output_dir: str = "data/peptipedia") -> Optional[str]:
    """
    Download the Peptipedia PostgreSQL dump from Google Drive.
    Requires gdown: pip install gdown

    Returns path to the downloaded file, or None if gdown unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "peptipedia.sql")

    if os.path.exists(output_path):
        logger.info(f"Peptipedia dump already exists: {output_path}")
        return output_path

    try:
        import gdown
    except ImportError:
        logger.warning("gdown not installed. Run: pip install gdown")
        logger.warning("Falling back to API-based extraction (slower).")
        return None

    logger.info("Downloading Peptipedia database dump (~large file)...")
    gdown.download(
        id="1uvTGOdjpsPYxvx00g8KbMv5tTDKsjSAg",
        output=output_path,
        quiet=False,
    )
    return output_path


def parse_peptipedia_sql_sequences(sql_path: str,
                                   activities: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract peptide sequences and activities from the Peptipedia SQL dump.
    This is a lightweight parser that reads INSERT statements without needing PostgreSQL.
    """
    logger.info(f"Parsing Peptipedia SQL dump: {sql_path}")

    # Read the SQL file and extract peptide inserts
    sequences = {}  # id -> sequence
    peptide_activities = []  # (peptide_id, activity_id, predicted)
    activity_names = {}  # id -> name

    with open(sql_path, "r", errors="replace") as f:
        for line in f:
            # Extract activity names
            if "INSERT INTO" in line and "activity" in line.lower() and "peptide_has" not in line.lower():
                for m in re.finditer(r"\((\d+),\s*'([^']+)'", line):
                    activity_names[int(m.group(1))] = m.group(2)

            # Extract peptide sequences
            if "INSERT INTO" in line and "peptide" in line.lower() and "has" not in line.lower():
                for m in re.finditer(r"\((\d+),\s*'([A-Z]+)'", line):
                    sequences[int(m.group(1))] = m.group(2)

            # Extract peptide-activity associations
            if "INSERT INTO" in line and "peptide_has_activity" in line.lower():
                for m in re.finditer(r"\((\d+),\s*(\d+),\s*(true|false)", line):
                    peptide_activities.append({
                        "peptide_id": int(m.group(1)),
                        "activity_id": int(m.group(2)),
                        "predicted": m.group(3) == "true",
                    })

    logger.info(f"  Parsed {len(sequences)} sequences, {len(peptide_activities)} activity links, {len(activity_names)} activities")

    # Build dataframe of experimentally validated (not predicted) peptides
    records = []
    target_activity_ids = None
    if activities:
        target_activity_ids = set()
        for act in activities:
            if act in PEPTIPEDIA_ACTIVITIES:
                target_activity_ids.add(PEPTIPEDIA_ACTIVITIES[act]["id"])

    for pa in peptide_activities:
        if pa["predicted"]:
            continue  # Skip predicted, keep only experimentally labeled
        if target_activity_ids and pa["activity_id"] not in target_activity_ids:
            continue
        seq = sequences.get(pa["peptide_id"])
        if not seq:
            continue
        act_name = activity_names.get(pa["activity_id"], f"activity_{pa['activity_id']}")
        records.append({
            "sequence": seq,
            "activity_peptipedia": act_name,
            "peptipedia_id": pa["peptide_id"],
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["sequence", "activity_peptipedia"])
    logger.info(f"  Filtered to {len(df)} experimentally labeled peptides")
    return df


# ============================================================
# 2. DFBP (Database of Food-derived Bioactive Peptides)
# ============================================================

DFBP_BASE = "http://www.cqudfbp.net"

# Category IDs mapped to our internal activity labels
DFBP_CATEGORIES = {
    "ace_inhibitor": "ace_inhibitory_peptides",
    "renin_inhibitor": "renin_inhibitory_peptides",
    "antihypertensive": "antihypertensive_peptides",
    "antioxidant": "antioxidative_peptides",
    "antimicrobial": "antimicrobial_peptides",
    "anticancer": "anticancer_peptides",
    "antithrombotic": "antithrombotic_peptides",
    "dpp4_inhibitor": "dpp_iv_inhibitory_peptides",
    "alpha_glucosidase": "alpha_glucosidase_inhibitory_peptides",
    "immunomodulating": "immunomodulatory_peptides",
    "opioid": "opioid_peptides",
    "mineral_binding": "mineral_binding_peptides",
    "hypocholesterolemic": "hypocholesterolemic_peptides",
    "celiac_toxic": "alpha_gliadin_peptides",
}

# Categories that have IC50 columns in their list tables
DFBP_IC50_CATEGORIES = {
    "ace_inhibitory_peptides", "renin_inhibitory_peptides",
    "dpp_iv_inhibitory_peptides", "alpha_glucosidase_inhibitory_peptides",
    "alpha_amylase_inhibitory_peptides", "pep_peptides",
}


def scrape_dfbp_category_sequences(category: str) -> List[dict]:
    """
    Scrape all peptide IDs and sequences from a DFBP download page.
    Returns list of dicts with dfbp_id and sequence.
    """
    url = f"{DFBP_BASE}/download/downloadPage.jsp"
    html = _cached_request(url, params={"pageMark": category}, delay=2.0,
                           encoding="utf-8")

    entries = []
    # Pattern: DFBPACEI0003: IHPF  (in <div> or plain text)
    for m in re.finditer(r'(DFBP\w+)\s*:\s*([A-Z]+)', html):
        entries.append({
            "dfbp_id": m.group(1),
            "sequence": m.group(2),
        })

    logger.info(f"  DFBP {category}: {len(entries)} sequences from download page")
    return entries


def scrape_dfbp_list_page(category: str, page: int = 1,
                          page_size: int = 500) -> pd.DataFrame:
    """
    Scrape the list display table for a DFBP category.
    Extracts IC50 for inhibitory categories, source for others.
    """
    url = f"{DFBP_BASE}/commonPages/ListDisplay/listDisplay.jsp"
    params = {"tableNames": category, "cp": str(page), "ls": str(page_size)}
    html = _cached_request(url, params=params, delay=2.0, encoding="utf-8")

    try:
        tables = pd.read_html(html)
        if not tables:
            return pd.DataFrame()
        df = tables[0]
        # Normalize column names
        df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"Failed to parse DFBP list for {category}: {e}")
        return pd.DataFrame()


def scrape_dfbp_activity(activity_label: str,
                         max_pages: int = 20) -> pd.DataFrame:
    """
    Scrape all peptides for a given activity from DFBP.
    Returns DataFrame with: sequence, ic50_uM (if available), source, activity, dfbp_id
    """
    category = DFBP_CATEGORIES.get(activity_label)
    if not category:
        logger.warning(f"Unknown DFBP activity: {activity_label}")
        return pd.DataFrame()

    has_ic50 = category in DFBP_IC50_CATEGORIES

    # First get all sequences from the download page
    entries = scrape_dfbp_category_sequences(category)
    if not entries:
        return pd.DataFrame()

    # Then try to get IC50 / source info from list pages
    all_list_data = []
    for page in range(1, max_pages + 1):
        df = scrape_dfbp_list_page(category, page=page, page_size=500)
        if df.empty:
            break
        all_list_data.append(df)
        if len(df) < 500:
            break  # Last page

    list_df = pd.concat(all_list_data, ignore_index=True) if all_list_data else pd.DataFrame()

    # Build IC50 lookup from list data
    ic50_map = {}
    source_map = {}
    if not list_df.empty:
        seq_col = None
        for c in list_df.columns:
            if "sequence" in c or "aa_sequence" in c:
                seq_col = c
                break

        ic50_col = None
        for c in list_df.columns:
            if "ic50" in c:
                ic50_col = c
                break

        source_col = None
        for c in list_df.columns:
            if "organism" in c or "source" in c:
                source_col = c
                break

        if seq_col:
            for _, row in list_df.iterrows():
                seq = str(row.get(seq_col, "")).strip().upper()
                if not seq or not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
                    continue

                if ic50_col and has_ic50:
                    ic50_raw = str(row.get(ic50_col, ""))
                    # Parse IC50 — may contain units, ranges, etc.
                    m = re.search(r'([\d.]+)', ic50_raw)
                    if m:
                        try:
                            val = float(m.group(1))
                            if val > 0:
                                ic50_map[seq] = val
                        except ValueError:
                            pass

                if source_col:
                    src = str(row.get(source_col, "")).strip()
                    if src and src != "nan":
                        source_map[seq] = src

    # Combine download page sequences with list page IC50/source data
    records = []
    for entry in entries:
        seq = entry["sequence"]
        records.append({
            "sequence": seq,
            "ic50_uM": ic50_map.get(seq),
            "source": source_map.get(seq, f"DFBP ({entry['dfbp_id']})"),
            "activity": activity_label,
            "dfbp_id": entry["dfbp_id"],
            "data_source": "dfbp",
        })

    df = pd.DataFrame(records)
    logger.info(f"  DFBP {activity_label}: {len(df)} peptides ({sum(1 for r in records if r['ic50_uM'])} with IC50)")
    return df


def scrape_dfbp_all(activities: Optional[List[str]] = None) -> pd.DataFrame:
    """Scrape all DFBP categories of interest."""
    if activities is None:
        activities = list(DFBP_CATEGORIES.keys())

    all_dfs = []
    for act in activities:
        if act not in DFBP_CATEGORIES:
            continue
        df = scrape_dfbp_activity(act)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Validate sequences
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    valid_mask = combined["sequence"].apply(
        lambda s: bool(s) and all(aa in valid_aas for aa in s)
    )
    combined = combined[valid_mask].copy()

    # Deduplicate
    combined = combined.sort_values("ic50_uM", na_position="last")
    combined = combined.drop_duplicates(subset=["sequence", "activity"], keep="first")

    return combined


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch external peptide databases")
    parser.add_argument("--dfbp", action="store_true", help="Scrape DFBP")
    parser.add_argument("--peptipedia-dump", action="store_true",
                        help="Download Peptipedia SQL dump")
    parser.add_argument("--activities", nargs="+", default=None)
    parser.add_argument("--output", default="data/external_peptides.csv")
    args = parser.parse_args()

    all_dfs = []

    if args.dfbp:
        print("=== Scraping DFBP ===")
        df = scrape_dfbp_all(args.activities)
        if not df.empty:
            all_dfs.append(df)
            print(f"DFBP: {len(df)} peptides")

    if args.peptipedia_dump:
        print("=== Downloading Peptipedia ===")
        path = download_peptipedia_dump()
        if path:
            df = parse_peptipedia_sql_sequences(path, args.activities)
            if not df.empty:
                df["data_source"] = "peptipedia"
                df["ic50_uM"] = None  # Peptipedia doesn't have IC50
                df.rename(columns={"activity_peptipedia": "activity"}, inplace=True)
                all_dfs.append(df)
                print(f"Peptipedia: {len(df)} peptides")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved {len(combined)} peptides to {args.output}")
        print(f"\nBy activity:")
        print(combined["activity"].value_counts().to_string())
    else:
        print("No data fetched. Use --dfbp or --peptipedia-dump.")
