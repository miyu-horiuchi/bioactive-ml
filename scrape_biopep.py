"""
BIOPEP-UWM Database Scraper

Programmatic access to the BIOPEP-UWM bioactive peptide database
(https://biochemia.uwm.edu.pl/biopep-uwm/).

No API is available — this scrapes the PHP web application directly.
Includes caching and rate limiting to be respectful of the academic server.

Two-phase approach:
  Phase 1: Bulk search by activity category → get IDs, sequences, names
  Phase 2: Detail page enrichment → get IC50/EC50, references, SMILES
"""

import os
import re
import json
import time
import hashlib
import logging
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://biochemia.uwm.edu.pl/biopep"
CACHE_DIR = Path("data/biopep_cache")

# Activity categories relevant for food bioactive peptide research.
# Maps our internal activity label → BIOPEP search term.
ACTIVITY_MAP = {
    # Original targets
    "ace_inhibitor": "ACE inhibitor",
    "dpp4_inhibitor": "dipeptidyl peptidase IV inhibitor",
    "alpha_glucosidase": "alpha-glucosidase inhibitor",
    "lipase": "pancreatic lipase inhibitor",
    "antioxidant": "antioxidative",
    "bile_acid_binding": "bile acid binding",
    "mineral_binding": "mineral binding",
    # New targets
    "antimicrobial": "antibacterial",
    "antifungal": "antifungal",
    "anti_inflammatory": "anti inflammatory",
    "anticancer": "anticancer",
    "antithrombotic": "antithrombotic",
    "immunomodulating": "immunomodulating",
    "renin_inhibitor": "renin inhibitor",
    "hypotensive": "hypotensive",
    "opioid": "opioid",
    "celiac_toxic": "celiac toxic",
    "bitter": "bitter",
    "umami": "umami",
}

# Rate limiting
REQUEST_DELAY = 1.5  # seconds between requests


class BiopepSearchParser(HTMLParser):
    """Parse BIOPEP search results HTML into structured peptide records."""

    def __init__(self):
        super().__init__()
        self.peptides = []
        self._in_data_row = False
        self._in_cell = False
        self._current_row = []
        self._current_cell = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "tr" and attrs_dict.get("class") == "info":
            self._in_data_row = True
            self._current_row = []
        elif tag == "td" and self._in_data_row:
            self._in_cell = True
            self._current_cell = ""
        elif tag == "a" and self._in_cell:
            href = attrs_dict.get("href", "")
            # Extract zm_ID from detail page link
            m = re.search(r"zm_ID=(\d+)", href)
            if m:
                self._current_cell = f"__ID__{m.group(1)}__"

    def handle_endtag(self, tag):
        if tag == "td" and self._in_cell:
            self._in_cell = False
            self._current_row.append(self._current_cell.strip().rstrip("\xa0"))
        elif tag == "tr" and self._in_data_row:
            self._in_data_row = False
            if len(self._current_row) >= 6:
                self._parse_row(self._current_row)

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data

    def handle_entityref(self, name):
        if self._in_cell and name == "nbsp":
            pass  # skip &nbsp;

    def _parse_row(self, cells):
        """Convert a row of cells into a peptide record."""
        # Typical columns: [link_with_id, id, name, sequence, chem_mass, mono_mass, activity, inchikey]
        # Some searches may have fewer columns
        biopep_id = None
        for cell in cells:
            m = re.match(r"__ID__(\d+)__", cell)
            if m:
                biopep_id = int(m.group(1))
                break

        # Find the numeric ID column (usually index 1)
        record = {"biopep_id": biopep_id}
        # The layout after link is: ID, Name, Sequence, ChemMass, MonoMass, [Activity], [InChIKey]
        idx = 0
        for cell in cells:
            if cell.startswith("__ID__"):
                idx += 1
                continue
            clean = cell.strip().rstrip("\xa0").strip()
            if idx == 1:
                try:
                    record["biopep_id"] = record.get("biopep_id") or int(clean)
                except ValueError:
                    pass
            elif idx == 2:
                record["name"] = clean
            elif idx == 3:
                record["sequence"] = clean.upper()
            elif idx == 4:
                try:
                    record["chem_mass"] = float(clean)
                except ValueError:
                    record["chem_mass"] = None
            elif idx == 5:
                try:
                    record["mono_mass"] = float(clean)
                except ValueError:
                    record["mono_mass"] = None
            elif idx == 6:
                record["activity_biopep"] = clean
            elif idx == 7:
                record["inchikey"] = clean
            idx += 1

        if record.get("sequence"):
            self.peptides.append(record)


def parse_detail_page(html: str) -> dict:
    """
    Parse a BIOPEP peptide detail card page using regex.

    The detail page uses a table layout with class="infobold" for labels
    and class="info" for values. The sequence is in a <div> after "sequence".
    IC50 is in a <td> after "IC50 :".
    """
    result = {
        "sequence": None,
        "ic50_uM": None,
        "ic50_type": "IC50",
        "name": None,
        "smiles": None,
        "inchikey": None,
    }

    # Sequence: in <div ... class="info">\nSEQUENCE</div>
    m = re.search(r'<div[^>]*class="info"[^>]*>\s*([A-Z]+)\s*</div', html)
    if m:
        result["sequence"] = m.group(1).strip()

    # Name: <td ... class="info">NAME</td> after "Name" header
    m = re.search(r'Name</td>\s*<td[^>]*class="info"[^>]*>(.*?)</td>', html, re.DOTALL)
    if m:
        result["name"] = m.group(1).strip()

    # IC50: look for "IC50 :" or "EC50 :" followed by value
    m = re.search(r'(IC50|EC50)\s*:\s*</td>\s*<td[^>]*>\s*<div[^>]*>\s*([\d.]+)', html)
    if m:
        result["ic50_type"] = m.group(1)
        try:
            val = float(m.group(2))
            if val > 0:
                result["ic50_uM"] = val
        except ValueError:
            pass

    # SMILES: in additional info section
    m = re.search(r'SMILES:\s*([^\s<]+)', html)
    if m:
        result["smiles"] = m.group(1).strip()

    # InChIKey
    m = re.search(r'InChIKey:\s*([A-Z\-]+)', html)
    if m:
        result["inchikey"] = m.group(1).strip()

    return result


def _cache_path(key: str) -> Path:
    """Get cache file path for a given key."""
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    safe_key = re.sub(r"[^\w\-]", "_", key)[:60]
    return CACHE_DIR / f"{safe_key}_{h}.json"


def _cached_get(url: str, params: Optional[Dict] = None, encoding: str = "iso-8859-1") -> str:
    """HTTP GET with file-based caching and rate limiting."""
    cache_key = url + (json.dumps(params, sort_keys=True) if params else "")
    cp = _cache_path(cache_key)

    if cp.exists():
        logger.debug(f"Cache hit: {cp.name}")
        return cp.read_text(encoding="utf-8")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    time.sleep(REQUEST_DELAY)
    logger.info(f"Fetching: {url}")

    resp = requests.get(url, params=params, timeout=60, headers={
        "User-Agent": "BioactivePeptideResearch/1.0 (academic research)"
    })
    resp.encoding = encoding
    html = resp.text

    cp.write_text(html, encoding="utf-8")
    return html


def search_by_activity(activity_label: str) -> List[dict]:
    """
    Search BIOPEP for all peptides with a given activity.
    Returns list of dicts with: biopep_id, name, sequence, chem_mass, mono_mass, activity_biopep, inchikey.
    """
    search_term = ACTIVITY_MAP.get(activity_label, activity_label)
    url = f"{BASE_URL}/peptide_data_search.php"
    params = {"txt_search": search_term, "menu_search": "activity"}

    html = _cached_get(url, params)

    parser = BiopepSearchParser()
    parser.feed(html)

    for p in parser.peptides:
        p["activity"] = activity_label

    logger.info(f"  {activity_label}: found {len(parser.peptides)} peptides")
    return parser.peptides


def fetch_detail(biopep_id: int) -> Optional[dict]:
    """
    Fetch the detail card for a single peptide.
    Returns dict with IC50/EC50, sequence, name, SMILES, InChIKey.
    """
    url = f"{BASE_URL}/peptidedatacard.php"
    params = {"zm_ID": str(biopep_id)}

    try:
        html = _cached_get(url, params, encoding="utf-8")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch detail for {biopep_id}: {e}")
        return None

    result = parse_detail_page(html)
    result["biopep_id"] = biopep_id
    return result


def scrape_activity(activity_label: str, fetch_details: bool = True,
                    max_details: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape all peptides for a given activity from BIOPEP-UWM.

    Args:
        activity_label: Our internal activity label (e.g., 'ace_inhibitor')
        fetch_details: If True, also fetch detail pages for IC50/EC50 values
        max_details: Limit detail fetches (for testing). None = fetch all.

    Returns:
        DataFrame with columns: sequence, ic50_uM, source, activity, biopep_id, name
    """
    peptides = search_by_activity(activity_label)

    if not peptides:
        logger.warning(f"No peptides found for {activity_label}")
        return pd.DataFrame()

    if fetch_details:
        ids_to_fetch = [p["biopep_id"] for p in peptides if p.get("biopep_id")]
        if max_details:
            ids_to_fetch = ids_to_fetch[:max_details]

        logger.info(f"  Fetching {len(ids_to_fetch)} detail pages for IC50 values...")
        ic50_map = {}
        for i, pid in enumerate(ids_to_fetch):
            if pid is None:
                continue
            detail = fetch_detail(pid)
            if detail and detail.get("ic50_uM"):
                ic50_map[pid] = detail["ic50_uM"]
            if (i + 1) % 50 == 0:
                logger.info(f"    ... {i+1}/{len(ids_to_fetch)} details fetched")

        # Merge IC50 values
        for p in peptides:
            if p.get("biopep_id") in ic50_map:
                p["ic50_uM"] = ic50_map[p["biopep_id"]]

    records = []
    for p in peptides:
        seq = p.get("sequence", "").strip()
        if not seq:
            continue
        records.append({
            "sequence": seq,
            "ic50_uM": p.get("ic50_uM"),
            "source": f"BIOPEP-UWM (ID:{p.get('biopep_id', '?')})",
            "activity": activity_label,
            "biopep_id": p.get("biopep_id"),
            "name": p.get("name", ""),
        })

    df = pd.DataFrame(records)
    return df


def scrape_all(activities: Optional[List[str]] = None,
               fetch_details: bool = True,
               max_details_per_activity: Optional[int] = None) -> pd.DataFrame:
    """
    Scrape multiple activity categories from BIOPEP-UWM.

    Args:
        activities: List of activity labels to scrape. None = all in ACTIVITY_MAP.
        fetch_details: Fetch IC50/EC50 from detail pages.
        max_details_per_activity: Cap detail fetches per activity (for testing).

    Returns:
        Combined DataFrame of all scraped peptides.
    """
    if activities is None:
        activities = list(ACTIVITY_MAP.keys())

    all_dfs = []
    for activity in activities:
        if activity not in ACTIVITY_MAP:
            logger.warning(f"Unknown activity: {activity}, skipping")
            continue
        logger.info(f"Scraping {activity}...")
        df = scrape_activity(activity, fetch_details=fetch_details,
                             max_details=max_details_per_activity)
        if not df.empty:
            all_dfs.append(df)
            logger.info(f"  → {len(df)} peptides ({df['ic50_uM'].notna().sum()} with IC50)")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Validate sequences — only standard amino acids
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    valid_mask = combined["sequence"].apply(
        lambda s: bool(s) and all(aa in valid_aas for aa in s)
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.info(f"Filtered {n_invalid} peptides with non-standard amino acids")
    combined = combined[valid_mask].copy()

    # Deduplicate: keep entry with IC50 when available
    combined = combined.sort_values("ic50_uM", na_position="last")
    combined = combined.drop_duplicates(subset=["sequence", "activity"], keep="first")

    return combined


def save_scraped(df: pd.DataFrame, path: str = "data/biopep_scraped.csv"):
    """Save scraped data to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} peptides to {path}")
    return path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scrape BIOPEP-UWM database")
    parser.add_argument("--activities", nargs="+", default=None,
                        help="Activity labels to scrape (default: all)")
    parser.add_argument("--no-details", action="store_true",
                        help="Skip fetching detail pages (no IC50)")
    parser.add_argument("--max-details", type=int, default=None,
                        help="Max detail pages per activity (for testing)")
    parser.add_argument("--output", default="data/biopep_scraped.csv",
                        help="Output CSV path")
    parser.add_argument("--list-activities", action="store_true",
                        help="List available activity categories and exit")
    args = parser.parse_args()

    if args.list_activities:
        print("Available activity categories:")
        for label, search_term in sorted(ACTIVITY_MAP.items()):
            print(f"  {label:<25} → BIOPEP search: '{search_term}'")
        exit(0)

    df = scrape_all(
        activities=args.activities,
        fetch_details=not args.no_details,
        max_details_per_activity=args.max_details,
    )

    if df.empty:
        print("No peptides scraped.")
        exit(1)

    save_scraped(df, args.output)

    print(f"\n{'='*60}")
    print(f"BIOPEP-UWM SCRAPE SUMMARY")
    print(f"{'='*60}")
    print(f"Total peptides: {len(df)}")
    print(f"With IC50 values: {df['ic50_uM'].notna().sum()}")
    print(f"Unique sequences: {df['sequence'].nunique()}")
    print(f"\nBy activity:")
    for activity, count in df["activity"].value_counts().items():
        with_ic50 = df[df["activity"] == activity]["ic50_uM"].notna().sum()
        print(f"  {activity:<25} {count:>5} peptides ({with_ic50} with IC50)")
    print(f"\nSaved to: {args.output}")
