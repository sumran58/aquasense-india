"""
Real CGWB Data Loader
=====================
Handles the actual column formats from:
  1. India Data Portal CGWB CSV
  2. India WRIS export
  3. CGWB district-wise PDFs converted to CSV
  4. State groundwater board CSVs

Run this first to understand your data:
    python -m app.data.loader --file data/raw/cgwb.csv --preview
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path


# ── Column name maps for known CGWB sources ────────────────────────────────
# Add your CSV's actual column names here if they differ

COLUMN_MAPS = {
    # India Data Portal format (most common)
    "india_data_portal": {
        "STATE_UT_NAME":   "state",
        "DISTRICT_NAME":   "district",
        "SITE_TYPE":       "well_type",
        "LATITUDE":        "latitude",
        "LONGITUDE":       "longitude",
        "YEAR":            "year",
        "SEASON":          "season",
        "WL_BELOW_GL":     "water_level_mbgl",
    },
    # India WRIS format
    "india_wris": {
        "State":           "state",
        "District":        "district",
        "StationName":     "well_id",
        "Latitude":        "latitude",
        "Longitude":       "longitude",
        "Year":            "year",
        "Season":          "season",
        "WaterLevel":      "water_level_mbgl",
    },
    # Generic / auto-detect format
    "generic": {
        "state_name":      "state",
        "district_name":   "district",
        "lat":             "latitude",
        "lon":             "longitude",
        "longitude":       "longitude",
        "year":            "year",
        "month":           "month",
        "depth_m":         "water_level_mbgl",
        "water_level":     "water_level_mbgl",
        "gwl":             "water_level_mbgl",
        "depth_mbgl":      "water_level_mbgl",
    },
}

# CGWB season → quarter mapping
SEASON_TO_QUARTER = {
    "JAN": 1, "JANUARY": 1, "PRE_MONSOON_JAN": 1,
    "MAY": 2, "PRE-MONSOON": 2, "PREMONSOON": 2, "MAR": 2, "APRIL": 2,
    "AUG": 3, "AUGUST": 3, "MONSOON": 3, "POST_MONSOON_AUG": 3,
    "NOV": 4, "NOVEMBER": 4, "POST-MONSOON": 4, "POSTMONSOON": 4, "OCT": 4,
    "1": 1, "2": 2, "3": 3, "4": 4,
}

MONTH_TO_QUARTER = {
    1: 1, 2: 1, 3: 1,     # Jan–Mar → Q1
    4: 2, 5: 2, 6: 2,     # Apr–Jun → Q2
    7: 3, 8: 3, 9: 3,     # Jul–Sep → Q3
    10: 4, 11: 4, 12: 4,  # Oct–Dec → Q4
}


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """
    Try to automatically detect which columns map to what.
    Returns a rename dictionary.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}

    # Try each known column map
    for fmt_name, col_map in COLUMN_MAPS.items():
        for src_col, target_col in col_map.items():
            if src_col.lower() in cols_lower:
                rename[cols_lower[src_col.lower()]] = target_col

    return rename


def load_cgwb_csv(filepath: str, source_format: str = "auto") -> pd.DataFrame:
    """
    Load and standardize a CGWB CSV file into the project schema.

    Args:
        filepath:      Path to your downloaded CSV
        source_format: "auto", "india_data_portal", "india_wris", or "generic"

    Returns:
        DataFrame with columns: state, district, year, quarter,
                                water_level_mbgl, latitude, longitude,
                                season, well_id
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Download real data from:\n"
            f"  https://ckandev.indiadataportal.com/dataset/groundwater\n"
            f"  https://indiawris.gov.in/wris/#/"
        )

    print(f"\nLoading: {filepath.name}")
    print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    # Read CSV
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

    print(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Raw columns: {list(df.columns)}")

    # ── Column renaming ────────────────────────────────────────────────────
    if source_format == "auto":
        rename_map = auto_detect_columns(df)
    else:
        rename_map = COLUMN_MAPS.get(source_format, {})

    df = df.rename(columns=rename_map)
    print(f"\nAfter rename, available columns: {list(df.columns)}")

    # Convert date → year + quarter
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        df["quarter"] = df["month"].map({
            1:1,2:1,3:1,
            4:2,5:2,6:2,
            7:3,8:3,9:3,
            10:4,11:4,12:4
        })

    # Map target column
    if "currentlevel" in df.columns:
        df["water_level_mbgl"] = df["currentlevel"
        ""]

    # ── Quarter derivation ─────────────────────────────────────────────────
    if "quarter" not in df.columns:
        if "season" in df.columns:
            df["quarter"] = df["season"].astype(str).str.upper().str.strip().map(SEASON_TO_QUARTER)
            unmapped = df["quarter"].isna().sum()
            if unmapped > 0:
                print(f"  Warning: {unmapped} rows have unrecognised season values — will be dropped")
                unique_seasons = df.loc[df["quarter"].isna(), "season"].unique()[:10]
                print(f"  Unknown season values: {unique_seasons}")
        elif "month" in df.columns:
            df["quarter"] = df["month"].astype(int).map(MONTH_TO_QUARTER)

    # ── Validate required columns ──────────────────────────────────────────
    required = ["state", "district", "year", "quarter", "water_level_mbgl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Edit COLUMN_MAPS in loader.py to map your column names."
        )

    # ── Type coercion ──────────────────────────────────────────────────────
    df["year"]             = pd.to_numeric(df["year"], errors="coerce")
    df["quarter"]          = pd.to_numeric(df["quarter"], errors="coerce")
    df["water_level_mbgl"] = pd.to_numeric(df["water_level_mbgl"], errors="coerce")

    if "latitude" in df.columns:
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # ── Cleaning ───────────────────────────────────────────────────────────
    before = len(df)

    # Remove rows with no water level
    df = df.dropna(subset=["state", "district", "year", "quarter", "water_level_mbgl"])

    # Remove physically impossible values
    df = df[df["water_level_mbgl"] > 0]
    df = df[df["water_level_mbgl"] < 200]   # 200m is extreme but possible in some areas

    # Remove future years
    current_year = 2024
    df = df[df["year"] <= current_year]
    df = df[df["year"] >= 1969]

    # Quarter must be 1–4
    df = df[df["quarter"].between(1, 4)]

    after = len(df)
    print(f"\nCleaning: removed {before - after:,} invalid rows")

    # ── Standardize text ───────────────────────────────────────────────────
    df["state"]    = df["state"].str.strip().str.title()
    df["district"] = df["district"].str.strip().str.title()

    # Fix common state name variations
    state_fixes = {
        "Jammu And Kashmir": "Jammu & Kashmir",
        "Andaman And Nicobar Islands": "Andaman & Nicobar",
        "Dadra And Nagar Haveli": "Dadra & Nagar Haveli",
        "Uttaranchal": "Uttarakhand",
        "Orissa": "Odisha",
    }
    df["state"] = df["state"].replace(state_fixes)

    # ── Final columns ──────────────────────────────────────────────────────
    keep = ["state", "district", "year", "quarter", "water_level_mbgl"]
    for optional in ["latitude", "longitude", "season", "well_id", "rainfall_mm"]:
        if optional in df.columns:
            keep.append(optional)

    df = df[keep].reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"CGWB Data Loaded Successfully")
    print(f"{'='*50}")
    print(f"  Total records  : {len(df):,}")
    print(f"  States         : {df['state'].nunique()}")
    print(f"  Districts      : {df['district'].nunique()}")
    print(f"  Year range     : {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"  Water level    : {df['water_level_mbgl'].min():.1f} – {df['water_level_mbgl'].max():.1f} mbgl")
    print(f"  Avg level      : {df['water_level_mbgl'].mean():.2f} mbgl")
    print(f"{'='*50}\n")

    return df


def load_imd_rainfall(filepath: str) -> pd.DataFrame:
    """
    Load IMD district rainfall CSV.
    Download from: https://imdpune.gov.in/

    Expected columns: state, district, year, month_or_season, rainfall_mm
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    rename = {}
    for col in df.columns:
        if "state" in col:        rename[col] = "state"
        elif "district" in col:   rename[col] = "district"
        elif "year" in col:       rename[col] = "year"
        elif "rainfall" in col or "rain" in col: rename[col] = "rainfall_mm"

    df = df.rename(columns=rename)

    if "month" in df.columns:
        df["quarter"] = df["month"].astype(int).map(MONTH_TO_QUARTER)
        df = df.groupby(["state", "district", "year", "quarter"])["rainfall_mm"].sum().reset_index()

    return df


def merge_rainfall(cgwb_df: pd.DataFrame, rainfall_df: pd.DataFrame) -> pd.DataFrame:
    """Merge rainfall data onto CGWB groundwater data by district/year/quarter."""
    merged = cgwb_df.merge(
        rainfall_df[["state", "district", "year", "quarter", "rainfall_mm"]],
        on=["state", "district", "year", "quarter"],
        how="left",
    )
    coverage = merged["rainfall_mm"].notna().mean() * 100
    print(f"Rainfall data coverage: {coverage:.1f}% of records matched")
    return merged


def preview_csv(filepath: str):
    """Quick preview of a CSV file to help you understand its structure."""
    try:
        df = pd.read_csv(filepath, nrows=5)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, nrows=5, encoding="latin-1")

    print(f"\nFile: {filepath}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.to_string())


# ── CLI for testing ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGWB Data Loader")
    parser.add_argument("--file",    required=True, help="Path to CSV file")
    parser.add_argument("--preview", action="store_true", help="Preview file columns only")
    parser.add_argument("--format",  default="auto", help="Source format: auto/india_data_portal/india_wris/generic")
    args = parser.parse_args()

    if args.preview:
        preview_csv(args.file)
    else:
        df = load_cgwb_csv(args.file, source_format=args.format)
        print(df.head())
        print(f"\nData types:\n{df.dtypes}")
