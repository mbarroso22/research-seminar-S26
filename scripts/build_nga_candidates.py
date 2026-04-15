from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NGA_DIR = ROOT / "nga" / "opendata" / "data"

OBJECTS_CSV = NGA_DIR / "objects.csv"
TERMS_CSV = NGA_DIR / "objects_terms.csv"
IMAGES_CSV = NGA_DIR / "published_images.csv"

OUT_DIR = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- load ----------------
objects = pd.read_csv(OBJECTS_CSV, low_memory=False)
terms = pd.read_csv(TERMS_CSV, low_memory=False)
images = pd.read_csv(IMAGES_CSV, low_memory=False)

# normalize columns
objects.columns = objects.columns.str.strip().str.lower()
terms.columns = terms.columns.str.strip().str.lower()
images.columns = images.columns.str.strip().str.lower()

# ---------------- image filtering ----------------
images = images.copy()
images = images[images["openaccess"] == 1]
images = images[images["depictstmsobjectid"].notna()]
images = images[images["iiifurl"].notna()]

if "viewtype" in images.columns:
    images = images[(images["viewtype"].isna()) | (images["viewtype"] == "primary")]

images["depictstmsobjectid"] = images["depictstmsobjectid"].astype(int)

# ---------------- object filtering ----------------
objects = objects.copy()
if "isvirtual" in objects.columns:
    objects = objects[objects["isvirtual"] == 0]

# ---------------- merge images -> objects ----------------
merged = images.merge(
    objects,
    left_on="depictstmsobjectid",
    right_on="objectid",
    how="inner",
    suffixes=("_img", "_obj")
)

# ---------------- attach terms ----------------
terms_small = terms[["objectid", "termtype", "term"]].copy()
terms_small["term"] = terms_small["term"].fillna("").astype(str)

term_agg = (
    terms_small.groupby("objectid")["term"]
    .apply(lambda s: " | ".join(sorted(set(t.strip() for t in s if t.strip()))))
    .reset_index()
    .rename(columns={"term": "all_terms"})
)

merged = merged.merge(term_agg, on="objectid", how="left")
merged["all_terms"] = merged["all_terms"].fillna("")

# ---------------- broad automobile candidate filter ----------------
keywords = [
    "car", "automobile", "motorcar", "vehicle", "wagon", "cart", "truck"
]

def row_matches(row) -> bool:
    blob = " ".join([
        str(row.get("title", "")),
        str(row.get("all_terms", "")),
        str(row.get("classification", "")),
        str(row.get("subclassification", "")),
        str(row.get("visualbrowserclassification", "")),
    ]).lower()
    return any(k in blob for k in keywords)

candidates = merged[merged.apply(row_matches, axis=1)].copy()

keep_cols = [
    "objectid",
    "title",
    "displaydate",
    "beginyear",
    "endyear",
    "medium",
    "classification",
    "subclassification",
    "visualbrowserclassification",
    "iiifurl",
    "iiifthumburl",
    "viewtype",
    "width",
    "height",
    "maxpixels",
    "openaccess",
    "all_terms",
]

keep_cols = [c for c in keep_cols if c in candidates.columns]
candidates = candidates[keep_cols].drop_duplicates()

all_valid_path = OUT_DIR / "nga_all_valid_objects_with_images.csv"
candidate_path = OUT_DIR / "automobile_candidates.csv"

merged.to_csv(all_valid_path, index=False)
candidates.to_csv(candidate_path, index=False)

print(f"Wrote: {all_valid_path}")
print(f"Wrote: {candidate_path}")
print(f"Candidate rows: {len(candidates)}")