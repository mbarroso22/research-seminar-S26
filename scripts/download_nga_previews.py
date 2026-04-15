from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests
import time
import re

ROOT = Path(__file__).resolve().parent.parent
META_CSV = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_candidates.csv"
OUT_DIR = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "previews"
MANIFEST_CSV = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_preview_manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META_CSV)

# first pass: preview download only
SLEEP_SEC = 0.15
PREVIEW_SIZE = 256  # try 256 first; increase to 384 if images are too small to inspect

def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120]

rows = []
count = 0

for _, row in df.iterrows():
   

    iiif_url = str(row["iiifurl"]).strip()
    object_id = int(row["objectid"])
    title = str(row.get("title", "untitled"))

    preview_url = iiif_url.rstrip("/") + f"/full/{PREVIEW_SIZE},/0/default.jpg"
    fname = f"nga_obj_{object_id}_{safe_name(title)}_preview.jpg"
    out_path = OUT_DIR / fname

    if out_path.exists():
        rows.append({
            "objectid": object_id,
            "title": title,
            "iiifurl": iiif_url,
            "preview_url": preview_url,
            "preview_path": str(out_path),
            "status_code": 200,
        })
        continue

    try:
        r = requests.get(preview_url, timeout=30)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            rows.append({
                "objectid": object_id,
                "title": title,
                "iiifurl": iiif_url,
                "preview_url": preview_url,
                "preview_path": str(out_path),
                "status_code": r.status_code,
            })
            count += 1
            print(f"[{count}] saved {fname}")
        else:
            rows.append({
                "objectid": object_id,
                "title": title,
                "iiifurl": iiif_url,
                "preview_url": preview_url,
                "preview_path": "",
                "status_code": r.status_code,
            })
            print(f"skip {object_id} status={r.status_code}")
    except Exception as e:
        rows.append({
            "objectid": object_id,
            "title": title,
            "iiifurl": iiif_url,
            "preview_url": preview_url,
            "preview_path": "",
            "status_code": f"ERROR: {e}",
        })
        print(f"error {object_id}: {e}")

    time.sleep(SLEEP_SEC)

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_CSV, index=False)

print(f"Wrote manifest: {MANIFEST_CSV}")
print(f"Downloaded previews: {count}")
