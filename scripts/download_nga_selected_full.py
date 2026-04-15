from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests
import time
import re

ROOT = Path(__file__).resolve().parent.parent

LABELS_CSV = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_labels.csv"
OUT_DIR = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "selected_full"
MANIFEST_CSV = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_selected_full_manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LABELS_CSV)
df = df[df["label"] == "yes"].copy()

SLEEP_SEC = 0.15
FINAL_SIZE = 1024  # good for experiments; models resize later anyway

def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120]

rows = []
count = 0

for _, row in df.iterrows():
    object_id = int(row["objectid"])
    title = str(row.get("title", "untitled"))
    iiif_url = str(row["iiifurl"]).strip()

    full_url = iiif_url.rstrip("/") + f"/full/{FINAL_SIZE},/0/default.jpg"
    fname = f"nga_obj_{object_id}_{safe_name(title)}.jpg"
    out_path = OUT_DIR / fname

    if out_path.exists():
        rows.append({
            "objectid": object_id,
            "title": title,
            "iiifurl": iiif_url,
            "download_url": full_url,
            "saved_path": str(out_path),
            "status_code": 200,
        })
        continue

    try:
        r = requests.get(full_url, timeout=30)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            rows.append({
                "objectid": object_id,
                "title": title,
                "iiifurl": iiif_url,
                "download_url": full_url,
                "saved_path": str(out_path),
                "status_code": r.status_code,
            })
            count += 1
            print(f"[{count}] saved {fname}")
        else:
            rows.append({
                "objectid": object_id,
                "title": title,
                "iiifurl": iiif_url,
                "download_url": full_url,
                "saved_path": "",
                "status_code": r.status_code,
            })
            print(f"skip {object_id} status={r.status_code}")
    except Exception as e:
        rows.append({
            "objectid": object_id,
            "title": title,
            "iiifurl": iiif_url,
            "download_url": full_url,
            "saved_path": "",
            "status_code": f"ERROR: {e}",
        })
        print(f"error {object_id}: {e}")

    time.sleep(SLEEP_SEC)

manifest = pd.DataFrame(rows)
manifest.to_csv(MANIFEST_CSV, index=False)

print(f"Wrote manifest: {MANIFEST_CSV}")
print(f"Downloaded selected full images: {count}")