from __future__ import annotations

import cv2
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PREVIEW_MANIFEST = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_preview_manifest.csv"
LABELS_CSV = ROOT / "raw_downloads" / "automobile" / "national_gallery" / "metadata" / "automobile_labels.csv"

YES_DIR = ROOT / "filtered" / "automobile_yes"
NO_DIR = ROOT / "filtered" / "automobile_no"

YES_DIR.mkdir(parents=True, exist_ok=True)
NO_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(PREVIEW_MANIFEST)

if LABELS_CSV.exists():
    labels = pd.read_csv(LABELS_CSV)
    labeled_ids = set(labels["objectid"].astype(int).tolist())
else:
    labels = pd.DataFrame(columns=["objectid", "title", "preview_path", "iiifurl", "label"])
    labeled_ids = set()

rows_to_add = []

for _, row in manifest.iterrows():
    object_id = int(row["objectid"])
    if object_id in labeled_ids:
        continue

    preview_raw = row.get("preview_path", "")
    if pd.isna(preview_raw) or not str(preview_raw).strip():
        continue

    preview_path = Path(str(preview_raw))
    if not preview_path.exists():
        continue

    img = cv2.imread(str(preview_path))
    if img is None:
        continue

    display = img.copy()
    cv2.putText(
        display,
        f"objectid={object_id} | Y=yes  N=no  Q=quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Label NGA automobile preview", display)
    key = cv2.waitKey(0)

    if key == ord("y"):
        label = "yes"
    elif key == ord("n"):
        label = "no"
    elif key == ord("q"):
        break
    else:
        continue

    rows_to_add.append({
        "objectid": object_id,
        "title": row.get("title", ""),
        "preview_path": str(preview_path),
        "iiifurl": row.get("iiifurl", ""),
        "label": label,
    })

    # write incrementally so progress is not lost
    tmp = pd.DataFrame(rows_to_add)
    out_df = pd.concat([labels, tmp], ignore_index=True)
    out_df.to_csv(LABELS_CSV, index=False)

cv2.destroyAllWindows()
print(f"Wrote labels to: {LABELS_CSV}")