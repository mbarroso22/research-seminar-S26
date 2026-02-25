import pandas as pd

FILES = [
    "results/corruptions_appearance_efficientnet.csv",
    "results/corruptions_geometry_efficientnet.csv",
    "results/corruptions_appearance_openclip.csv",
    "results/corruptions_geometry_openclip.csv",
]

N = 33

for f in FILES:
    df = pd.read_csv(f)
    keys = ["model", "axis", "transform", "level"]
    for k, g in df.groupby(keys):
        ids = sorted(g["image_id"].tolist())
        ok = (ids == list(range(N)))
        if not ok:
            print("FAIL:", f, k)
            print("  min/max:", min(ids), max(ids))
            print("  unique:", len(set(ids)), "count:", len(ids))
            missing = sorted(set(range(N)) - set(ids))
            dupes = len(ids) - len(set(ids))
            print("  missing:", missing[:20], "..." if len(missing) > 20 else "")
            print("  dupes:", dupes)
            raise SystemExit(1)
    print("PASS:", f)

print("All image_id checks passed.")