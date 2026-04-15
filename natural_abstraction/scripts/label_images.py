import cv2
import os
from pathlib import Path
import shutil

SOURCE = Path("../raw_downloads/car")
YES_DIR = Path("../filtered/car_yes")
NO_DIR = Path("../filtered/car_no")

YES_DIR.mkdir(exist_ok=True, parents=True)
NO_DIR.mkdir(exist_ok=True, parents=True)

images = list(SOURCE.glob("*.*"))

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    cv2.imshow("Label - Press Y (yes) / N (no) / Q (quit)", img)
    key = cv2.waitKey(0)

    if key == ord("y"):
        shutil.move(str(img_path), YES_DIR / img_path.name)
    elif key == ord("n"):
        shutil.move(str(img_path), NO_DIR / img_path.name)
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

