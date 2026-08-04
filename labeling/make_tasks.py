"""Generate a Label Studio task file for one Claricyte class.

Picks images of the target class that aren't labeled yet, converts non-viewable
formats (tif/bmp) to jpg under labeling/cache/, and writes tasks.json. Each task
keeps the true metadata image_path so exports map back to the real image.

Usage: python labeling/make_tasks.py --label Blast --n 300
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent  # Label Studio document root
CACHE_DIR = REPO_ROOT / "labeling" / "cache"
VIEWABLE = {".jpg", ".jpeg", ".png"}


def already_labeled() -> set[str]:
    labeled: set[str] = set()
    for name in ("attributes.csv", "manual_attributes.csv"):
        p = REPO_ROOT / "metadata" / name
        if p.exists():
            labeled |= set(pd.read_csv(p)["image_path"])
    return labeled


def viewable_copy(image_path: str) -> Path:
    """Copy/convert the image into labeling/cache/ so every task points at one
    folder (Label Studio serves a single registered local-storage path)."""
    src = REPO_ROOT / image_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(image_path.encode()).hexdigest()[:16]
    if src.suffix.lower() in VIEWABLE:
        dst = CACHE_DIR / f"{src.stem}_{key}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy(src, dst)
    else:
        dst = CACHE_DIR / f"{src.stem}_{key}.jpg"
        if not dst.exists():
            Image.open(src).convert("RGB").save(dst, "JPEG", quality=90)
    return dst


def local_files_url(abspath: Path) -> str:
    rel = abspath.relative_to(REPO_ROOT).as_posix()
    return "/data/local-files/?d=" + quote(rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="claricyte_label to serve")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="tasks.json")
    args = ap.parse_args()

    meta = pd.read_csv(REPO_ROOT / "metadata" / "metadata.csv")
    done = already_labeled()
    is_label = meta["claricyte_label"] == args.label
    not_done = ~meta["image_path"].isin(done)
    pool = meta[is_label & not_done]
    picked = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)

    tasks = []
    for row in picked.itertuples():
        view = viewable_copy(row.image_path)
        tasks.append(
            {
                "data": {
                    "image": local_files_url(view),
                    "image_path": row.image_path,
                    "claricyte_label": row.claricyte_label,
                    "source_dataset": row.source_dataset,
                }
            }
        )

    out = REPO_ROOT / "labeling" / args.out
    out.write_text(json.dumps(tasks, indent=2))
    print(f"{len(tasks)} tasks -> {out}  (pool had {len(pool)} unlabeled {args.label})")


if __name__ == "__main__":
    main()
