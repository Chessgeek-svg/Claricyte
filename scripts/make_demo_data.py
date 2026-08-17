"""Build a self-contained dataset for the hosted demo.

The full corpus (roughly 120k images across six source datasets) is far too large
to commit, and the demo only ever reads the validation split. This copies just
those images into demo_data/ alongside cut-down CSVs that point at the new
location, so a fresh clone can run `streamlit run app.py` with no dataset
acquisition step.

Image paths are rewritten with forward slashes. The originals were built by
ingest.py with pathlib on Windows, and a backslash is an ordinary filename
character on Linux, so those paths resolve to nothing on a hosted instance.

    python scripts/make_demo_data.py
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from claricyte.vocab import ATTRIBUTES, CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes", default="metadata/attributes.csv")
    parser.add_argument("--metadata", default="metadata/metadata.csv")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="demo_data")
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="cap images per class, for a smaller package (default: all)",
    )
    args = parser.parse_args()

    out = Path(args.out)
    images_dir = out / "images"

    attributes = pd.read_csv(args.attributes)
    metadata = pd.read_csv(args.metadata)
    merged = attributes.merge(metadata, on="image_path", how="inner")

    rows = merged[
        (merged["split"] == args.split) & (merged["claricyte_label"].isin(CLASSES))
    ]
    if args.per_class:
        rows = rows.groupby("claricyte_label", group_keys=False).head(args.per_class)
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"no rows for split={args.split!r}")

    # Copy under demo_data/images/<Class>/<filename>, so the package is browsable
    # and a missing image is obvious from the directory listing.
    copied, missing = [], []
    for row in rows.itertuples():
        source = Path(str(row.image_path).replace("\\", "/"))
        if not source.exists():
            missing.append(str(source))
            copied.append(None)
            continue
        target = images_dir / row.claricyte_label.replace(" ", "_") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        # as_posix keeps forward slashes in the CSV regardless of host platform
        copied.append(target.as_posix())

    if missing:
        raise SystemExit(
            f"{len(missing)} source images are missing, e.g. {missing[:3]}. "
            "Run this from the repo root with the full dataset present."
        )

    rows["image_path"] = copied

    # Two CSVs with the same column layout the full pipeline uses, so app.py needs
    # only its two path constants changed and nothing else.
    rows[["image_path", *ATTRIBUTES]].to_csv(out / "attributes.csv", index=False)
    rows[
        ["image_path", "source_dataset", "original_label", "claricyte_label", "split"]
    ].to_csv(out / "metadata.csv", index=False)

    total_mb = sum(f.stat().st_size for f in images_dir.rglob("*")) / 1e6
    print(f"wrote {len(rows)} images to {images_dir} ({total_mb:.1f} MB)")
    print(rows["claricyte_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
