"""Ingest raw datasets into a single standardized metadata file."""

import json
from pathlib import Path
from typing import Iterator, NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from claricyte.vocab import ATTRIBUTES


class RawRow(NamedTuple):
    """One image as emitted by an adapter, before taxonomy mapping/splitting."""

    image_path: Path
    source_dataset: str
    original_label: str


def ingest_mll23(root: Path) -> Iterator[RawRow]:
    """Yield one RawRow per image in the MLL23 dataset.

    MLL23 is folder-based: each subdirectory of `root` is a class, and the
    folder name is the original label for every image inside it.
    """
    for folder in root.iterdir():
        if not folder.is_dir() or "MACOSX" in folder.name:
            continue
        # images live in an inner folder of the same name (class/class/*.tif)
        for image in (folder / folder.name).glob("*.tif"):
            yield RawRow(
                image_path=image, source_dataset="mll23", original_label=folder.name
            )


# original MLL23 folder name -> one of the 17 Claricyte classes
MLL23_LABEL_MAP: dict[str, str] = {
    "basophil": "Basophil",
    "eosinophil": "Eosinophil",
    "hairy_cell": "Hairy Cell",
    "lymphocyte": "Lymphocyte",
    "lymphocyte_large_granular": "Lymphocyte",
    "lymphocyte_neoplastic": "Blast",
    # NOTE: MLL23 count for reactive lymphs (33) is unreliable; may drop later
    "lymphocyte_reactive": "Reactive Lymphocyte",
    "metamyelocyte": "Metamyelocyte",
    "monocyte": "Monocyte",
    "myeloblast": "Blast",
    "myelocyte": "Myelocyte",
    "neutrophil_band": "Band Neutrophil",
    "neutrophil_segmented": "Segmented Neutrophil",
    "normoblast": "Erythroblast",
    "plasma_cell": "Plasma Cell",
    "promyelocyte": "Promyelocyte",
    "promyelocyte_atypical": "Promyelocyte",
    "smudge_cell": "Smudge Cell",
}


def ingest_hrls(root: Path) -> Iterator[RawRow]:
    """Yield one RawRow per image in the HRLS dataset.

    HRLS is folder-based: each subdirectory of `root` is a class, and the
    folder name is the original label for every image inside it.
    """
    for folder in root.iterdir():
        if not folder.is_dir() or "MACOSX" in folder.name:
            continue
        for image in folder.glob("*.bmp"):
            yield RawRow(
                image_path=image, source_dataset="hrls", original_label=folder.name
            )


# original HRLS folder name -> one of the 8 Claricyte classes
HRLS_LABEL_MAP: dict[str, str] = {
    "Basophile": "Basophil",
    "Eosinophile": "Eosinophil",
    "Lymphoblast": "Blast",
    "Lymphocyte": "Lymphocyte",
    "Monocyte": "Monocyte",
    "Myeloblast": "Blast",
    "Neutrophile Band": "Band Neutrophil",
    "Neutrophile Segment": "Segmented Neutrophil",
    "Normoblast": "Erythroblast",
}

# Acevedo/PBC class -> Claricyte class. The neutrophil and ig folders are mixed,
# their real class comes from the filename prefix (BNE/SNE, MMY/MY/PMY), not the folder
ACEVEDO_LABEL_MAP: dict[str, str] = {
    "basophil": "Basophil",
    "eosinophil": "Eosinophil",
    "erythroblast": "Erythroblast",
    "lymphocyte": "Lymphocyte",
    "monocyte": "Monocyte",
    "platelet": "Giant Platelet",
    "BNE": "Band Neutrophil",
    "SNE": "Segmented Neutrophil",
    "NEUTROPHIL": "Segmented Neutrophil",
    "MMY": "Metamyelocyte",
    "MY": "Myelocyte",
    "PMY": "Promyelocyte",
}

# folders whose real class is in the filename prefix, not the folder name
ACEVEDO_PREFIX_FOLDERS = {"neutrophil", "ig"}


def ingest_acevedo(root: Path) -> Iterator[RawRow]:
    """Yield one RawRow per Acevedo/PBC image (folder-labeled, .jpg).

    In the neutrophil and ig folders the class is in the filename prefix; the
    generic IG_ files aren't sub-typed, so they're skipped.
    """
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name == "labels":
            continue
        for image in folder.glob("*.jpg"):
            if folder.name in ACEVEDO_PREFIX_FOLDERS:
                label = image.name.split("_")[0]
                if label == "IG":
                    continue
            else:
                label = folder.name
            yield RawRow(
                image_path=image, source_dataset="acevedo", original_label=label
            )


def wbcatt_attributes(root: Path) -> pd.DataFrame:
    """Read WB-CAtt's per-image attribute labels (an annotation layer on Acevedo).

    Joined to the metadata on image_path. Only the 5 annotated classes appear.
    """
    files = ["pbc_attr_v1_train.csv", "pbc_attr_v1_val.csv", "pbc_attr_v1_test.csv"]
    data = pd.concat(
        [pd.read_csv(root / "labels" / f) for f in files], ignore_index=True
    )
    data["image_path"] = [
        str(root / label.lower() / img_name)
        for label, img_name in zip(data["label"], data["img_name"])
    ]
    attributes = data[["image_path", *ATTRIBUTES]].copy()
    attributes["source"] = "wbcatt"
    return attributes


RAABIN_LABEL_MAP = {
    1: "Segmented Neutrophil",  # Raabin doesn't split band/seg, default to seg
    2: "Lymphocyte",
    3: "Monocyte",
    4: "Eosinophil",
    5: "Basophil",
}


def ingest_raabin(root: Path) -> Iterator[RawRow]:
    for subset in ("Train", "Test"):
        with open(root / f"{subset}.json") as f:
            data = json.load(f)

        for filename, code in data.items():
            yield RawRow(
                image_path=root / subset / filename,
                source_dataset="raabin",
                original_label=code,
            )


YARIKAN_LABEL_MAP = {
    "segmented_neutrophil": "Segmented Neutrophil",
    "band_neutrophil": "Band Neutrophil",
    "lymphocyte": "Lymphocyte",
    "reactive_lymphocyte": "Reactive Lymphocyte",
    "monocyte": "Monocyte",
    "eosinophil": "Eosinophil",
    "basophil": "Basophil",
    "blast": "Blast",
    "myelocyte": "Myelocyte",
    "metamyelocyte": "Metamyelocyte",
    "erythroblast": "Erythroblast",
    "giant_platelet": "Giant Platelet",
    "platelet_cluster": "Platelet Cluster",
}


def ingest_yarikan(root: Path) -> pd.DataFrame:
    """Ingest Yarikan/Koc. CSV-labeled, ships its own patient-level split.

    The on-disk class folders are already the Claricyte names; the CSV `path`
    gives the physical train/val/test folder, which can differ from the split
    column since the split is re-assigned per patient.
    """
    data = pd.read_csv(root / "metadata_with_patient_level_splits.csv")
    claricyte = data["cell_type"].map(YARIKAN_LABEL_MAP)
    physical = data["path"].str.split("/").str[0]
    data["image_path"] = [
        str(root / "dataset" / phys / label / name)
        for phys, label, name in zip(physical, claricyte, data["image_name"])
    ]
    return pd.DataFrame(
        {
            "image_path": data["image_path"],
            "source_dataset": "yarikan",
            "original_label": data["cell_type"],
            "claricyte_label": claricyte,
            "split": data["split"].replace({"validation": "val"}),
        }
    )


def assign_splits(metadata: pd.DataFrame, val_frac: float, test_frac: float) -> None:
    """Fill the 'split' column in place for rows that don't already have one.

    Rows arriving with `split` already set (datasets that ship an official
    train/val/test split, e.g. WB-CAtt) are left untouched. The remaining rows
    are split with stratification on `claricyte_label`.
    """
    unassigned = metadata.index[metadata["split"].isna()].tolist()
    if not unassigned:
        return

    labels = metadata["claricyte_label"].loc[unassigned].tolist()

    trainval_idx, test_idx = train_test_split(
        unassigned, test_size=test_frac, stratify=labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_frac / (1 - test_frac),
        stratify=metadata["claricyte_label"].loc[trainval_idx].tolist(),
        random_state=42,
    )

    metadata.loc[train_idx, "split"] = "train"
    metadata.loc[val_idx, "split"] = "val"
    metadata.loc[test_idx, "split"] = "test"


def main() -> None:
    data_root = Path("data")

    LABEL_MAPS = {
        "mll23": MLL23_LABEL_MAP,
        "hrls": HRLS_LABEL_MAP,
        "raabin": RAABIN_LABEL_MAP,
        "acevedo": ACEVEDO_LABEL_MAP,
    }

    raw: list[RawRow] = []
    raw.extend(ingest_mll23(data_root / "MLL23"))
    raw.extend(ingest_hrls(data_root / "HRLS" / "Labelled"))
    raw.extend(ingest_raabin(data_root / "Raabin-WBC"))
    raw.extend(ingest_acevedo(data_root / "PBC_dataset_WBCAtt"))

    folder_rows = [
        {
            "image_path": str(r.image_path),
            "source_dataset": r.source_dataset,
            "original_label": r.original_label,
            "claricyte_label": LABEL_MAPS[r.source_dataset][r.original_label],
            "split": None,  # filled in by assign_splits
        }
        for r in raw
    ]
    metadata = pd.DataFrame(folder_rows)

    # Yarikan ships its own patient-level split, so it bypasses assign_splits
    metadata = pd.concat(
        [metadata, ingest_yarikan(data_root / "yarikan")], ignore_index=True
    )

    assign_splits(metadata, val_frac=0.15, test_frac=0.15)

    # WB-CAtt attribute labels sit on the Acevedo images, joined on image_path
    attributes = wbcatt_attributes(data_root / "PBC_dataset_WBCAtt")

    metadata.to_csv("metadata/metadata.csv", index=False)
    attributes.to_csv("metadata/attributes.csv", index=False)


if __name__ == "__main__":
    main()
