import warnings

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from claricyte.vocab import (
    ATTRIBUTES,
    CLASS_TO_INDEX,
    CLASSES,
    VALUE_TO_INDEX,
    find_invalid_rows,
)


class MorphologyDataset(Dataset):
    def __init__(self, attributes_filepath, metadata_filepath, split=None):
        self.attributes_filepath = attributes_filepath
        self.metadata_filepath = metadata_filepath
        self.split = split

        df = pd.read_csv(self.attributes_filepath)
        metadata = pd.read_csv(self.metadata_filepath)
        df = df.merge(metadata, on="image_path", how="inner")

        # ingest.py built these paths with pathlib on Windows, so they arrive with
        # backslash separators. On Linux a backslash is an ordinary filename
        # character, not a separator, so those paths resolve to nothing once the app
        # is hosted. Forward slashes work on both platforms. Done once here rather
        # than per item in __getitem__, which runs 1.5k times per epoch.
        df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)

        # Keep only classes the model has an output unit for. Without this, the
        # first attribute-labeled Blast or Erythroblast would raise a bare KeyError
        # from CLASS_TO_INDEX partway through an epoch. Filtering here also makes
        # the trained class set an explicit property of vocab.CLASSES rather than an
        # accident of which rows happen to carry attribute labels. The warning
        # matters as much as the filter: silently dropping rows would let a typo in
        # a label name, or a genuinely new class, disappear without a trace.
        known = df["claricyte_label"].isin(CLASSES)
        if not known.all():
            dropped = df.loc[~known, "claricyte_label"].value_counts().to_dict()
            warnings.warn(
                f"dropping {(~known).sum()} rows whose class is not in vocab.CLASSES: "
                f"{dropped}. Add them to vocab.CLASSES to train on them.",
                stacklevel=2,
            )
            df = df[known]

        # Validate BEFORE encoding. The .map() below turns any value outside the
        # vocabulary into NaN, which then survives all the way to __getitem__ and
        # surfaces mid-epoch as a cryptic int64 cast error with no indication of
        # which column or which file caused it.
        invalid = find_invalid_rows(df)
        if not invalid.empty:
            offenders = {
                attr: sorted(
                    set(
                        invalid.loc[
                            ~invalid[attr].isin(VALUE_TO_INDEX[attr]), attr
                        ].astype(str)
                    )
                )
                for attr in ATTRIBUTES
                if not invalid[attr].isin(VALUE_TO_INDEX[attr]).all()
            }
            raise ValueError(
                f"{len(invalid)} rows hold attribute values outside "
                f"vocab.ATTRIBUTE_VOCAB: {offenders}. "
                f"First offending image: {invalid['image_path'].iloc[0]}"
            )

        for attr in ATTRIBUTES:
            df[attr] = df[attr].map(VALUE_TO_INDEX[attr])

        if self.split is not None:
            df = df[df["split"] == self.split]

        df = df.reset_index(drop=True)
        self.df = df

        train_pipeline = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.RandomRotation(degrees=15), # requires a fill for corners
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        eval_pipeline = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),  # PIL -> uint8 tensor, Channel, Height, Width
                v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
                # ImageNet hardcoded mean/std, could consider calculating my own from my
                # dataset aggregate to balance out color tint and / or stain quality
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform = train_pipeline if self.split == "train" else eval_pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 11 attribute targets, in ATTRIBUTES order, as an int64 tensor -> shape (11,)
        attr_targets = torch.tensor(row[ATTRIBUTES].to_numpy(dtype="int64"))

        # class target: string -> canonical int index
        class_target = CLASS_TO_INDEX[row["claricyte_label"]]

        path = row["image_path"]
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, attr_targets, class_target


if __name__ == "__main__":
    ds = MorphologyDataset("metadata/attributes.csv", "metadata/metadata.csv")
    print(len(ds))
    print(ds[0])
