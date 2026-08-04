import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from claricyte.vocab import ATTRIBUTES, CLASS_TO_INDEX, VALUE_TO_INDEX


class MorphologyDataset(Dataset):
    def __init__(self, attributes_filepath, metadata_filepath, split=None):
        self.attributes_filepath = attributes_filepath
        self.metadata_filepath = metadata_filepath
        self.split = split

        df = pd.read_csv(self.attributes_filepath)
        metadata = pd.read_csv(self.metadata_filepath)
        df = df.merge(metadata, on="image_path", how="inner")

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
                #v2.RandomRotation(degrees=15), # requires a fill for corners
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
