import random

import torch

from claricyte.data import MorphologyDataset
from claricyte.explain import explain
from claricyte.model import Model
from claricyte.predict import contributions, predict
from claricyte.vocab import CLASSES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model.from_checkpoint(path="checkpoints/class_head_mlp.pt", device=device)
model.eval()

valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

# Eyeball quiz-mode explanations on a handful of random val cells.
for i in random.sample(range(len(valset)), 5):
    image_tensor, _, class_target = valset[i]
    true_label = CLASSES[class_target]

    result, class_dist, concepts = predict(model, image_tensor)
    predicted_label = max(class_dist, key=lambda c: class_dist[c])
    scores = contributions(model, result, concepts, true_label)

    print(valset.df.iloc[i]["image_path"])
    print(f"true={true_label}  predicted={predicted_label}")
    print(explain(result, scores, true_label))
    print()
