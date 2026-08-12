import random

import torch

from claricyte.data import MorphologyDataset
from claricyte.explain import explain
from claricyte.model import Model
from claricyte.predict import contributions, predict
from claricyte.vocab import CLASSES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model("resnet50")
model.load_state_dict(
    torch.load("checkpoints/best_model_seq.pt", map_location=device, weights_only=True)
)
model.to(device)
model.eval()

valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

# Eyeball quiz-mode explanations on a handful of random val cells.
for i in random.sample(range(len(valset)), 5):
    image_tensor, _, class_target = valset[i]
    true_label = CLASSES[class_target]

    result, class_dist = predict(model, image_tensor)
    predicted_label = max(class_dist, key=lambda c: class_dist[c])
    scores = contributions(model, result, true_label)

    print(valset.df.iloc[i]["image_path"])
    print(f"true={true_label}  predicted={predicted_label}")
    print(explain(result, scores, true_label))
    print()
