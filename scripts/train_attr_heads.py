import torch

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.training import attr_only_loss, balanced_loader, train

torch.manual_seed(0)

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

# Backbone is frozen in Model.__init__; stage 1 trains ONLY the attribute heads, on
# the attribute loss. The class head is left untrained here and fitted separately in
# stage 2 (scripts/train_class_head.py), so the class objective never leaks back into
# the concept predictions.
model = Model("resnet50")
model.to(device)
optimizer = torch.optim.AdamW(
    model.attribute_heads.parameters(), lr=1e-3, weight_decay=0.01
)

loader = balanced_loader(trainset, batch_size=8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)

train(
    model,
    loader,
    val_loader,
    optimizer,
    attr_only_loss,
    device,
    epochs=20,
    checkpoint_path="checkpoints/best_model.pt",
    accum_steps=4,
    log_every=50,
    select_by="attr",
)
