import torch

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.training import attr_only_loss, balanced_loader, train

# Windows spawns fresh processes for DataLoader workers (num_workers>0), and each
# one re-imports this module. Without this guard, every worker would re-run all
# the training code below instead of just importing the functions it needs.
if __name__ == "__main__":
    torch.manual_seed(0)

    ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
    valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

    # Backbone is frozen in Model.__init__; stage 1 trains ONLY the attribute heads,
    # on the attribute loss. The class head is left untrained here and fitted
    # separately in stage 2 (scripts/train_class_head.py), so the class objective
    # never leaks back into the concept predictions.
    model = Model("resnet50")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.attribute_heads.parameters(), lr=1e-3, weight_decay=0.01
    )

    loader = balanced_loader(trainset, batch_size=8)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    train(
        model,
        loader,
        val_loader,
        optimizer,
        attr_only_loss,
        device,
        epochs=20,
        checkpoint_path="checkpoints/attr_heads.pt",
        accum_steps=4,
        log_every=50,
        select_by="attr",
    )
