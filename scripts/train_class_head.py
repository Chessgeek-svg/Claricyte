"""Sequential CBM training, stage 2: train only the class head.

Stage 1 (train_attr_heads.py) already produced good attribute heads. Here we
FREEZE everything except the class head and retrain just that head on a
class-balanced loader. The class head then learns to separate classes from stable,
accurate concept vectors instead of chasing a moving target and riding the majority
class.
"""

import torch

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.training import balanced_loader, class_only_loss, train

# Windows spawns fresh processes for DataLoader workers (num_workers>0), and each
# one re-imports this module. Without this guard, every worker would re-run all
# the training code below instead of just importing the functions it needs.
if __name__ == "__main__":
    ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # Load the stage-1 model: good attribute heads, plus the untrained class head we
    # are about to fit.
    model = Model("resnet50")
    model.load_state_dict(
        torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)

    # Freeze the attribute heads so only the class head learns; reset the class head
    # so we start clean rather than from whatever came in on the checkpoint.
    for param in model.attribute_heads.parameters():
        param.requires_grad = False
    model.class_head.reset_parameters()

    trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
    valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")
    loader = balanced_loader(trainset, batch_size=32)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Only the class head trains, with no weight decay so its logits can spread
    # instead of being shrunk toward a flat prior.
    optimizer = torch.optim.AdamW(
        model.class_head.parameters(), lr=1e-3, weight_decay=0.0
    )

    train(
        model,
        loader,
        val_loader,
        optimizer,
        class_only_loss,
        device,
        epochs=15,
        checkpoint_path="checkpoints/best_model_seq.pt",
    )
