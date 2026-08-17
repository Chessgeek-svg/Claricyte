"""Sequential CBM training, stage 1: train the attribute heads.

Trains ONLY the attribute heads, on the attribute loss. The class head is built by
Model.__init__ but left untrained here and fitted separately in stage 2
(scripts/train_class_head.py), so the class objective never leaks back into the
concept predictions and bend them toward classifiability at the cost of honesty.

--unfreeze fine-tunes part of the backbone alongside the heads. Frozen ImageNet
features cap attribute accuracy at roughly 0.81, which is where most of the
pipeline's remaining error lives, but the train split is only ~7k images against
23.5M backbone parameters, so start with the last block rather than the lot.
"""

import argparse

import torch

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.training import attr_only_loss, balanced_loader, train

# Windows spawns fresh processes for DataLoader workers (num_workers>0), and each
# one re-imports this module. Without this guard, every worker would re-run all
# the training code below instead of just importing the functions it needs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unfreeze",
        default=None,
        help="backbone submodules to fine-tune, comma separated (e.g. layer3,layer4), "
        "or 'all' (default: none)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="learning rate for unfrozen backbone params, normally 10x lower than "
        "the head rate: the backbone starts from good weights and only needs "
        "nudging, while the heads start from noise",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--out", default="checkpoints/attr_heads.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  unfreeze: {args.unfreeze}  seed: {args.seed}", flush=True)

    trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
    valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

    model = Model("resnet50", unfreeze=args.unfreeze)
    model.to(device)

    # Two parameter groups so the pretrained backbone moves at its own, slower rate.
    # A single rate would either wreck the backbone or leave the heads undertrained.
    groups = [{"params": model.attribute_heads.parameters(), "lr": args.lr}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
        trainable = sum(p.numel() for p in backbone_params)
        print(f"fine-tuning {trainable:,} backbone params @ lr={args.backbone_lr}")
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    loader = balanced_loader(trainset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    train(
        model,
        loader,
        val_loader,
        optimizer,
        attr_only_loss,
        device,
        epochs=args.epochs,
        checkpoint_path=args.out,
        accum_steps=args.accum_steps,
        log_every=50,
        select_by="attr",
    )
