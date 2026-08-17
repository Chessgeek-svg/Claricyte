"""Sequential CBM training, stage 2: train only the class head.

Stage 1 (train_attr_heads.py) already produced good attribute heads. Here we
FREEZE everything except the class head and retrain just that head on a
class-balanced loader. The class head then learns to separate classes from stable,
accurate concept vectors instead of chasing a moving target and riding the majority
class.

Because everything upstream of the class head is frozen, the concept vectors are
fixed: we compute them ONCE up front and train on the cache, instead of re-running
a ResNet50 over the whole dataset every epoch to feed a ~1k-parameter head.

--head linear is one weight per concept. --head mlp adds a hidden layer + ReLU so
the head can key on combinations of concepts rather than just linearly weighting them.
"""

import argparse

import torch

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.training import (
    balanced_concept_loader,
    precompute_concepts,
    train_class_head_cached,
)

# Windows spawns fresh processes for DataLoader workers (num_workers>0), and each
# one re-imports this module. Without this guard, every worker would re-run all
# the training code below instead of just importing the functions it needs.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # How many independently augmented copies of the train set to cache. >1 keeps
    # some augmentation variety; concepts are 31 floats each, so this is cheap.
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--attr-heads", default="checkpoints/attr_heads.pt")
    parser.add_argument("--out", default="checkpoints/class_head.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  head: {args.head}  seed: {args.seed}", flush=True)

    # Stage 1's attribute heads, under whichever class head --head asked for. The
    # architecture comes from the flags rather than from the file, so the stored
    # class head is dropped whenever the two disagree. Nothing is lost: stage 1's
    # optimizer only ever held the attribute heads, so that head was never trained.
    model = Model.from_checkpoint(
        args.attr_heads,
        device=device,
        head=args.head,
        hidden_dim=args.hidden_dim,
    )

    # Only the class head learns. The optimizer below already holds nothing else, so
    # this changes no behaviour today. It is here to defend the no-leakage property
    # that justifies training in two stages at all, in case that optimizer is ever
    # widened to model.parameters().
    for param in model.attribute_heads.parameters():
        param.requires_grad = False

    # Always start from a fresh class head. A no-op when from_checkpoint dropped the
    # stored one, and necessary when it did not: pointing this script at a stage-2
    # checkpoint would otherwise warm-start from the previous run rather than
    # training clean, which would quietly invalidate any seed-to-seed comparison.
    model.reset_class_head()

    trainset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="train")
    valset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split="val")

    # One pass of the frozen backbone + attribute heads, then train on the cache.
    # The val transform is deterministic, so a single view is all there is.
    print(f"caching concepts ({args.views} train views)...", flush=True)
    train_concepts, train_targets = precompute_concepts(
        model, trainset, device, views=args.views
    )
    val_concepts, val_targets = precompute_concepts(model, valset, device, views=1)
    print(
        f"cached train {tuple(train_concepts.shape)}  val {tuple(val_concepts.shape)}",
        flush=True,
    )

    loader = balanced_concept_loader(train_concepts, train_targets, batch_size=32)

    # Only the class head trains. A small weight decay (not the wd=0.0 used to
    # isolate the original flat-logit diagnosis) plus label smoothing to discourage
    # the head from overfitting to full certainty on this small, frozen-feature
    # dataset.
    optimizer = torch.optim.AdamW(
        model.class_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_class_head_cached(
        model,
        loader,
        val_concepts,
        val_targets,
        optimizer,
        device,
        epochs=args.epochs,
        checkpoint_path=args.out,
        label_smoothing=args.label_smoothing,
    )
