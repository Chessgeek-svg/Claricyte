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
    parser.add_argument("--out", default="checkpoints/best_model_seq.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  head: {args.head}  seed: {args.seed}", flush=True)

    model = Model("resnet50", head=args.head, hidden_dim=args.hidden_dim)

    # Take the backbone and attribute heads from stage 1. Its class head is dropped:
    # stage 1 never trained it, and its shape won't even match when --head mlp.
    state = torch.load(
        "checkpoints/best_model.pt", map_location=device, weights_only=True
    )
    state = {k: v for k, v in state.items() if not k.startswith("class_head.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
    assert all(k.startswith("class_head.") for k in missing), (
        f"stage-1 checkpoint is missing more than the class head: {missing}"
    )
    model.to(device)

    # Freeze the attribute heads so only the class head learns; reset the class head
    # so we start clean rather than from whatever came in on the checkpoint.
    for param in model.attribute_heads.parameters():
        param.requires_grad = False
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
