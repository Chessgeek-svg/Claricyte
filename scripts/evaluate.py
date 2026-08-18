"""Evaluate a trained checkpoint on one split.

The only place in the repo that reads split="test". Training and selection use
val exclusively, so keeping the test read behind a separate script makes "do not
let test inform decisions" a property of the structure rather than of memory.

Reports three things that answer different questions:

  soft       class accuracy from the full concept vector, i.e. what the model
             actually does. This is the headline number.
  hardened   the same, but each attribute forced to its argmax before the class
             head reads it. This is what the STATED attributes are worth, so the
             gap between the two is the value of the model's uncertainty rather
             than evidence of anything leaking.
  collapsed  band and segmented merged into one neutrophil class, which is the
             5-class task the WBCAtt paper reports on. Ours is a harder 6-class
             problem, so this is the only figure directly comparable to theirs.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from claricyte.data import MorphologyDataset
from claricyte.model import Model
from claricyte.vocab import ATTRIBUTES, CLASSES

ATTR_PATH, METADATA_PATH = "metadata/attributes.csv", "metadata/metadata.csv"

# The two classes the WBCAtt paper does not separate. Merging them reproduces
# their "neutrophil" class so the numbers can sit side by side.
NEUTROPHILS = ("Band Neutrophil", "Segmented Neutrophil")


def evaluate(model, loader, device):
    """Return (attr_correct, class_pred_soft, class_pred_hard, class_target)."""
    attr_correct = {attr: 0 for attr in ATTRIBUTES}
    soft, hard, targets = [], [], []
    total = 0

    with torch.no_grad():
        for images, attr_targets, class_targets in loader:
            images = images.to(device)
            attr_logits, _ = model(images)

            probs = [F.softmax(attr_logits[attr], dim=1) for attr in ATTRIBUTES]
            for i, attr in enumerate(ATTRIBUTES):
                predicted = probs[i].argmax(dim=1).cpu()
                attr_correct[attr] += int((predicted == attr_targets[:, i]).sum())

            # Hardened: every attribute forced to a single value, discarding the
            # confidences, which is what a strict discrete bottleneck would see.
            one_hot = torch.cat(
                [F.one_hot(p.argmax(1), p.shape[1]).float() for p in probs], dim=1
            )
            soft.append(model.class_head(torch.cat(probs, dim=1)).argmax(1).cpu())
            hard.append(model.class_head(one_hot).argmax(1).cpu())
            targets.append(class_targets)
            total += images.shape[0]

    return (
        {attr: correct / total for attr, correct in attr_correct.items()},
        torch.cat(soft),
        torch.cat(hard),
        torch.cat(targets),
    )


def confusion(predicted, target):
    """Rows are the true class, columns the predicted one."""
    matrix = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int32)
    for t, p in zip(target.tolist(), predicted.tolist()):
        matrix[t, p] += 1
    return matrix


def collapse_neutrophils(labels):
    """Map Band -> Segmented so band/segmented confusions stop counting as errors."""
    band, seg = (CLASSES.index(name) for name in NEUTROPHILS)
    return torch.where(labels == band, torch.full_like(labels, seg), labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="test is a one-shot read: pick the model on val FIRST (default: val)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.split == "test":
        print(
            "\n  Reading the TEST split. This is only honest if the checkpoint was\n"
            "  chosen on val beforehand. Comparing several checkpoints here and\n"
            "  reporting the best makes test a selection metric.\n"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    dataset = MorphologyDataset(ATTR_PATH, METADATA_PATH, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    attr_acc, soft, hard, target = evaluate(model, loader, device)

    print(f"checkpoint: {args.checkpoint}")
    print(f"split: {args.split}  ({len(dataset)} cells)\n")

    print("attribute accuracy")
    for attr in ATTRIBUTES:
        print(f"  {attr:<28} {attr_acc[attr]:.3f}")
    print(f"  {'mean':<28} {sum(attr_acc.values()) / len(attr_acc):.3f}\n")

    soft_acc = (soft == target).float().mean().item()
    hard_acc = (hard == target).float().mean().item()
    print(f"class accuracy (soft)      {soft_acc:.3f}")
    print(f"class accuracy (hardened)  {hard_acc:.3f}")

    collapsed = (
        (collapse_neutrophils(soft) == collapse_neutrophils(target))
        .float()
        .mean()
        .item()
    )
    print(f"class accuracy (5-class)   {collapsed:.3f}   band+segmented merged\n")

    matrix = confusion(soft, target)
    width = max(len(name) for name in CLASSES)
    print("confusion matrix (rows = true, columns = predicted)")
    print(" " * (width + 2) + "".join(f"{name[:6]:>8}" for name in CLASSES))
    for i, name in enumerate(CLASSES):
        counts = "".join(f"{int(c):>8}" for c in matrix[i])
        recall = matrix[i, i] / matrix[i].sum() if matrix[i].sum() else 0.0
        print(f"  {name:<{width}}{counts}    recall {recall:.3f}")

    errors = int((soft != target).sum())
    band, seg = (CLASSES.index(name) for name in NEUTROPHILS)
    swapped = int(matrix[band, seg] + matrix[seg, band])
    share = swapped / errors if errors else 0.0
    print(f"\nerrors: {errors}  of which band<->segmented: {swapped} ({share:.1%})")


if __name__ == "__main__":
    main()
