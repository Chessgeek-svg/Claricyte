"""Shared training machinery for the concept bottleneck model.

Both stages share the same epoch loop, evaluation, and class-balanced sampling; only
the loss, the set of parameters the optimizer holds, and a few hyperparameters differ.
That common machinery lives here so the runner scripts (scripts/train_attr_heads.py for
the attribute stage 1, scripts/train_class_head.py for the class-head stage 2) stay
thin.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from claricyte.vocab import ATTRIBUTES


def balanced_loader(dataset, batch_size, label_column="claricyte_label", num_workers=4):
    """A DataLoader whose sampler draws every class about equally often.

    Each sample is weighted by the inverse of its class frequency, so a majority
    class (e.g. eosinophil) stops dominating gradient updates.

    num_workers > 0 loads/decodes images in parallel worker processes instead of
    the main thread, so the GPU isn't left idle waiting on CPU image decoding
    between batches. pin_memory speeds up the CPU -> GPU transfer.
    """
    counts = dataset.df[label_column].value_counts()
    weights = [1.0 / counts[label] for label in dataset.df[label_column]]
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def joint_loss(attr_logits, class_logits, attr_targets, class_targets):
    """Sum of the 11 attribute cross-entropies plus the class cross-entropy."""
    attr_loss = sum(
        F.cross_entropy(attr_logits[attr], attr_targets[:, i])
        for i, attr in enumerate(ATTRIBUTES)
    )
    return attr_loss + F.cross_entropy(class_logits, class_targets)


def class_only_loss(attr_logits, class_logits, attr_targets, class_targets):
    """Class cross-entropy alone, for the sequential stage where attrs are frozen."""
    return F.cross_entropy(class_logits, class_targets)


def attr_only_loss(attr_logits, class_logits, attr_targets, class_targets):
    """The 11 attribute cross-entropies alone, for pure concept supervision.

    Stage 1 uses this so the class objective never leaks back into the attribute
    heads and bends them toward classifiability at the cost of honesty.
    """
    return sum(
        F.cross_entropy(attr_logits[attr], attr_targets[:, i])
        for i, attr in enumerate(ATTRIBUTES)
    )


def evaluate(model, loader, device):
    """Run over `loader` without training; return (per-attribute acc dict, class acc)"""
    model.eval()  # disable dropout / BN updates
    attr_correct = {attr: 0 for attr in ATTRIBUTES}
    class_correct = 0
    total = 0
    with torch.no_grad():  # no gradients needed, saves memory
        for images, attr_targets, class_targets in loader:
            images = images.to(device)
            attr_targets = attr_targets.to(device)
            class_targets = class_targets.to(device)

            attr_logits, class_logits = model(images)

            # for each attribute, the prediction is the highest-scoring value index
            for i, attr in enumerate(ATTRIBUTES):
                preds = attr_logits[attr].argmax(dim=1)
                attr_correct[attr] += (preds == attr_targets[:, i]).sum().item()

            class_preds = class_logits.argmax(dim=1)
            class_correct += (class_preds == class_targets).sum().item()
            total += class_targets.size(0)

    attr_acc = {attr: attr_correct[attr] / total for attr in ATTRIBUTES}
    return attr_acc, class_correct / total


def train(
    model,
    loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    *,
    epochs,
    checkpoint_path,
    accum_steps=1,
    log_every=0,
    select_by="class",
):
    """Train `model`, saving the best-on-val checkpoint.

    loss_fn(attr_logits, class_logits, attr_targets, class_targets) -> scalar loss,
    so the same loop serves both the attribute and class-only stages. Uses AMP; set
    accum_steps > 1 to accumulate gradients over that many batches before stepping,
    and log_every > 0 to print intra-epoch progress every N steps. select_by picks
    which validation metric decides the best checkpoint: "class" for the class-head
    stage, "attr" (mean attribute accuracy) for the attribute stage where the class
    head is untrained.
    """
    scaler = torch.amp.GradScaler("cuda")  # type: ignore
    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_score = 0.0

    for epoch in range(epochs):
        # evaluate() leaves the model in eval; flip heads back to train each epoch.
        # (Model.train keeps the frozen backbone in eval regardless.)
        model.train()
        running_loss = 0.0
        for i, (images, attr_targets, class_targets) in enumerate(loader):
            images = images.to(device)
            attr_targets = attr_targets.to(device)
            class_targets = class_targets.to(device)

            with torch.amp.autocast("cuda"):  # type: ignore
                attr_logits, class_logits = model(images)
                loss = loss_fn(attr_logits, class_logits, attr_targets, class_targets)
                # scale by 1/accum so the accumulated grads sum to one normal update
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps  # unscale for reporting
            if log_every and i % log_every == 0:
                print(
                    f"  step {i}/{len(loader)} avg_loss={running_loss / (i + 1):.4f}",
                    flush=True,
                )

        # Flush any gradients left if the epoch didn't end on an accumulation boundary.
        if len(loader) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = running_loss / len(loader)
        attr_acc, class_acc = evaluate(model, val_loader, device)
        mean_attr_acc = sum(attr_acc.values()) / len(attr_acc)
        score = mean_attr_acc if select_by == "attr" else class_acc
        print(
            f"Epoch {epoch}: train_loss={avg_loss:.4f}  "
            f"val_class_acc={class_acc:.3f}  mean_attr_acc={mean_attr_acc:.3f}"
        )
        for attr, acc in attr_acc.items():
            print(f"    {attr:28s} {acc:.3f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  saved new best ({select_by}_acc={best_score:.3f})")

    return best_score
