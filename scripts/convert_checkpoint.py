"""Convert an old full-state_dict checkpoint to the heads-only format.

Checkpoints used to be `torch.save(model.state_dict())`, which stored the frozen
backbone (23.5M params, 94MB) alongside the 64k trained head params, and recorded
nothing about which class head architecture produced them. The new format stores
only the heads plus the recipe for rebuilding the model. This script migrates the
old files so they do not have to be retrained.

Before discarding the backbone weights it verifies they are bit-identical to what
timm downloads for the named tag, so nothing unrecoverable is thrown away.

    python scripts/convert_checkpoint.py checkpoints/best_model_mlp_cached.pt \
        checkpoints/mlp.pt --val-acc 0.774
"""

import argparse

import timm
import torch

from claricyte.model import CHECKPOINT_FORMAT

BACKBONE_PREFIX = "backbone."


def infer_class_head(state: dict) -> tuple[str, int]:
    """Recover the class head architecture from the shape of its keys.

    nn.Linear registers "class_head.weight"; nn.Sequential names children by
    position, so the MLP registers "class_head.0.weight" (the ReLU at index 1
    has no parameters). For the MLP the hidden size is the first Linear's output
    dimension, which is the row count of its weight matrix.
    """
    if "class_head.weight" in state:
        return "linear", 32  # hidden_dim is unused for a linear head
    if "class_head.0.weight" in state:
        return "mlp", state["class_head.0.weight"].shape[0]
    raise ValueError("checkpoint has no recognizable class head")


def verify_backbone(state: dict, backbone: str) -> None:
    """Confirm the stored backbone matches a fresh download, or raise."""
    fresh = timm.create_model(backbone, pretrained=True, num_classes=0).state_dict()
    stored = {
        key[len(BACKBONE_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(BACKBONE_PREFIX)
    }
    if not stored:
        raise ValueError("checkpoint has no backbone weights; already converted?")
    if set(stored) != set(fresh):
        raise ValueError(f"backbone keys do not match {backbone}")
    differing = [k for k in stored if not torch.equal(stored[k], fresh[k])]
    if differing:
        raise ValueError(
            f"{len(differing)} backbone tensors differ from {backbone}; these weights "
            f"are NOT recoverable and must not be dropped (first: {differing[:3]})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="old full-state_dict checkpoint")
    parser.add_argument("dst", help="path to write the heads-only checkpoint")
    parser.add_argument("--backbone", default="resnet50.a1_in1k")
    parser.add_argument(
        "--val-acc", type=float, default=None, help="recorded alongside, for provenance"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the bit-identity check against a fresh timm download",
    )
    args = parser.parse_args()

    state = torch.load(args.src, map_location="cpu", weights_only=True)
    if "heads" in state:
        raise SystemExit(f"{args.src} is already in the heads-only format")

    if not args.skip_verify:
        print(f"verifying backbone against {args.backbone}...", flush=True)
        verify_backbone(state, args.backbone)
        print("backbone is bit-identical, safe to drop", flush=True)

    head, hidden_dim = infer_class_head(state)
    heads = {k: v for k, v in state.items() if not k.startswith(BACKBONE_PREFIX)}

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "backbone": args.backbone,
        "head": head,
        "hidden_dim": hidden_dim,
        "heads": heads,
    }
    if args.val_acc is not None:
        checkpoint["val_acc"] = args.val_acc

    torch.save(checkpoint, args.dst)
    params = sum(v.numel() for v in heads.values())
    print(f"wrote {args.dst}: head={head} hidden_dim={hidden_dim} params={params}")


if __name__ == "__main__":
    main()
