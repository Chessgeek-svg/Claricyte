import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from claricyte import vocab

# Bumped whenever the on-disk checkpoint layout changes incompatibly, so an old
# file fails with a clear message instead of a confusing KeyError.
CHECKPOINT_FORMAT = 1


class Model(nn.Module):
    def __init__(self, model_name, head="linear", hidden_dim=32) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feat_dim = self.backbone.num_features

        # Record how this model was built so save_heads can write it into the
        # checkpoint and from_checkpoint can rebuild it without guessing.
        # Resolve the bare name to its fully qualified timm tag: "resnet50" is a
        # DEFAULT that timm is free to repoint at different weights in a future
        # release, whereas "resnet50.a1_in1k" names one specific set. Since the
        # checkpoint no longer carries the backbone, that distinction is the
        # difference between reloading the model that was trained and silently
        # reloading a different one.
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        tag = cfg.get("tag")
        self.backbone_tag = f"{cfg['architecture']}.{tag}" if tag else model_name
        self.head = head
        self.hidden_dim = hidden_dim
        self.attribute_heads = nn.ModuleDict(
            {
                attr: nn.Linear(self.feat_dim, vocab.num_classes(attr))  # type: ignore
                for attr in vocab.ATTRIBUTES
            }
        )

        # The concept vector: 11 attribute softmaxes concatenated (31 dims today).
        concept_dim = sum(vocab.num_classes(attr) for attr in vocab.ATTRIBUTES)

        if head == "linear":
            # One fixed weight per concept dimension. Cannot represent interactions
            # between attributes: a concept's contribution is the same regardless of
            # what the other concepts say.
            self.class_head = nn.Linear(concept_dim, len(vocab.CLASSES))
        elif head == "mlp":
            # The hidden layer + ReLU lets a unit fire only on a COMBINATION of
            # concepts (e.g. band-shaped nucleus AND small cell), which a single
            # linear layer provably cannot express. Two stacked Linears with no
            # nonlinearity between them would collapse back to one Linear, so the
            # ReLU is what actually buys the extra expressiveness.
            self.class_head = nn.Sequential(
                nn.Linear(concept_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(vocab.CLASSES)),
            )
        else:
            raise ValueError(f"unknown head {head!r}, expected 'linear' or 'mlp'")

    def reset_class_head(self):
        """Re-initialize the class head, whichever architecture it is.

        nn.Sequential has no reset_parameters(), so walk its submodules. Stage 2
        calls this so training starts from a clean head rather than whatever
        untrained weights came in on the stage-1 checkpoint.
        """
        for module in self.class_head.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def save_heads(self, path, **metadata) -> None:
        """Save the trained heads plus the recipe for rebuilding this model.

        The backbone is frozen and Model.train() keeps it in eval, so its weights
        and BatchNorm stats are bit-identical to what timm downloads. Saving them
        turns a 256KB file into a 94MB one for no information. We store only the
        attribute and class heads, and record the backbone by tag so
        from_checkpoint can fetch the same weights again.

        `metadata` is free-form extra fields to record alongside (e.g.
        val_acc=0.774). Anything passed must be a plain scalar, string, list or
        tensor, since checkpoints are read back with weights_only=True.
        """
        heads = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "backbone": self.backbone_tag,
                "head": self.head,
                "hidden_dim": self.hidden_dim,
                "heads": heads,
                **metadata,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, device="cpu", head=None, hidden_dim=None):
        """Rebuild a model from a checkpoint written by save_heads.

        The checkpoint names its own architecture, so callers no longer have to
        guess it. That guess is what previously made the MLP checkpoint
        unloadable by app.py, which assumed the default linear head.

        `head` / `hidden_dim` override what the file recorded, for stage 2, which
        wants stage 1's attribute heads under a DIFFERENT class head. When they
        differ from the file, the stored class head describes an architecture we
        are not building, so it is dropped and the new one is left freshly
        initialized.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        found = checkpoint.get("format")
        if found != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path}: checkpoint format {found!r}, expected {CHECKPOINT_FORMAT}. "
                "Convert it with scripts/convert_checkpoint.py."
            )

        head = head if head is not None else checkpoint["head"]
        hidden_dim = hidden_dim if hidden_dim is not None else checkpoint["hidden_dim"]
        model = cls(checkpoint["backbone"], head=head, hidden_dim=hidden_dim)

        weights = checkpoint["heads"]
        replaced = (head, hidden_dim) != (checkpoint["head"], checkpoint["hidden_dim"])
        if replaced:
            weights = {
                key: value
                for key, value in weights.items()
                if not key.startswith("class_head.")
            }

        # strict=False because the backbone keys are deliberately absent. That
        # makes the load permissive, so verify explicitly that the ONLY things
        # missing are the ones we meant to leave out. Without this, a checkpoint
        # missing an attribute head would load silently with random weights.
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"{path}: unexpected keys in checkpoint: {unexpected}")
        expected_missing = ("backbone.", "class_head.") if replaced else ("backbone.",)
        unaccounted = [k for k in missing if not k.startswith(expected_missing)]
        if unaccounted:
            raise ValueError(f"{path}: checkpoint is missing weights for {unaccounted}")

        return model.to(device)

    def train(self, mode: bool = True):
        # nn.Module.train() recurses into EVERY submodule, which would flip the frozen
        # backbone back into train mode and let its BatchNorm resume updating running
        # stats. Force the backbone back to eval so those stats stay frozen.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        # x: (B, 3, 224, 224) batch of normalized image tensors.
        # Frozen backbone -> one pooled feature vector per image: (B, feat_dim).
        feats = self.backbone(x)

        # Each attribute head maps feats -> raw scores for THAT attribute's values.
        # attr_logits[attr] has shape (B, num_classes(attr)); e.g. cell_size -> (B, 2).
        attr_logits = {attr: head(feats) for attr, head in self.attribute_heads.items()}

        # Soft bottleneck: convert each head's logits to a probability vector. Iterate
        # vocab.ATTRIBUTES so the concat order matches class_head's weights.
        probs = [F.softmax(attr_logits[attr], dim=1) for attr in vocab.ATTRIBUTES]

        # Glue the 11 prob vectors into one (B, 31) concept vector,
        # then map those concepts -> 6 class scores.
        class_logits = self.class_head(torch.cat(probs, dim=1))

        # attr_logits: 11 x (B, n_values); class_logits: (B, 6).
        return attr_logits, class_logits
