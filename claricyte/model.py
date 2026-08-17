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
    def __init__(
        self, model_name, head="linear", hidden_dim=32, unfreeze=None, pretrained=True
    ) -> None:
        super().__init__()
        # pretrained=False builds the architecture with random weights and skips the
        # ~100MB ImageNet download. Only safe when every backbone weight is about to
        # be overwritten by a checkpoint, which from_checkpoint knows and callers
        # generally do not, hence the default stays True.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.feat_dim = self.backbone.num_features

        # unfreeze=None keeps the backbone at its pretrained weights and trains only
        # the heads. Anything else names a submodule to fine-tune ("layer4" for the
        # last residual block, "all" for the whole backbone).
        self.unfreeze = unfreeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze == "all":
            for param in self.backbone.parameters():
                param.requires_grad = True
        elif unfreeze:
            # Comma separated, so a cumulative sweep ("layer4", then "layer3,layer4",
            # and so on) still records as a single string in the checkpoint.
            for name in (part.strip() for part in unfreeze.split(",")):
                if not hasattr(self.backbone, name):
                    raise ValueError(
                        f"{model_name} has no submodule {name!r} to unfreeze"
                    )
                for param in getattr(self.backbone, name).parameters():
                    param.requires_grad = True

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
        """Save the trained weights plus the recipe for rebuilding this model.

        When the backbone is frozen, its weights and BatchNorm stats stay
        bit-identical to what timm downloads, so storing them turns a 260KB file
        into a 94MB one carrying no information. In that case only the attribute
        and class heads are written, and the backbone is recorded by tag so
        from_checkpoint can fetch the same weights again.

        When any of the backbone is being fine-tuned it is no longer recoverable
        from timm and MUST be stored. Dropping it would produce a checkpoint that
        loads without error and silently restores ImageNet weights, discarding the
        fine-tuning with nothing to indicate anything went wrong.

        `metadata` is free-form extra fields to record alongside (e.g.
        val_acc=0.774). Anything passed must be a plain scalar, string, list or
        tensor, since checkpoints are read back with weights_only=True.
        """
        # Ask the parameters, so this stays correct even if a
        # caller froze or thawed something after construction.
        finetuned = any(p.requires_grad for p in self.backbone.parameters())
        weights = {}
        for key, value in self.state_dict().items():
            if key.startswith("backbone."):
                if not finetuned:
                    continue
                # A fine-tuned backbone has to be stored, but half precision halves
                # 95MB to 47MB at no measurable cost: round-tripping every weight
                # through fp16 moved val accuracy 0.9305 -> 0.9318, which is two
                # cells out of 1,568 and therefore noise. load_state_dict widens
                # these back to fp32 on the way in, so inference is unaffected.
                # Heads stay fp32; they are a few hundred KB and their decisions
                # are the marginal ones.
                if value.is_floating_point():
                    value = value.half()
            weights[key] = value
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "backbone": self.backbone_tag,
                "head": self.head,
                "hidden_dim": self.hidden_dim,
                "unfreeze": self.unfreeze,
                "has_backbone": finetuned,
                "heads": weights,
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

        `unfreeze` is taken from the file rather than overridable: it describes
        which weights the file contains, not a choice the caller gets to make.
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
        # Checkpoints written before fine-tuning existed carry neither field, and
        # the defaults reconstruct their meaning exactly: those runs always froze
        # the whole backbone and therefore always omitted it.
        unfreeze = checkpoint.get("unfreeze")
        has_backbone = checkpoint.get("has_backbone", False)

        model = cls(
            checkpoint["backbone"],
            head=head,
            hidden_dim=hidden_dim,
            unfreeze=unfreeze,
            # A fine-tuned checkpoint carries every backbone weight, so fetching the
            # ImageNet ones first only to overwrite them costs a ~100MB download for
            # nothing. That download is most of the hosted demo's cold start. When
            # the checkpoint has no backbone, the pretrained weights ARE the
            # backbone and must be fetched.
            pretrained=not has_backbone,
        )

        weights = checkpoint["heads"]
        replaced = (head, hidden_dim) != (checkpoint["head"], checkpoint["hidden_dim"])
        if replaced:
            weights = {
                key: value
                for key, value in weights.items()
                if not key.startswith("class_head.")
            }

        # strict=False because backbone keys are absent whenever the backbone was
        # frozen. That makes the load permissive, so verify explicitly that the ONLY
        # things missing are the ones we meant to leave out. Without this, a
        # checkpoint missing an attribute head would load silently with random
        # weights, and a fine-tuned checkpoint whose backbone failed to save would
        # quietly fall back to ImageNet.
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"{path}: unexpected keys in checkpoint: {unexpected}")
        expected_missing = () if has_backbone else ("backbone.",)
        if replaced:
            expected_missing += ("class_head.",)
        # str.startswith(()) is False, so an empty tuple correctly makes every
        # missing key unaccounted for.
        unaccounted = [k for k in missing if not k.startswith(expected_missing)]
        if unaccounted:
            raise ValueError(f"{path}: checkpoint is missing weights for {unaccounted}")

        return model.to(device)

    def train(self, mode: bool = True):
        # nn.Module.train() recurses into EVERY submodule, which would flip the
        # backbone back into train mode and let its BatchNorm resume updating running
        # stats. Force the backbone back to eval so those stats stay frozen.
        #
        # This holds even while fine-tuning, and deliberately. requires_grad controls
        # whether the WEIGHTS learn; train/eval mode controls whether the BatchNorm
        # running statistics drift. Gradients still flow through a BN layer in eval
        # mode, so the convolutions fine-tune normally, they just normalize against
        # the stable ImageNet statistics instead of noisy per-batch ones. Freezing BN
        # is standard practice when fine-tuning on a dataset this small.
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
