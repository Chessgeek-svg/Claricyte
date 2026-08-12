import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from claricyte import vocab


class Model(nn.Module):
    def __init__(self, model_name, head="linear", hidden_dim=32) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feat_dim = self.backbone.num_features
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
