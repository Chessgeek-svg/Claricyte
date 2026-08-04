import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from claricyte import vocab


class Model(nn.Module):
    def __init__(self, model_name) -> None:
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
        self.class_head = nn.Linear(
            in_features=sum(vocab.num_classes(attr) for attr in vocab.ATTRIBUTES),
            out_features=len(vocab.CLASSES),
        )

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
