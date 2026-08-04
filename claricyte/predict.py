import torch
import torch.nn.functional as F

from claricyte import vocab
from claricyte.vocab import ATTRIBUTES, decode


def predict(model, image_tensor):
    """Run one image through the CBM and decode its outputs.

    Args:
        model: a trained ``Model`` already in eval mode.
        image_tensor: a single preprocessed image, shape (3, H, W) — i.e. one
            item straight out of ``MorphologyDataset`` (no batch dimension).

    Returns:
        result: dict mapping each attribute name to a
            ``(value_string, probability)`` tuple.
        class_dist: dict mapping each class name to its predicted probability. The
            predicted label is ``max(class_dist, key=class_dist.get)``.
    """
    device = next(model.parameters()).device
    batch = image_tensor.unsqueeze(0).to(device)  # (3, H, W) -> (1, 3, H, W)

    with torch.no_grad():  # inference only: no gradients, saves memory
        # attr_logits: 11 x (1, n_values); class_logits: (1, 6)
        attr_logits, class_logits = model(batch)

    result = {}
    for attr in ATTRIBUTES:
        # probs: (1, n_values); sum(n_values) = 1
        probs = F.softmax(attr_logits[attr], dim=1)  
        top = probs.max(dim=1)  # tuple: (values, indices)
        # result: dict[attr] = (value string, probability), e.g.
        # {'cell_size': ('small', 0.99), 'cell_shape': ('round', 0.95), ...}
        result[attr] = (decode(attr, int(top.indices.item())), float(top.values.item()))

    # Full class distribution as {class_name: probability}. The predicted label is
    # just its argmax, so callers derive it rather than us returning it separately.
    class_probs = F.softmax(class_logits, dim=1).squeeze(0)  # (1, 6) -> (6,)
    class_dist = {cls: prob for cls, prob in zip(vocab.CLASSES, class_probs.tolist())}

    return result, class_dist


def contributions(model, result, label):
    """Signed contribution of each reported attribute value toward `label`.

    The soft bottleneck computes ``class_logit[c] = sum_i W[c, i] * concept_prob_i``
    (+ bias), where the concept vector is the 11 attribute softmaxes concatenated in
    ``vocab.ATTRIBUTES`` order. For the value we actually report on each attribute
    (its argmax), this returns ``W[label, selected_dim] * confidence``, how strongly
    that stated feature pushed toward (positive) or against (negative) the given
    label. Used to rank which features to mention and to flag contradictions.

    Because the reported value string plus ``vocab`` recover ``selected_dim``, this
    needs nothing from ``predict`` beyond its result dict.
    """
    label_idx = vocab.CLASS_TO_INDEX[label]
    weight = model.class_head.weight.detach()  # (num_classes, 31)

    # Where each attribute's block starts in the concatenated 31-dim concept vector.
    offsets = {}
    running = 0
    for attr in ATTRIBUTES:
        offsets[attr] = running
        running += vocab.num_classes(attr)

    scores = {}
    for attr, (value, confidence) in result.items():
        dim = offsets[attr] + vocab.encode(attr, value)
        scores[attr] = float(weight[label_idx, dim].item()) * confidence
    return scores
