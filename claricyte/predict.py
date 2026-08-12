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
        concepts: the (1, 31) concept vector the class head actually reads, i.e. the
            11 attribute softmaxes concatenated in ``vocab.ATTRIBUTES`` order. Needed
            by ``contributions`` because a nonlinear head's attribution depends on
            every concept, not just the argmax ones.
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

    # Rebuild the concept vector the class head saw, in the same ATTRIBUTES order
    # forward() uses, so index i here means the same thing it does to the head.
    concepts = torch.cat(
        [F.softmax(attr_logits[attr], dim=1) for attr in ATTRIBUTES], dim=1
    )

    # Full class distribution as {class_name: probability}. The predicted label is
    # just its argmax, so callers derive it rather than us returning it separately.
    class_probs = F.softmax(class_logits, dim=1).squeeze(0)  # (1, 6) -> (6,)
    class_dist = {cls: prob for cls, prob in zip(vocab.CLASSES, class_probs.tolist())}

    return result, class_dist, concepts


def contributions(model, result, concepts, label):
    """Signed contribution of each reported attribute value toward `label`.

    Uses input x gradient on the class head: for the concept dimension each
    attribute actually reported (its argmax value),

        contribution = d(class_logit[label]) / d(concept_i)  *  concept_i

    which says how strongly that stated feature pushed toward (positive) or against
    (negative) the given label. Used to rank which features to mention and to flag
    contradictions.

    This works for ANY class-head architecture. For a linear head it is exactly the
    old closed form: d/d(concept_i) of (sum_j W[c,j]*concept_j + b) is W[c,i], so the
    product is W[label, i] * confidence. For the MLP head the gradient depends on the
    whole concept vector (that is the point of a nonlinear head), which is why the
    full vector from ``predict`` is required rather than just the argmax values.

    Being a first-order local measure, it is exact for the linear head and a
    linearization for the MLP; contributions no longer sum exactly to the logit in
    the nonlinear case.
    """
    label_idx = vocab.CLASS_TO_INDEX[label]

    # Gradient of the target class logit w.r.t. the concept vector. autograd.grad
    # (rather than .backward()) keeps this side-effect free: no .grad left on the
    # model's parameters.
    concepts = concepts.detach().clone().requires_grad_(True)
    class_logits = model.class_head(concepts)  # (1, num_classes)
    (grad,) = torch.autograd.grad(class_logits[0, label_idx], concepts)
    attribution = (grad[0] * concepts[0]).detach()  # (31,)

    # Where each attribute's block starts in the concatenated 31-dim concept vector.
    offsets = {}
    running = 0
    for attr in ATTRIBUTES:
        offsets[attr] = running
        running += vocab.num_classes(attr)

    scores = {}
    for attr, (value, _confidence) in result.items():
        dim = offsets[attr] + vocab.encode(attr, value)
        scores[attr] = float(attribution[dim].item())
    return scores
