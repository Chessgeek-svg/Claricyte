"""Tests for model wiring and the checkpoint format.

Every model here is built with pretrained=False on resnet18, so the suite needs
no network and no dataset. That is deliberate: these tests have to run on a
clean CI checkout.

The checkpoint tests exist because of a real failure. Checkpoints used to record
nothing about the architecture that produced them, so app.py built the default
linear head, met an MLP checkpoint's class_head.0/class_head.2 keys, and raised.
Nothing caught it until the demo would not start.
"""

import pytest
import torch

from claricyte import vocab
from claricyte.explain import explain
from claricyte.model import CHECKPOINT_FORMAT, Model
from claricyte.predict import contributions, predict

BACKBONE = "resnet18"
CONCEPT_DIM = sum(vocab.num_classes(a) for a in vocab.ATTRIBUTES)


def build(**kwargs):
    kwargs.setdefault("pretrained", False)
    return Model(BACKBONE, **kwargs)


# ------------------------------------------------------------- wiring


def test_concept_dim_is_the_sum_of_the_attribute_vocabularies():
    assert CONCEPT_DIM == 31


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_forward_shapes(head):
    model = build(head=head).eval()
    attr_logits, class_logits = model(torch.randn(2, 3, 224, 224))

    assert set(attr_logits) == set(vocab.ATTRIBUTES)
    for attr, logits in attr_logits.items():
        assert logits.shape == (2, vocab.num_classes(attr))
    assert class_logits.shape == (2, len(vocab.CLASSES))


def test_each_attribute_block_is_a_probability_distribution():
    """The class head reads 31 dims but only 20 degrees of freedom: each of the
    11 softmax blocks sums to 1. If a block stopped summing to 1 the bottleneck
    would no longer be a set of concept probabilities."""
    model = build().eval()
    attr_logits, _ = model(torch.randn(4, 3, 224, 224))
    for attr in vocab.ATTRIBUTES:
        probs = torch.softmax(attr_logits[attr], dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)


@pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
def test_heads_adapt_to_the_backbone_feature_width(backbone):
    """Everything else here uses resnet18 because it is half the cost and none
    of it is architecture-specific. This one pays for the real backbone, so the
    suite covers what actually ships: resnet18 pools to 512 and resnet50 to
    2048, and both have to reach the heads without anything hardcoding a width.
    It also confirms the layer4 that --unfreeze names exists on both."""
    model = Model(backbone, pretrained=False, unfreeze="layer4").eval()

    assert model.attribute_heads["cell_size"].in_features == model.feat_dim
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())

    attr_logits, class_logits = model(torch.randn(1, 3, 224, 224))
    assert class_logits.shape == (1, len(vocab.CLASSES))
    assert attr_logits["cell_size"].shape == (1, vocab.num_classes("cell_size"))


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError, match="unknown head"):
        build(head="transformer")


def test_unknown_unfreeze_target_is_rejected():
    """Silently ignoring a typo here would produce a fully frozen backbone and a
    run that looks like it fine-tuned but did not."""
    with pytest.raises(ValueError, match="no submodule"):
        build(unfreeze="layer9")


def test_unfreeze_thaws_only_the_named_blocks():
    model = build(unfreeze="layer4")
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer1.parameters())


def test_unfreeze_accepts_a_comma_separated_list():
    model = build(unfreeze="layer3,layer4")
    assert all(p.requires_grad for p in model.backbone.layer3.parameters())
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())
    assert not any(p.requires_grad for p in model.backbone.layer2.parameters())


def test_unfreeze_all_thaws_everything():
    model = build(unfreeze="all")
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_backbone_stays_in_eval_mode_when_training():
    """requires_grad controls whether weights learn; train/eval controls whether
    BatchNorm running stats drift. Fine-tuning wants the first, not the second."""
    model = build(unfreeze="layer4")
    model.train()
    assert model.training
    assert not model.backbone.training


def test_reset_class_head_reinitializes_both_mlp_layers():
    model = build(head="mlp")
    before = [w.clone() for w in model.class_head.state_dict().values()]
    model.reset_class_head()
    after = list(model.class_head.state_dict().values())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


# -------------------------------------------------------- checkpoints


def test_frozen_backbone_is_not_written_to_disk(tmp_path):
    """A frozen backbone is bit-identical to what timm downloads, so storing it
    turns a 260KB file into a 94MB one carrying no information."""
    path = tmp_path / "heads.pt"
    build().save_heads(str(path))

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["has_backbone"] is False
    assert not any(k.startswith("backbone.") for k in checkpoint["heads"])
    assert any(k.startswith("attribute_heads.") for k in checkpoint["heads"])


def test_finetuned_backbone_is_written_in_half_precision(tmp_path):
    """It cannot be refetched from timm once fine-tuned, so it must be stored.
    fp16 halves the file at no measurable cost to accuracy."""
    path = tmp_path / "heads.pt"
    build(unfreeze="layer4").save_heads(str(path))

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["has_backbone"] is True
    backbone = {
        k: v for k, v in checkpoint["heads"].items() if k.startswith("backbone.")
    }
    assert backbone
    assert all(
        v.dtype == torch.float16 for v in backbone.values() if v.is_floating_point()
    )
    # Heads stay fp32: they are a few hundred KB and make the marginal calls.
    assert checkpoint["heads"]["class_head.weight"].dtype == torch.float32


def test_checkpoint_records_the_recipe(tmp_path):
    path = tmp_path / "heads.pt"
    build(head="mlp", hidden_dim=16, unfreeze="layer4").save_heads(str(path))

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["format"] == CHECKPOINT_FORMAT
    assert checkpoint["head"] == "mlp"
    assert checkpoint["hidden_dim"] == 16
    assert checkpoint["unfreeze"] == "layer4"
    # The fully qualified timm tag, not the bare name: "resnet18" is a default
    # timm may repoint at different weights in a future release.
    assert checkpoint["backbone"].startswith("resnet18.")


def test_metadata_is_recorded_alongside(tmp_path):
    path = tmp_path / "heads.pt"
    build().save_heads(str(path), val_acc=0.93, selected_by="attr")
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["val_acc"] == 0.93
    assert checkpoint["selected_by"] == "attr"


def test_save_creates_missing_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "heads.pt"
    build().save_heads(str(path))
    assert path.exists()


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_round_trip_restores_the_same_model(tmp_path, head):
    """The regression test for the app.py failure: a checkpoint must reload into
    the architecture that wrote it, without the caller having to know which."""
    path = tmp_path / "heads.pt"
    original = build(head=head, unfreeze="layer4")
    original.save_heads(str(path))

    restored = Model.from_checkpoint(str(path))
    assert restored.head == head

    original_weights = original.state_dict()
    for key, value in restored.state_dict().items():
        if key.startswith("backbone."):
            continue  # stored as fp16, so not bit-identical
        assert torch.equal(value, original_weights[key]), key


def test_round_trip_preserves_the_backbone_within_fp16_tolerance(tmp_path):
    path = tmp_path / "heads.pt"
    original = build(unfreeze="layer4")
    original.save_heads(str(path))
    restored = Model.from_checkpoint(str(path))

    originals = original.state_dict()
    for key, value in restored.state_dict().items():
        if key.startswith("backbone.") and value.is_floating_point():
            assert torch.allclose(value, originals[key], atol=1e-2), key


def test_head_override_drops_the_stored_class_head(tmp_path):
    """Stage 2 wants stage 1's attribute heads under a different class head. The
    stored class head describes an architecture we are not building."""
    path = tmp_path / "heads.pt"
    stage1 = build(head="linear", unfreeze="layer4")
    stage1.save_heads(str(path))

    stage2 = Model.from_checkpoint(str(path), head="mlp", hidden_dim=16)
    assert stage2.head == "mlp"
    # Attribute heads carried over verbatim; class head is freshly initialized.
    weights = stage1.state_dict()
    for key, value in stage2.state_dict().items():
        if key.startswith("attribute_heads."):
            assert torch.equal(value, weights[key]), key


def test_wrong_format_version_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": 0, "heads": {}}, path)
    with pytest.raises(ValueError, match="checkpoint format"):
        Model.from_checkpoint(str(path))


def test_missing_format_is_rejected(tmp_path):
    """Checkpoints predating the format field fail loudly rather than with a
    confusing KeyError halfway through loading."""
    path = tmp_path / "ancient.pt"
    torch.save({"heads": {}}, path)
    with pytest.raises(ValueError, match="checkpoint format"):
        Model.from_checkpoint(str(path))


def test_missing_weights_are_rejected(tmp_path):
    """strict=False makes the load permissive, so an attribute head that failed
    to save would otherwise load silently with random weights."""
    path = tmp_path / "heads.pt"
    model = build(unfreeze="layer4")
    model.save_heads(str(path))

    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["heads"]["attribute_heads.cell_size.weight"]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="missing weights"):
        Model.from_checkpoint(str(path))


def test_unexpected_weights_are_rejected(tmp_path):
    path = tmp_path / "heads.pt"
    model = build(unfreeze="layer4")
    model.save_heads(str(path))

    checkpoint = torch.load(path, weights_only=True)
    checkpoint["heads"]["attribute_heads.nonsense.weight"] = torch.zeros(1)
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="unexpected keys"):
        Model.from_checkpoint(str(path))


# ------------------------------------------------- end-to-end composition


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_predict_contributions_explain_compose(head):
    """The three inference stages have to agree on the concept vector's layout.
    A mismatch would silently attribute the wrong feature."""
    model = build(head=head).eval()
    result, class_dist, concepts = predict(model, torch.randn(3, 224, 224))

    assert set(result) == set(vocab.ATTRIBUTES)
    for attr, (value, confidence) in result.items():
        assert value in vocab.ATTRIBUTE_VOCAB[attr]
        assert 0.0 <= confidence <= 1.0

    assert set(class_dist) == set(vocab.CLASSES)
    assert abs(sum(class_dist.values()) - 1.0) < 1e-5
    assert concepts.shape == (1, CONCEPT_DIM)

    label = max(class_dist, key=class_dist.get)
    scores = contributions(model, result, concepts, label)
    assert set(scores) == set(vocab.ATTRIBUTES)

    text = explain(result, scores, label)
    assert text and text.endswith(".")


def test_linear_contributions_match_the_closed_form():
    """For a linear head, input x gradient reduces exactly to W[label, i] *
    confidence. If this drifts, the attribution is no longer the thing the
    docstring claims it is."""
    model = build(head="linear").eval()
    result, class_dist, concepts = predict(model, torch.randn(3, 224, 224))
    label = vocab.CLASSES[0]
    scores = contributions(model, result, concepts, label)

    weight = model.class_head.weight[vocab.CLASS_TO_INDEX[label]]
    offset = 0
    for attr in vocab.ATTRIBUTES:
        value, confidence = result[attr]
        dim = offset + vocab.encode(attr, value)
        assert scores[attr] == pytest.approx(float(weight[dim]) * confidence, abs=1e-5)
        offset += vocab.num_classes(attr)
