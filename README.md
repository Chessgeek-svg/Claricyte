# Claricyte

An AI-powered study tool for hematology morphology identification. Claricyte helps
clinical laboratory professionals and students learn to identify white blood cell
types by pairing each cell with a plain-English explanation of the morphological
features that support its classification.

## How it works

Claricyte is built on a concept bottleneck model (CBM) rather than a black-box
classifier. Instead of mapping an image straight to a label, it uses a shared vision backbone to feed 11 attribute heads, which predict morphological featurtes of the cell. Finally, those predictions and their confidence values are taken and fed to the classification head, which provides the classification prediction for the cell.

```
image → 11 morphological attributes (each with a confidence) → cell class → explanation
```


Because the explanation can only ever speak from the model's own predicted
attributes, it reports what the model actually observed on this cell, rather 
than regurgitating a rote textbook description of the cell.

## Study (quiz) mode

The demo runs as a quiz. Claricyte draws a real validation-set cell, you guess its
type, and then it reveals the answer along with the model's predicted morphological
attributes and a plain-English explanation of why those features fit the known-correct
classification. You can also browse a specific cell type to study it directly.

The explanation is always built for the known-correct label (the answer key), which is what 
makes it a reliable learning aid that connects the visible morphology to the classification.

## Running the demo

The demo is a Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Nothing else to acquire. The repository ships the validation split it quizzes on
under `demo_data/`, and the trained model under `checkpoints/`. The checkpoint
stores only the trained heads (about 260KB); the frozen ResNet50 backbone is
fetched by timm on first run rather than committed, since it never changes during
training.

Reproducing the model from the full corpus is a separate matter: that needs the
source datasets, which carry their own licensing terms (see below), followed by
the two-stage training in `scripts/` (`train_attr_heads.py`, then
`train_class_head.py`).

## Scope

- Phase 1 (current): WBC morphology on single-cell crops. The demo covers 6 classes
  drawn from the attribute-labeled subset; the full unified taxonomy spans 17 classes
  across six source datasets.
- Planned: a learned language layer (fine-tuned VLM) for more fluent explanations over
  the same faithful attribute bottleneck; RBC morphology and parasite detection;
  user-submitted microscopy.

## Credits & attribution

Claricyte is trained and evaluated on publicly released data, used here under their
respective licenses:

- Blood cell images: Acevedo et al., "A dataset of microscopic peripheral blood cell
  images for development of automatic recognition systems," Data in Brief, 2020.
  Licensed CC BY 4.0.
- Morphological attribute annotations: the WBCAtt dataset (Tsutsui et al., MIT
  License), introduced in:
  Satoshi Tsutsui, Winnie Pang, Shuting He, and Bihan Wen, "WBCAtt+: Fine-Grained
  Pixel-Level Morphological Annotations for White Blood Cell Images," Medical Image
  Analysis, 2026. arXiv:2605.19692

## License

The code in this repository is released under the MIT License. Dataset images and
annotations retain the licenses listed above; any bundled sample images are
redistributed under CC BY 4.0 with attribution to their original authors.
