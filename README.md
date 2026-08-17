# Claricyte

An AI-powered study tool for hematology morphology identification. Claricyte helps
clinical laboratory professionals and students learn to identify white blood cell
types by pairing each cell with a plain-English explanation of the morphological
features that support its classification.

Live demo: [claricyte.streamlit.app](https://claricyte.streamlit.app/)

## How it works

Claricyte is built on a concept bottleneck model (CBM). Instead of mapping an image straight to a label like in a standard black-box classifier, a shared vision backbone feeds 11 attribute heads that predict the cell's morphological features. Those predictions, with their confidence values, are the only input to the classification head, which of course makes the final prediction.

```
image -> 11 morphological attributes (each with a confidence) -> cell class -> explanation
```

Because the classifier sees nothing but the 11 predicted attributes, and the explanation is generated from those same predictions, the explanation text reports what the model actually observed on this cell rather than reciting a textbook description of the cell type. When the model is wrong, the explanation shows you why it was wrong.

The 11 attributes come from the WBCAtt schema: cell size and shape, nucleus shape, nuclear-to-cytoplasmic ratio, chromatin density, cytoplasm colour, texture and vacuolation, and granularity with granule type and colour. The current version of the model analyzes 6 classes, splitting neutrophils into band and segmented neutrophils, while also categorizing monocytes, lymphocytes, eosinophils, and basophils. More cell types are planned, see [Scope](#scope) for details

## Study (quiz) mode

The demo runs as a quiz, where Claricyte draws a real validation-set cell, you guess its type, then it reveals the answer along with the model's predicted attributes and an explanation of why those features fit the known-correct classification. You can also browse a specific cell type to study it directly.

Explanations are always built for the known-correct label rather than the model's own guess. This way, the model is still always explaining features that it actually saw in the cell that correlate with the correct cell type, even if it happened to predict incorrectly on its own. The model's raw class prediction is still shown, under "Model internals", so you can see where it agrees and where it does not.

## Current performance

Six classes: band neutrophil, basophil, eosinophil, lymphocyte, monocyte, segmented neutrophil.

| Metric | Value |
|---|---|
| Validation class accuracy | 0.930 |
| Mean attribute accuracy (11 heads) | 0.926 |
| Ceiling given perfect discrete attributes | 0.925 |
| Random baseline | 0.167 |

Measured on 1,568 held-out validation cells. These are validation figures, and validation also selects the checkpoint, so they are mildly optimistic. A held-out test split exists and has deliberately not been touched, as selecting the model based on test performance is data leakage.

The previous iteration of the model was unimpressive at both attribute accurace and class accuracy. To work out where the error was actually coming from, I fed the classifier the ground-truth attributes instead of the predicted ones. It reached an accuracy of 0.920, against a theoretical maximum of 0.925 for any model reading only these 11 discrete attributes. That told me the classifier itself was already solid, and the larger issue lied in attribute heads misinterpreting the morphological features present in the cell. Fine-tuning the vision backbone closed most of the vision error, taking mean attribute accuracy from 0.81 to 0.926 and class accuracy from 0.774 to 0.930. 

The model now sits slightly above the 0.925 maximum, which is possible because the classifier reads the full probability distribution over each attribute rather than only the single most likely value. A cell whose nucleus shape is a coin flip between band and bilobed is carried forward as a coin flip, which lets the other attributes count for proportionally more in the final call rather than being overridden by a forced choice. For reference, discretizing those distributions before the classifier would drop the accuracy to 0.876.

The remaining error comes from cells with overlapping attribute characteristics, which the model cannot possibly separate. 83.5% of all remaining mistakes are band and segmented neutrophils being confused for each other; from my examination of the dataset, I strongly suspect that a large portion of this stems from the nucleus shape labeling. 39.6% of segmented neutrophils are labeled as having a nucleus shape of unsegmented-band, which is the key feature that would distinguish segmented neutrophils from band neutrophils. This is supported by the segmented neutrophil class having 0.679 recall even when handed the ground-truth attributes, while all other classes are over .900

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repository ships the validation split it quizzes on under `demo_data/`, and the trained model under `checkpoints/`. The checkpoint is about 47MB, because the ResNet50 backbone is fine-tuned rather than frozen and so cannot simply be refetched from timm; it is stored in half precision, which halves the file at no measurable cost to accuracy.

Retraining the model would require the Acevedo image set and the WBCAtt attribute annotations, which carry their own licensing terms (see [Credits & attribution](#credits--attribution)), followed by the two-stage training in `scripts/` (`train_attr_heads.py`, then `train_class_head.py`). The two stages are trained separately on purpose, as otherwise the class objective (having received the correct label and then backpropagating to the attribute heads) would influence the attribute heads towards misidentified morphological cell features in order to make classification more accurate. For example, if an image of a segmented neutrophil was mistakenly labeled as an eosinophil, the jointly trained attribute head would be influenced by the class head to predict red granules on that image, even though none were actually present. Separating them allows the attribnute heads to visually identify morphological features as accurately as possible, and then rely on the class head to convert those confidence values into a final class prediction, which can then be explained by what attributes were actually seen in that individual cell.

## Scope

Claricyte is in phase 1: white blood cell morphology on single-cell crops. The current model covers 6 classes, and the full taxonomy for this phase is 17. The plan from here, in order:

1. Clarify the band versus segmented neutrophil confusion, which is where nearly all of the remaining error sits. The attribute vocabulary is already capable of expressing the distinction, it is just being applied inconsistently.
2. Finish phase 1 by extending to all 17 classes. This is gated on attribute labels for the immature and neoplastic cell types, which do not exist yet and have to be produced by hand.
3. Multi-cell detection, so that a whole field image can be broken into individual cells rather than requiring them to be cropped in advance. The goal is being able to accept user-submitted images.
4. Red cell morphology and parasite detection, which will likely need separate models rather than more classes on this one.

## Credits & attribution

Claricyte is trained and evaluated on publicly released data, used here under their respective licenses:

- Blood cell images: Acevedo et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems," Data in Brief, 2020. Licensed CC BY 4.0.
- Morphological attribute annotations: the WBCAtt dataset (Tsutsui et al., MIT License), introduced in:
  Satoshi Tsutsui, Winnie Pang, and Bihan Wen, "WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes," Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2306.13531
  The authors also ask that work using WBCAtt cite its successor, which extends the dataset with pixel-level annotations not used here:
  Satoshi Tsutsui, Winnie Pang, Shuting He, and Bihan Wen, "WBCAtt+: Fine-Grained Pixel-Level Morphological Annotations for White Blood Cell Images," Medical Image Analysis, 2026. arXiv:2605.19692

## License

The code in this repository is released under the MIT License. Dataset images and annotations retain the licenses listed above; any bundled sample images are redistributed under CC BY 4.0 with attribution to their original authors.