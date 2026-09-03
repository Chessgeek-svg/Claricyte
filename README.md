# Claricyte

A vision model powered study tool for hematology morphology identification. Claricyte helps clinical laboratory professionals and students learn to identify white blood cell types by pairing each cell with a plain-English explanation of the morphological features that support its classification. For educational use only.

Live demo: [claricyte.streamlit.app](https://claricyte.streamlit.app/)

## How it works

Claricyte is built on a concept bottleneck model (CBM). Instead of mapping an image straight to a label like in a standard black-box classifier, a shared vision backbone feeds 11 attribute heads that predict the cell's morphological features. Those predictions, with their confidence values, are the only input to the classification head, which of course makes the final prediction.

```
image -> 11 morphological attributes (each with a confidence) -> cell class -> explanation
```

Because the classifier sees nothing but the 11 predicted attributes, and the explanation is generated from those same predictions, the explanation text reports what the model actually observed on this cell rather than reciting a textbook description of the cell type. When the model is wrong, the explanation shows you why it was wrong.

The 11 attributes come from the WBCAtt schema: cell size and shape, nucleus shape, nuclear-to-cytoplasmic ratio, chromatin density, cytoplasm colour, texture and vacuolation, and granularity with granule type and colour. The current version of the model analyzes 6 classes, splitting neutrophils into band and segmented neutrophils, while also categorizing monocytes, lymphocytes, eosinophils, and basophils. More cell types are planned, see [Scope](#scope) for details.

## Study (quiz) mode

The demo runs as a quiz, where Claricyte draws a real validation-set cell, you guess its type, then it reveals the answer along with the model's predicted attributes and an explanation of why those features fit the known-correct classification. You can also browse a specific cell type to study it directly.

Explanations are always built for the known-correct label rather than the model's own guess. This way, the model is still always explaining features that it actually saw in the cell that correlate with the correct cell type, even if it happened to predict incorrectly on its own. The model's raw class prediction is still shown, under "Model internals", so you can see where it agrees and where it does not.

## Clinical context

Once a cell is classified, Claricyte surfaces a cited reference panel highlighting associated clinical conditions and related smear findings, alongside an open prompt for follow-up questions.

This pipeline relies on a strict separation of concerns. The language model never sees the image, ensuring it cannot make ungrounded visual assertions. The concept bottleneck model handles all morphology, while the RAG pipeline exclusively synthesizes context from vetted, license-compliant reference texts, and does not fabricate an answer when it fails to find one in its sources.

```
predicted class -> metadata filter + prose query -> retrieved chunks -> cited answer
```

The corpus is 1,316 chunks from 111 open-access PubMed Central articles and two open-licensed textbooks. Retrieval uses BGE-small-en-v1.5 embeddings in ChromaDB, with the predicted class as a metadata filter over chunks tagged at ingest by alias matching. Generation runs on gpt-4.1-nano at temperature 0, and every emitted citation is checked against what was actually retrieved before the answer is shown.

The per-class panels are generated offline and committed, so that just browsing cells provides some information but doesn't automatically perform an API call with each image.

### Retrieval and grounding performance

Measured on a 55-question gold set, 49 of which are scored on retrieval and 6 of which are adversarial questions the system is supposed to refuse.

| Metric | Value |
|---|---|
| Hit rate at 5 | 0.98 |
| Groundedness (supported claims) | 0.93 |
| Abstention on unanswerable questions | 0.83 |

Four retrieval configurations were compared on the same gold set:

| Configuration | Hit rate at 5 | MRR |
|---|---|---|
| Class as filter, question as prose query | 0.98 | 0.78 |
| No class filter | 0.96 | 0.76 |
| No class prepended to the query text | 0.80 | 0.61 |
| Class name alone as the query | 0.55 | 0.45 |

One of the design decisions I thought was clever was to filter sources on the classes they discussed, which are tagged in the metadata of each chunk. The reasoning behind this was essentially to guarantee that sources are relevant, and prevent false positive hits in sources that refer to unrelated conditions. But removing the metadata filter makes little difference, while removing the class from the query text caused 9 more misses, which would be a hit rate drop of 0.18. So while the class is necessary to ensure the returned information is relevant, it's more useful when it's just steering the embedding, not restricting the candidate pool. Currently the filter is kept since results are still marginally better, and there is always a risk of losing the signal of the class somewhere int he query embedding.

Groundedness is scored by reading every generated claim against the full text of all five retrieved chunks by hand. An LLM judge was built to do this first, and it is still in the eval, but tt was wrong too often to be a usable measurement. Of 14 claims it flagged, only 4 were genuinely unsupported, and it missed 4 that were. The figure above is the hand count, 8 unsupported claims out of 120.

Those 8 share a cause. Every one comes from a chunk that describes two cells at once, and in each the model took a property from the wrong side of the comparison. It defined a hypersegmented neutrophil as having two to five lobes, which is the normal cell, where the source says six or more. It gave a monocyte the lymphocyte's size and nuclear-to-cytoplasmic ratio, which sit in an adjacent row of the same flattened table. It placed myeloperoxidase in the secondary granules when both retrieved sources put it in the primary. So the retrieval is succeeding in returning relevant information, but the model is not always able to correctly interpet it.

## Current performance

Six classes: band neutrophil, basophil, eosinophil, lymphocyte, monocyte, segmented neutrophil.

| Metric | Value |
|---|---|
| Test class accuracy | 0.935 |
| Mean attribute accuracy (11 heads) | 0.936 |
| Ceiling given perfect discrete attributes | 0.930 |
| Random baseline | 0.167 |

Measured on 3,099 held-out test cells. Backbone configurations were compared on the validation split and the winner picked there, then the test split was read once. Selecting a model based on test performance is data leakage, so test never informed a decision.

The partition is WBCAtt's own published train/validation/test split, so these numbers sit directly beside the ones in their paper. On their 5-class task, which merges band and segmented neutrophils into a single neutrophil class, this model scores 99.02 macro F-measure against the 99.40 they report for their attribute-based classifier, and 90.12 against their 91.20 on mean attribute macro F-measure. Slightly behind on both. The 5-class figure is produced by merging our band and segmented predictions back into one class but the model is optimized with the goal of distinguishing them into their own classes. Keeping the two apart, on the 6-class task the model is actually trained for, gives a macro F-measure of 93.27, though the paper has no equivalent figure to compare it against.

The previous iteration of the model was unimpressive at both attribute accuracy and class accuracy. To work out where the error was actually coming from, I fed the classifier the ground-truth attributes instead of the predicted ones. It reached an accuracy of 0.926, against a theoretical maximum of 0.930 for any model reading only these 11 discrete attributes. That told me the classifier itself was already solid, and the larger issue lay in attribute heads misinterpreting the morphological features present in the cell. Fine-tuning the vision backbone closed most of the vision error, taking mean attribute accuracy from 0.81 to 0.936 and class accuracy from 0.774 to 0.935.

The model now sits slightly above the 0.930 maximum, which is possible because the classifier reads the full probability distribution over each attribute rather than only the single most likely value. A cell whose nucleus shape is a coin flip between band and bilobed is carried forward as a coin flip, which lets the other attributes count for proportionally more in the final call rather than being overridden by a forced choice. For reference, discretizing those distributions before the classifier would drop the accuracy to 0.908.

Performance across the six classes:

| Class | Recall |
|---|---|
| Basophil | 0.997 |
| Eosinophil | 0.995 |
| Monocyte | 0.984 |
| Lymphocyte | 0.975 |
| Band Neutrophil | 0.885 |
| Segmented Neutrophil | 0.766 |

Four of the six classes sit above 0.97. Merging the two neutrophil classes into the single neutrophil class the WBCAtt paper uses takes class accuracy to 0.992, so nearly everything the model gets wrong is one boundary.

The remaining error comes from cells with overlapping attribute characteristics, which the model cannot possibly separate. 87.6% of all remaining mistakes are band and segmented neutrophils being confused for each other; from my examination of the dataset, I strongly suspect that a large portion of this stems from the nucleus shape labeling. 39.6% of segmented neutrophils are labeled as having a nucleus shape of unsegmented-band, which is the key feature that would distinguish segmented neutrophils from band neutrophils. This is supported by the segmented neutrophil class having 0.704 recall even when handed the ground-truth attributes, while all other classes are over .900.

## Running the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The quiz and the clinical context panels run offline. The question box calls the OpenAI API, so it needs a key in `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`) or in `OPENAI_API_KEY`. Without one, everything else still works.

The repository ships the validation split it quizzes on under `demo_data/`, and the trained model under `checkpoints/`. The checkpoint is about 47MB, because the ResNet50 backbone is fine-tuned rather than frozen and so cannot simply be refetched from timm; it is stored in half precision, which halves the file at no measurable cost to accuracy.

Retraining the model would require the Acevedo image set and the WBCAtt attribute annotations, which carry their own licensing terms (see [Credits & attribution](#credits--attribution)), followed by the two-stage training in `scripts/` (`train_attr_heads.py`, then `train_class_head.py`). The two stages are trained separately on purpose, as otherwise the class objective (having received the correct label and then backpropagating to the attribute heads) would influence the attribute heads towards misidentified morphological cell features in order to make classification more accurate. For example, if an image of a segmented neutrophil was mistakenly labeled as an eosinophil, the jointly trained attribute head would be influenced by the class head to predict red granules on that image, even though none were actually present. Separating them allows the attribute heads to visually identify morphological features as accurately as possible, and then rely on the class head to convert those confidence values into a final class prediction, which can then be explained by what attributes were actually seen in that individual cell.

## Scope

Claricyte is in phase 1: white blood cell morphology on single-cell crops. The current model covers 6 classes, and the full taxonomy for this phase is 17. The plan from here, in order:

1. Clarify the band versus segmented neutrophil confusion, which is where nearly all of the remaining error sits. The attribute vocabulary is already capable of expressing the distinction, it is just being applied inconsistently. This likely means a manual cleanup of the nucleus shape of segmented neutrophils
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

The clinical context corpus is drawn from openly licensed text, and every chunk notes its licence and a link back to its source:

- Articles from the PubMed Central Open Access subset, each under its own CC0, CC BY, CC BY-SA or CC BY-NC licence, or PMC's text-mining terms. Articles under no-derivatives licences are excluded at ingest, since chunking and reassembling them is arguable as a derivative work.
- Taylor and Doty, *Clinical Hematology Atlas: A Pictorial Guide for the Hematology Laboratory*, Oregon Institute of Technology, via Medicine LibreTexts. Licensed CC BY-NC-SA 4.0.
- Villatoro and To, *A Laboratory Guide to Clinical Hematology*, Open Education Alberta, via Medicine LibreTexts. Licensed CC BY-NC 4.0.

## License

The code in this repository is released under the MIT License. Dataset images and annotations retain the licenses listed above; any bundled sample images are redistributed under CC BY 4.0 with attribution to their original authors.
