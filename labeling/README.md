# Morphological Attribute Labeling

Tooling for manually labeling the 11 WB-CAtt morphological attributes on cell
images from classes WB-CAtt doesn't cover (blast, hairy cell, plasma cell,
promyelocyte, etc.). These manual labels extend the attribute training set used
by the concept-bottleneck model. Labels are collected in Label Studio using the
same attribute schema as WB-CAtt so the two are interchangeable.

Paths below assume commands are run from the repo root.

## One-time setup

Label Studio is a standalone app; install it isolated from the project env:

```bash
pipx install label-studio          # or: pip install label-studio in its own env
```

## Per-batch workflow

1. **Generate a task batch** for one class (run in the `claricyte` env):

   ```bash
   python labeling/make_tasks.py --label Blast --n 300
   ```

   This picks unlabeled images of that class, copies/converts them to
   browser-viewable JPGs under `labeling/cache/`, and writes `labeling/tasks.json`.
   Each task keeps the true `image_path` so exports map back to the real image.

2. **Start Label Studio with local file serving.** The env vars must be set in
   the same shell, before starting — otherwise images won't load. The document
   root must be the repo root so it covers `labeling/cache/`:

   ```bash
   export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
   export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)"
   label-studio start
   ```

3. **Configure the project** (once):
   - Settings -> Labeling Interface -> Code -> paste `labeling/wbcatt_config.xml`.
   - Settings -> Cloud Storage -> Add Source Storage -> **Local files**, absolute
     path pointing at `<repo root>/labeling/cache`, **Save** (do NOT Sync — sync
     would create tasks without metadata).
   - Data Import -> `labeling/tasks.json`.

4. **Label**, then export (converter to `metadata/manual_attributes.csv` is TBD).

`labeling/cache/` and `labeling/tasks.json` are generated and gitignored.
