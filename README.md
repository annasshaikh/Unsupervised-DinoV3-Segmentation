# dinov3_seg

**Modular unsupervised segmentation pipeline powered by DINOv2 patch embeddings.**

`dinov3_seg` lets you assemble, ablate, and evaluate a full segmentation pipeline from
precomputed DINOv2 features — with no annotations required at clustering time, and full
metric evaluation using provided ground-truth masks.

---

## 1. Project Goal

Given precomputed DINOv2 patch-level embeddings (`*.npy`, one per image), produce
pixel-level semantic segmentation masks through:

1. **Resolution Recovery** – upsample 14×14 patch features to 224×224 pixel features.
2. **Low-level Fusion** *(optional)* – enrich with hand-crafted features (HOG, LBP, colour histograms, …).
3. **Clustering** – group pixels/patches into K unsupervised segments.
4. **Assignment** – map cluster IDs to class labels using GT masks.
5. **Post-processing** *(optional)* – refine masks (CRF, morphology, superpixels, …).

---

## 2. Directory Structure

```
dinov3_seg/
│
├── __init__.py        – Package entry point
├── dataloader.py      – SegmentationDataset + get_dataloaders()
├── resolution.py      – R-1…R-5 resolution recovery methods
├── clustering.py      – C-1…C-7 clustering methods
├── assignment.py      – A-1…A-6 cluster-to-class assignment methods
├── postprocess.py     – P-1…P-7 post-processing methods
├── lowlevel.py        – L-1…L-8 low-level feature extractors
├── pipeline.py        – Pipeline class (run / evaluate / run_experiment)
├── metrics.py         – mIoU, Dice, Boundary F1, diagnostics
├── config.py          – PipelineConfig dataclass + get_default_config()
└── utils.py           – load_dino_model, save_results, visualize_segmentation, …

segmentation_frontend.ipynb   – Interactive Kaggle notebook
README.md
```

---

## 3. Available Methods (Catalogue)

### Resolution Recovery (`resolution.py`)

| ID  | Name        | Description                                      |
|-----|-------------|--------------------------------------------------|
| R-1 | `nearest`   | Nearest-neighbour reshape + upsample             |
| R-2 | `bilinear`  | Bilinear interpolation                           |
| R-3 | `bicubic`   | Bicubic interpolation                            |
| R-4 | `pca`       | PCA compression → bilinear upsample → centroid back-projection |
| R-5 | `bilateral` | Guided joint bilateral upsampling (edge-guided)  |

### Clustering (`clustering.py`)

| ID  | Name           | Description                                                  |
|-----|----------------|--------------------------------------------------------------|
| C-1 | `kmeans`       | MiniBatch K-Means (sklearn)                                  |
| C-2 | `kmeans_pca`   | PCA + K-Means                                                |
| C-3 | `hdbscan`      | HDBSCAN density clustering                                   |
| C-4 | `spectral`     | Spectral clustering (+ optional spatial regularisation)      |
| C-5 | `hierarchical` | Agglomerative: Ward / complete / average linkage             |
| C-6 | `ncut`         | Normalised Cuts (cosine-affinity spectral approximation)     |
| C-7 | `joint_kmeans` | K-Means fitted jointly across all train images               |

### Assignment (`assignment.py`)

| ID  | Name               | Description                                                  |
|-----|--------------------|--------------------------------------------------------------|
| A-1 | `majority_vote`    | Most frequent GT label per cluster                          |
| A-2 | `weighted_majority`| Majority weighted by inverse distance to centroid            |
| A-3 | `hungarian`        | Hungarian matching via IoU cost matrix                       |
| A-4 | `label_propagation`| Soft assignment via K-NN label propagation                   |
| A-5 | `abstention`       | Majority vote + abstain (→ 255) if confidence < threshold    |
| A-6 | `cross_image`      | Per-image majority + cross-image consistency metric          |

### Post-processing (`postprocess.py`)

| ID  | Name             | Description                                             |
|-----|------------------|---------------------------------------------------------|
| P-1 | `morphology`     | Morphological open + close                              |
| P-2 | `connected_comp` | Remove small connected components                       |
| P-3 | `dense_crf`      | DenseCRF refinement (pydensecrf)                        |
| P-4 | `superpixel`     | SLIC superpixel majority-vote pooling                   |
| P-5 | `bilateral_soft` | Bilateral filter on per-class probability maps          |
| P-6 | `graph_cut`      | Graph-cut refinement *(stub — see TODO in source)*      |
| P-7 | `tta`            | Test-time augmentation ensembling                       |

### Low-level Features (`lowlevel.py`)

| ID  | Name          | Description                                                   |
|-----|---------------|---------------------------------------------------------------|
| L-1 | `color_hist`  | HSV colour histogram per patch                                |
| L-2 | `hog`         | Histogram of Oriented Gradients                               |
| L-3 | `lbp`         | Local Binary Patterns                                         |
| L-4 | `slic_pool`   | SLIC superpixel average-pooling of DINO features              |
| L-5 | `edge_avg`    | Sobel edge magnitude map                                      |
| L-6 | `sam_dino`    | SAM proposals + DINO labelling *(optional, needs checkpoint)* |
| L-7 | `watershed`   | Watershed over DINO PCA                                       |
| L-8 | `late_fusion` | Two parallel extractors fused (concat or add)                 |

---

## 4. Dataset Layout

```
/kaggle/input/<dataset-name>/
    images/
        train/  <stem>.jpg
        test/   <stem>.jpg
    masks/
        train/  <stem>.png
        test/   <stem>.png
    embeddings/
        train/  <stem>.npy    # shape (196, 768) for ViT-B/14
        test/   <stem>.npy
```

The `embeddings/` directory must be precomputed with a DINOv2 model before running the
pipeline.  Use `utils.extract_patch_tokens()` as a helper.

---

## 5. Quick-start Example

```python
from dinov3_seg import Pipeline, get_default_config

# Minimal config (bilinear resolution, K-Means K=8, majority vote, no post-proc)
cfg = get_default_config(
    dataset="Kvasir",
    dataset_path="/kaggle/input/kvasir-seg",
    n_classes=2,
    device="cuda",
)

# Override any stage
cfg.clustering["n_clusters"] = 16
cfg.resolution["method"] = "bicubic"
cfg.postprocess = [{"method": "dense_crf", "n_iter": 10}]

pipeline = Pipeline(cfg)

# Evaluate on test set
metrics = pipeline.evaluate()
# → {"miou": 0.72, "pixel_acc": 0.91, "dice": 0.83, "bf1_w2": 0.61, ...}

# Single image
import torch
image        = torch.randn(3, 224, 224)   # your normalised image
patch_tokens = torch.randn(196, 768)       # your DINO embeddings
gt_mask      = torch.randint(0, 2, (224, 224))

pred_mask = pipeline.run(image, patch_tokens=patch_tokens, gt_mask=gt_mask)
```

### Ablation sweep

```python
results_df = pipeline.run_experiment(
    sweep={"clustering.n_clusters": [2, 4, 8, 16, 32]},
)
# Returns a pandas DataFrame: n_clusters | miou | pixel_acc | dice | …
```

---

## 6. Kaggle Notebook (`segmentation_frontend.ipynb`)

Open `segmentation_frontend.ipynb` in a Kaggle notebook with GPU enabled:

1. **Cell 0** – installs missing Python packages.
2. **Cell 1** – imports and GPU check.
3. **Cell 2** – set dataset paths (edit `DATASET_PATHS` dict).
4. **Cell 3** – interactive `ipywidgets` UI for selecting all pipeline options.
5. **Cell 5** – click **▶ Run Pipeline** to evaluate and see:
   - Metrics table (mIoU, Dice, Boundary F1 …)
   - Visual gallery (image | predicted mask | GT mask)
6. **Cell 6** – click **📊 Run Ablation** to sweep K and plot mIoU vs K.
7. **Cell 7** – quick-start without UI.

---

## 7. Extending the Pipeline

### Adding a new clustering method

1. Open `dinov3_seg/clustering.py`.
2. Create a class inheriting from `Clustering` and implement `cluster()`.
3. Decorate it with `@_register("my_method")`.
4. That's it — the factory `get_clustering_method("my_method")` will find it.

```python
@_register("my_method")
class MyMethod(Clustering):
    def cluster(self, features, n_clusters=None, spatial_positions=None, **kwargs):
        # Your implementation here
        labels = ...
        return torch.from_numpy(labels).long()
```

The same pattern applies to all other modules — every stage uses the same
`@_register` decorator + factory function pattern.

### Adding a new post-processor

```python
# postprocess.py
@_register("my_pp")
class MyPostProcessor(PostProcessor):
    def process(self, pred_mask, image):
        # refine pred_mask here
        return refined_mask
```

Then include it in your config:
```python
cfg.postprocess = [{"method": "my_pp", "my_param": 42}]
```

### Switching DINO layers

Since embeddings are precomputed `.npy` files, switching to a different layer only
requires re-running the extraction script and saving new embeddings under the same
`embeddings/` layout.  The rest of the pipeline is unchanged.

---

## 8. Dependencies

| Package          | Version   | Purpose                                 |
|------------------|-----------|-----------------------------------------|
| torch            | ≥ 2.0     | Core tensor ops                         |
| torchvision      | ≥ 0.15    | Image transforms                        |
| numpy            | ≥ 1.23    | Array operations                        |
| scikit-learn     | ≥ 1.2     | KMeans, PCA, Spectral, HDBSCAN (≥1.3)  |
| scikit-image     | ≥ 0.20    | SLIC, HOG, LBP, watershed               |
| opencv-python    | ≥ 4.7     | Morphology, CRF pre/post, Sobel         |
| scipy            | ≥ 1.10    | Hungarian algorithm, stats              |
| pandas           | ≥ 1.5     | Ablation result DataFrames              |
| matplotlib       | ≥ 3.7     | Visualisation                           |
| ipywidgets       | ≥ 8.0     | Interactive Kaggle UI                   |
| pydensecrf       | latest    | Dense CRF (P-3); optional              |
| segment-anything | latest    | SAM (L-6); optional                     |

Install on Kaggle:
```bash
pip install -q scikit-learn scikit-image scipy opencv-python-headless pandas matplotlib ipywidgets
pip install -q git+https://github.com/lucasb-eyer/pydensecrf.git   # optional
pip install -q segment-anything                                       # optional (L-6)
```

---

## 9. Notes

- All code is device-agnostic: tensors are moved to the configured device automatically.
- For reproducibility, call `utils.set_seed(42)` before running experiments.
- The CRF (P-3) will warn and skip if `pydensecrf` is not installed.
- SAM (L-6) will warn and fall back to `edge_avg` if the checkpoint is missing.
- Batch size 1 is recommended for evaluation; larger batches work for throughput.
