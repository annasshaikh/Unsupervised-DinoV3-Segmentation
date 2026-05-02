"""
2: config.py – Pipeline configuration dataclass and defaults.
3: 
4: The :class:`PipelineConfig` is a regular dataclass (no external dependencies).
5: Nested dicts specify per-stage method names and parameters.
6: 
7: Usage
8: -----
9: >>> cfg = get_default_config()
10: >>> cfg.clustering["method"] = "kmeans_pca"
11: >>> cfg.clustering["pca_dim"] = 64
12: >>> pipeline = Pipeline(cfg)
13: 
14: Dataset layout
15: --------------
16: The pipeline expects the new split-first layout::
17: 
18:     <dataset_path>/
19:         train/
20:             images/     <stem>.jpg
21:             masks/      <stem>.jpg
22:             embeddings/
23:                 patch/  <stem>.npy   (196, 768)
24:                 cls/    <stem>.npy   (768,)
25:                 mask/   <stem>.npy   (768,)
26:         test/  …same structure…
27: """

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    """
    Top-level configuration for the segmentation pipeline.

    Attributes
    ----------
    dataset : str
        Dataset name, e.g. ``"HumanSeg"``, ``"Kvasir"``.
    dataset_path : str
        Root path to the dataset directory (under ``/kaggle/input/...``).
        Must contain ``train/`` and ``test/`` sub-directories each with
        ``images/``, ``masks/``, and ``embeddings/{patch,cls,mask}/``.
    batch_size : int
    num_workers : int
    device : str
        ``"cuda"`` or ``"cpu"``.
    n_classes : int
        Number of semantic classes (including background).
    ignore_index : int
        Label value to ignore in metrics (default 255).
    debug : bool
        If True, compute and log per-stage diagnostic metrics.
    seed : int

    resolution : dict
        Keys: ``method`` (str), plus method-specific kwargs.
        Method names: ``"none"``, ``"nearest"``, ``"bilinear"``, ``"bicubic"``,
        ``"pca"``, ``"bilateral"``.

    clustering : dict
        Keys: ``method`` (str), ``n_clusters`` (int), plus extras.
        Method names: ``"kmeans"``, ``"kmeans_pca"``, ``"hdbscan"``,
        ``"spectral"``, ``"hierarchical"``, ``"ncut"``, ``"joint_kmeans"``.

    assignment : dict
        Keys: ``method`` (str), plus method-specific kwargs.
        Method names:
        - ``"majority_vote"``      (A-1) – most frequent GT label per cluster
        - ``"weighted_majority"``  (A-2) – distance-weighted majority
        - ``"hungarian"``          (A-3) – Hungarian IoU matching
        - ``"label_propagation"``  (A-4) – K-NN label propagation
        - ``"abstention"``         (A-5) – majority vote + abstain
        - ``"cross_image"``        (A-6) – cross-image consistency tracking
        - ``"mask_embedding_cosine"`` (A-7, **recommended**) – cosine sim
          between cluster centroid and precomputed mask embedding.
        - ``"mask_embedding_cosine_global"`` (A-8) – per-patch cosine sim to a
          **global** reference built by averaging mask embeddings across the
          training set. Truly annotation-free at test time; call
          ``pipeline.compute_global_mask_embedding(train_loader)`` before
          ``pipeline.evaluate(..., use_global_embedding=True)``.

    postprocess : list[dict]
        List of post-processors to apply in order.
        Each dict has key ``method`` (str) plus kwargs.
        Empty list = no post-processing.
        Method names: ``"morphology"``, ``"connected_comp"``, ``"dense_crf"``,
        ``"superpixel"``, ``"bilateral_soft"``, ``"graph_cut"``, ``"tta"``.

    lowlevel : dict or None
        If set, specifies a low-level feature extractor to fuse before
        clustering.  Keys: ``method``, ``fusion_mode``, ``low_weight``,
        plus method kwargs.
        Set to ``None`` (default) to disable.
        Method names: ``"color_hist"``, ``"hog"``, ``"lbp"``, ``"slic_pool"``,
        ``"edge_avg"``, ``"sam_dino"``, ``"watershed"``, ``"late_fusion"``.

    target_size : tuple[int, int]
        Spatial size (H, W) to which all outputs are resized (default 224×224).
    """

    # ── Dataset / runtime ────────────────────────────────────────────────────
    dataset:       str = "HumanSeg"
    dataset_path:  str = (
        "/kaggle/input/datasets/muhammadannasshaikh/"
        "dinov3-human-segmentation/human-seg-dataset"
    )
    batch_size:    int = 8
    num_workers:   int = 2
    device:        str = "cuda"
    n_classes:     int = 2
    ignore_index:  int = 255
    debug:         bool = False
    seed:          int  = 42

    # ── Pipeline stages ──────────────────────────────────────────────────────
    resolution:  Dict[str, Any] = field(default_factory=dict)
    clustering:  Dict[str, Any] = field(default_factory=dict)
    assignment:  Dict[str, Any] = field(default_factory=dict)
    postprocess: List[Dict[str, Any]] = field(default_factory=list)
    lowlevel:    Optional[Dict[str, Any]] = None

    # ── Image size ───────────────────────────────────────────────────────────
    target_size: tuple = (224, 224)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config to a plain dict."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Deserialise from a plain dict (e.g. loaded from JSON/YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Default factory ───────────────────────────────────────────────────────────

_DEFAULT_DATASET_PATH = (
    "/kaggle/input/datasets/muhammadannasshaikh/"
    "dinov3-human-segmentation/human-seg-dataset"
)


def get_default_config(
    dataset: str = "HumanSeg",
    dataset_path: str = _DEFAULT_DATASET_PATH,
    n_classes: int = 2,
    device: str = "cuda",
) -> PipelineConfig:
    """
    Return a :class:`PipelineConfig` with sensible defaults for the
    DINOv3 Human-Segmentation dataset.

    Defaults
    --------
    * Bilinear upsampling
    * K-Means with K=8
    * ``mask_embedding_cosine`` assignment (A-7):
      foreground cluster selected by cosine similarity to the precomputed
      mask embedding — no pixel-level GT annotations needed at cluster time.
    * No post-processing
    * No low-level feature fusion

    Parameters
    ----------
    dataset      : str
    dataset_path : str
    n_classes    : int
    device       : str

    Returns
    -------
    PipelineConfig
    """
    return PipelineConfig(
        dataset=dataset,
        dataset_path=dataset_path,
        n_classes=n_classes,
        device=device,
        resolution={
            "method": "bilinear",
        },
        clustering={
            "method":     "kmeans",
            "n_clusters": 8,
        },
        assignment={
            "method": "mask_embedding_cosine",
        },
        postprocess=[],   # No post-processing by default
        lowlevel=None,    # No low-level fusion by default
    )


# ── Preset configs ────────────────────────────────────────────────────────────

def get_small_config(
    dataset: str = "HumanSeg",
    dataset_path: str = _DEFAULT_DATASET_PATH,
    n_classes: int = 2,
) -> PipelineConfig:
    """
    A configuration that runs at patch resolution (no upsampling).
    Fastest but least accurate.
    """
    return PipelineConfig(
        dataset=dataset,
        dataset_path=dataset_path,
        n_classes=n_classes,
        resolution={
            "method": "none",
        },
        clustering={
            "method":     "kmeans",
            "n_clusters": 8,
        },
        assignment={
            "method": "mask_embedding_cosine",
        },
        postprocess=[],
        lowlevel=None,
    )


def get_strong_config(
    dataset: str = "HumanSeg",
    dataset_path: str = _DEFAULT_DATASET_PATH,
    n_classes: int = 2,
) -> PipelineConfig:
    """
    A stronger configuration combining multiple refinement techniques.

    Uses:
    * Bicubic resolution recovery
    * K-Means PCA with K=16 and PCA dim 64
    * ``mask_embedding_cosine`` assignment (A-7)
    * CRF + connected-component post-processing
    * Edge-weighted low-level fusion
    """
    return PipelineConfig(
        dataset=dataset,
        dataset_path=dataset_path,
        n_classes=n_classes,
        resolution={
            "method": "bicubic",
        },
        clustering={
            "method":     "kmeans_pca",
            "n_clusters": 16,
            "pca_dim":    64,
        },
        assignment={
            "method": "mask_embedding_cosine",
        },
        postprocess=[
            {"method": "dense_crf", "n_iter": 10},
            {"method": "connected_comp", "min_area": 100},
        ],
        lowlevel={
            "method":      "edge_avg",
            "fusion_mode": "concat",
            "low_weight":  1.0,
        },
    )


def get_global_config(
    dataset: str = "HumanSeg",
    dataset_path: str = _DEFAULT_DATASET_PATH,
    n_classes: int = 2,
    device: str = "cuda",
) -> PipelineConfig:
    """
    Configuration using A-8 (``mask_embedding_cosine_global``).

    The global reference is built from the training set at evaluation time —
    no per-image GT annotations are required at test time.

    Usage
    -----
    ::

        cfg      = get_global_config(dataset_path=...)
        pipeline = Pipeline(cfg)
        # Build global reference from train split:
        pipeline.evaluate(use_global_embedding=True, split="test")
        # -- or manually --
        pipeline.compute_global_mask_embedding(train_loader)
        pipeline.evaluate(use_global_embedding=True, split="test")
    """
    return PipelineConfig(
        dataset=dataset,
        dataset_path=dataset_path,
        n_classes=n_classes,
        device=device,
        resolution={
            "method": "bilinear",
        },
        clustering={
            "method":     "kmeans",
            "n_clusters": 8,
        },
        assignment={
            "method": "mask_embedding_cosine_global",
        },
        postprocess=[],
        lowlevel=None,
    )
