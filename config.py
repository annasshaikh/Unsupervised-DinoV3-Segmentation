"""
config.py – Pipeline configuration dataclass and defaults.

The :class:`PipelineConfig` is a regular dataclass (no external dependencies).
Nested dicts specify per-stage method names and parameters.

Usage
-----
>>> cfg = get_default_config()
>>> cfg.clustering["method"] = "kmeans_pca"
>>> cfg.clustering["pca_dim"] = 64
>>> pipeline = Pipeline(cfg)
"""

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
        Dataset name, e.g. ``"Kvasir"``, ``"BCCD1"``, ``"BCCD2"``.
    dataset_path : str
        Root path to the dataset directory (under ``/kaggle/input/...``).
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
        Method names: ``"nearest"``, ``"bilinear"``, ``"bicubic"``,
        ``"pca"``, ``"bilateral"``.

    clustering : dict
        Keys: ``method`` (str), ``n_clusters`` (int), plus extras.
        Method names: ``"kmeans"``, ``"kmeans_pca"``, ``"hdbscan"``,
        ``"spectral"``, ``"hierarchical"``, ``"ncut"``, ``"joint_kmeans"``.

    assignment : dict
        Keys: ``method`` (str), plus method-specific kwargs.
        Method names: ``"majority_vote"``, ``"weighted_majority"``,
        ``"hungarian"``, ``"label_propagation"``, ``"abstention"``,
        ``"cross_image"``.

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
    dataset:       str = "Kvasir"
    dataset_path:  str = "/kaggle/input/kvasir-seg"
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

def get_default_config(
    dataset: str = "Kvasir",
    dataset_path: str = "/kaggle/input/kvasir-seg",
    n_classes: int = 2,
    device: str = "cuda",
) -> PipelineConfig:
    """
    Return a :class:`PipelineConfig` with sensible defaults.

    Defaults
    --------
    * Bilinear upsampling
    * K-Means with K=8
    * Majority-vote assignment
    * No post-processing
    * No low-level feature fusion

    Parameters
    ----------
    dataset : str
    dataset_path : str
    n_classes : int
    device : str

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
            "method": "majority_vote",
        },
        postprocess=[],          # No post-processing by default
        lowlevel=None,           # No low-level fusion by default
    )


# ── Preset configs ────────────────────────────────────────────────────────────

def get_strong_config(
    dataset: str = "Kvasir",
    dataset_path: str = "/kaggle/input/kvasir-seg",
    n_classes: int = 2,
) -> PipelineConfig:
    """
    A stronger default configuration combining multiple refinement techniques.

    Uses:
    * Bicubic resolution recovery
    * K-Means PCA with K=16 and PCA dim 64
    * Hungarian matching assignment
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
            "method": "hungarian",
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
