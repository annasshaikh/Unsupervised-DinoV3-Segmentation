"""
pipeline.py – End-to-end segmentation pipeline.

The :class:`Pipeline` class ties together all stages:
    1. Resolution recovery (patch → pixel features)
    2. Low-level feature fusion   (optional)
    3. Clustering
    4. Assignment
    5. Post-processing            (optional, chained)

It supports:
    * Single-image inference  : :meth:`run`
    * Full test-set evaluation : :meth:`evaluate`
    * Ablation sweeps         : :meth:`run_experiment`

New in this version
-------------------
* ``run()`` now accepts ``mask_embedding`` (shape 768) and ``cls_embedding``
  (shape 768) from the precomputed dataset embeddings.
* When the assignment method is ``"mask_embedding_cosine"`` (A-7), the
  cluster-to-class mapping is determined by cosine similarity between each
  cluster's mean pixel-feature and the precomputed ``mask_embedding``, with
  no need for GT pixel annotations at inference time.
* All other assignment methods continue to work unchanged.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import PipelineConfig, get_default_config
from .resolution import get_resolution_method
from .clustering import get_clustering_method
from .assignment import get_assignment_method
from .postprocess import get_postprocessor
from .lowlevel import get_lowlevel_feature, fuse_features
from .metrics import compute_all_metrics, print_metrics


class Pipeline:
    """
    End-to-end DINO-based unsupervised segmentation pipeline.

    Parameters
    ----------
    config : PipelineConfig or dict
        Full pipeline configuration.  Pass a dict to auto-convert.

    Examples
    --------
    >>> from dinov3_seg import Pipeline, get_default_config
    >>> cfg = get_default_config()
    >>> pipeline = Pipeline(cfg)
    >>> pred_mask = pipeline.run(image_tensor, patch_tokens=tokens,
    ...                          mask_embedding=mask_emb)
    """

    def __init__(self, config: Union[PipelineConfig, Dict[str, Any]] = None) -> None:
        if config is None:
            config = get_default_config()
        if isinstance(config, dict):
            config = PipelineConfig.from_dict(config)
        self.config = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu"
            else "cpu"
        )
        if config.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available; using CPU.")

        # ── Instantiate stages ────────────────────────────────────────────────
        self._resolution   = get_resolution_method(
            config.resolution.get("method", "bilinear"),
            **{k: v for k, v in config.resolution.items() if k != "method"},
        )

        clust_cfg  = copy.deepcopy(config.clustering)
        clust_name = clust_cfg.pop("method", "kmeans")
        self._clustering   = get_clustering_method(clust_name, **clust_cfg)

        assign_cfg  = copy.deepcopy(config.assignment)
        assign_name = assign_cfg.pop("method", "majority_vote")
        self._assignment_name = assign_name
        self._assignment      = get_assignment_method(assign_name, **assign_cfg)

        self._postprocessors = []
        for pp_cfg in config.postprocess:
            pp_cfg  = copy.deepcopy(pp_cfg)
            pp_name = pp_cfg.pop("method")
            self._postprocessors.append(get_postprocessor(pp_name, **pp_cfg))

        self._lowlevel = None
        if config.lowlevel is not None:
            ll_cfg  = copy.deepcopy(config.lowlevel)
            ll_name = ll_cfg.pop("method")
            self._ll_fusion_mode  = ll_cfg.pop("fusion_mode", "concat")
            self._ll_low_weight   = ll_cfg.pop("low_weight", 1.0)
            self._lowlevel        = get_lowlevel_feature(ll_name, **ll_cfg)

        self._global_mask_embedding: Optional[Tensor] = None

    # ── Core inference ────────────────────────────────────────────────────────

    def run(
        self,
        image: Tensor,
        patch_tokens: Optional[Tensor] = None,
        gt_mask: Optional[Tensor] = None,
        mask_embedding: Optional[Tensor] = None,
        cls_embedding: Optional[Tensor] = None,
        return_stages: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        """
        Run the full pipeline on a single image (no batch dimension).

        Parameters
        ----------
        image          : Tensor (3, H, W) – ImageNet-normalised
        patch_tokens   : Tensor (196, 768) – precomputed patch embeddings
        gt_mask        : Tensor (H, W) – required for most assignment methods;
                         if None and assignment is not mask_embedding_cosine,
                         cluster IDs are returned as-is.
        mask_embedding : Tensor (768,) – avg patch feature inside GT mask.
                         Required when assignment == "mask_embedding_cosine".
        cls_embedding  : Tensor (768,) – [CLS] token (stored but not used
                         by the pipeline core; available for downstream tasks).
        return_stages  : bool – if True also return a dict of intermediate outputs.

        Returns
        -------
        pred_mask    : Tensor (H, W)
        stage_outputs : dict  (only when return_stages=True)
        """
        stages: Dict[str, Any] = {}
        cfg = self.config

        # ── 0. Validate inputs ────────────────────────────────────────────────
        if patch_tokens is None:
            raise ValueError(
                "patch_tokens must be provided. "
                "Precompute DINO embeddings and pass them here."
            )

        image        = image.to(self.device)
        patch_tokens = patch_tokens.to(self.device)          # (N, D)

        if mask_embedding is not None:
            mask_embedding = mask_embedding.to(self.device)  # (D,)
        if cls_embedding is not None:
            cls_embedding  = cls_embedding.to(self.device)   # (D,)

        stages["cls_embedding"]  = cls_embedding
        stages["mask_embedding"] = mask_embedding

        # ── 1. Resolution recovery ────────────────────────────────────────────
        # patch_tokens: (N, D) → add batch dim → (1, N, D)
        pixel_feats = self._resolution.upsample(
            patch_tokens.unsqueeze(0),
            original_image=image.unsqueeze(0),
            target_size=cfg.target_size,
        )[0]                                                  # (H, W, D)
        stages["pixel_features"] = pixel_feats

        # ── 2. Low-level fusion (optional) ────────────────────────────────────
        if self._lowlevel is not None:
            ll_kwargs: Dict[str, Any] = {}
            if hasattr(self._lowlevel, "extract"):
                import inspect
                sig = inspect.signature(self._lowlevel.extract)
                if "dino_features" in sig.parameters:
                    ll_kwargs["dino_features"] = pixel_feats
            ll_feats = self._lowlevel.extract(image.cpu(), **ll_kwargs).to(self.device)
            # Ensure spatial alignment
            if ll_feats.shape[:2] != pixel_feats.shape[:2]:
                H_pf, W_pf = pixel_feats.shape[:2]
                ll_feats = F.interpolate(
                    ll_feats.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(H_pf, W_pf), mode="bilinear", align_corners=True,
                )[0].permute(1, 2, 0)
            pixel_feats = fuse_features(
                pixel_feats, ll_feats,
                mode=self._ll_fusion_mode,
                low_weight=self._ll_low_weight,
            )
            stages["fused_features"] = pixel_feats

        # ── 3. Clustering ─────────────────────────────────────────────────────
        H, W, D    = pixel_feats.shape
        flat_feats = pixel_feats.reshape(-1, D)              # (H*W, D)

        # Build spatial positions for methods that use them
        rows = torch.arange(H, device=self.device).float() / H
        cols = torch.arange(W, device=self.device).float() / W
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        spatial_pos = torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], dim=1)

        n_clusters = cfg.clustering.get("n_clusters", 8)
        cluster_labels = self._clustering.cluster(
            flat_feats.cpu(),
            n_clusters=n_clusters,
            spatial_positions=spatial_pos.cpu(),
        ).reshape(H, W).to(self.device)                      # (H, W)
        stages["cluster_labels"] = cluster_labels

        # ── 4. Assignment ─────────────────────────────────────────────────────
        # Align GT mask size if it differs from cluster resolution (e.g. "none" resolution)
        _gt = gt_mask
        if _gt is not None and _gt.shape != cluster_labels.shape:
            _gt = F.interpolate(
                _gt.unsqueeze(0).unsqueeze(0).float(),
                size=cluster_labels.shape,
                mode="nearest",
            )[0, 0].long()

        pred_mask = self._assign(
            cluster_labels=cluster_labels,
            gt_mask=_gt,
            pixel_feats=pixel_feats,
            mask_embedding=mask_embedding,
        )

        stages["pred_mask_before_postprocess"] = pred_mask

        # ── 5. Post-processing ────────────────────────────────────────────────
        for pp in self._postprocessors:
            pred_mask = pp.process(pred_mask, image.cpu())
        stages["pred_mask"] = pred_mask

        if return_stages:
            return pred_mask, stages
        return pred_mask

    # ── Internal assignment router ────────────────────────────────────────────

    def _assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Optional[Tensor],
        pixel_feats: Tensor,
        mask_embedding: Optional[Tensor],
    ) -> Tensor:
        """
        Route to the correct assignment call depending on the chosen method.

        * ``mask_embedding_cosine`` (A-7): uses mask_embedding + pixel_feats;
          does NOT require gt_mask.
        * All other methods: require gt_mask (falls back to cluster IDs if None).
        """
        if self._assignment_name == "mask_embedding_cosine":
            # A-7 needs pixel_feats and mask_embedding; gt_mask is only for fallback
            _gt = gt_mask if gt_mask is not None else torch.zeros(
                cluster_labels.shape, dtype=torch.long, device=cluster_labels.device
            )
            return self._assignment.apply(
                cluster_labels, _gt,
                mask_embedding=mask_embedding,
                pixel_features=pixel_feats,
            )

        # All other methods require gt_mask for supervised assignment
        if gt_mask is not None:
            gt_mask = gt_mask.to(cluster_labels.device)
            return self._assignment.apply(cluster_labels, gt_mask)
        else:
            # No GT available → return raw cluster labels
            return cluster_labels.long()

    # ── Global Embedding Support ──────────────────────────────────────────────

    def set_global_mask_embedding(self, embedding: Tensor) -> None:
        """Set a global reference vector for MaskEmbeddingCosine (A-7)."""
        self._global_mask_embedding = embedding
        if self._assignment_name == "mask_embedding_cosine":
            self._assignment.global_embedding = embedding

    def compute_global_mask_embedding(self, dataloader) -> Tensor:
        """
        Average all mask_embeddings in the dataloader to produce a global
        prototype vector.
        """
        all_embs = []
        for batch in dataloader:
            embs = batch["mask_embedding"]  # (B, D)
            # Only include non-zero embeddings (images with foreground)
            for i in range(embs.shape[0]):
                if embs[i].norm() > 1e-6:
                    all_embs.append(embs[i].cpu())

        if not all_embs:
            warnings.warn("No valid mask embeddings found in dataloader.")
            return torch.zeros(768)

        global_avg = torch.stack(all_embs, dim=0).mean(dim=0)
        self.set_global_mask_embedding(global_avg)
        return global_avg

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        dataloader=None,
        dataset_path: Optional[str] = None,
        split: str = "test",
        use_global_embedding: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run the pipeline on a full split and aggregate metrics.

        Parameters
        ----------
        dataloader   : DataLoader or None
            If None, one is built from ``config.dataset_path``.
        dataset_path : str, optional
            Override ``config.dataset_path``.
        split        : str
        use_global_embedding : bool
            If True and split is "test", the pipeline first computes (or uses)
             the global average mask embedding from the training set.
        verbose      : bool

        Returns
        -------
        metrics : dict
        """
        from .dataloader import get_dataloaders

        dp = dataset_path or self.config.dataset_path
        if dataloader is None:
            _, test_loader = get_dataloaders(
                dp,
                batch_size=1,
                num_workers=self.config.num_workers,
            )
            if split == "train":
                train_loader, _ = get_dataloaders(
                    dp, batch_size=1, num_workers=self.config.num_workers
                )
                dataloader = train_loader
            else:
                dataloader = test_loader

        # ── Handle Global Embedding ──
        if use_global_embedding and split == "test":
            if self._global_mask_embedding is None:
                if verbose:
                    print("Computing global mask embedding from train set...")
                train_loader, _ = get_dataloaders(
                    dp, batch_size=1, num_workers=self.config.num_workers
                )
                self.compute_global_mask_embedding(train_loader)

        all_metrics: List[Dict[str, float]] = []
        for batch in dataloader:
            images         = batch["image"]          # (B, 3, H, W)
            masks          = batch["mask"]           # (B, 1, H, W)
            patch_tokens   = batch["patch_tokens"]   # (B, 196, D)
            cls_embeddings = batch["cls_embedding"]  # (B, D)
            mask_embeddings= batch["mask_embedding"] # (B, D)

            B = images.shape[0]
            for b in range(B):
                img     = images[b]
                gt      = masks[b, 0].long()
                tokens  = patch_tokens[b]
                cls_emb = cls_embeddings[b]
                mask_emb= None if use_global_embedding else mask_embeddings[b]

                try:
                    # Hide ground truth from the pipeline during inference for test split
                    # to ensure strict unsupervised evaluation without leaking GT.
                    pipeline_gt = None if split == "test" else gt
                    
                    pred = self.run(
                        img,
                        patch_tokens=tokens,
                        gt_mask=pipeline_gt,
                        mask_embedding=mask_emb,
                        cls_embedding=cls_emb,
                    )
                    # Align prediction to GT size if they differ (e.g. "none" resolution)
                    if pred.shape != gt.shape:
                        pred = F.interpolate(
                            pred.unsqueeze(0).unsqueeze(0).float(),
                            size=gt.shape,
                            mode="nearest",
                        )[0, 0].long()

                    m = compute_all_metrics(pred, gt, self.config.n_classes)
                    all_metrics.append(m)
                except Exception as exc:
                    warnings.warn(f"[Pipeline.evaluate] Error on sample: {exc}")

        if not all_metrics:
            return {}

        # Aggregate: mean over valid (non-nan) values
        keys = all_metrics[0].keys()
        agg  = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))]
            agg[k] = float(np.mean(vals)) if vals else float("nan")

        if verbose:
            print_metrics(agg, title=f"Evaluation ({split})")

        return agg

    # ── Ablation sweep ────────────────────────────────────────────────────────

    def run_experiment(
        self,
        sweep: Dict[str, List[Any]],
        dataloader=None,
        fixed_config: Optional[PipelineConfig] = None,
        verbose: bool = True,
    ):
        """
        Sweep over a grid of configuration values and collect metrics.

        Parameters
        ----------
        sweep : dict
            Each key is a dotted path into the config
            (e.g. ``"clustering.n_clusters"``), mapping to a list of values
            to try.  Currently supports single-parameter sweeps (the first
            key/value pair is used for the sweep; others are fixed).
        dataloader   : DataLoader or None
        fixed_config : PipelineConfig or None  – base config to sweep from.
        verbose      : bool

        Returns
        -------
        results_df : pandas.DataFrame
            Columns are sweep parameter + metric names.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for run_experiment.")

        base_cfg = copy.deepcopy(fixed_config or self.config)
        rows     = []

        for param_path, values in sweep.items():
            for val in values:
                # Apply value to config
                cfg  = copy.deepcopy(base_cfg)
                keys = param_path.split(".")
                obj  = cfg
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                if isinstance(obj, dict):
                    obj[keys[-1]] = val
                else:
                    setattr(obj, keys[-1], val)

                # Run pipeline
                pl  = Pipeline(cfg)
                m   = pl.evaluate(dataloader=dataloader, verbose=False)
                row = {param_path: val, **m}
                rows.append(row)
                if verbose:
                    print(
                        f"  {param_path}={val!r}  →  "
                        f"mIoU={m.get('miou', float('nan')):.4f}"
                    )

        return pd.DataFrame(rows)
