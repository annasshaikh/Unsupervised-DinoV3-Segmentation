"""
lowlevel.py – Low-level feature extraction and DINO feature fusion.

All methods inherit from :class:`LowLevelFeature` and implement::

    extract(image: Tensor[3, H, W]) -> Tensor[H, W, D_low]

Registered methods
------------------
L-1  color_hist   – Color histogram per patch/pixel
L-2  hog          – Histogram of Oriented Gradients
L-3  lbp          – Local Binary Patterns
L-4  slic_pool    – SLIC superpixel average-pooling of DINO features
L-5  edge_avg     – Edge-weighted feature averaging
L-6  sam_dino     – SAM proposals + DINO labelling (optional)
L-7  watershed    – Watershed over DINO PCA
L-8  late_fusion  – Parallel pipeline late fusion
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_lowlevel_feature(name: str, **kwargs) -> "LowLevelFeature":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown low-level feature {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def fuse_features(
    dino_feats: Tensor,
    low_feats: Tensor,
    mode: str = "concat",
    low_weight: float = 1.0,
) -> Tensor:
    """
    Fuse DINO pixel features with low-level features.

    Parameters
    ----------
    dino_feats : Tensor (H, W, D)
    low_feats  : Tensor (H, W, D_low)
    mode       : ``"concat"`` (default) or ``"add"`` (requires D == D_low).
    low_weight : Scale applied to low_feats before fusion.

    Returns
    -------
    Tensor (H, W, D + D_low) for concat, (H, W, D) for add.
    """
    low_scaled = low_feats * low_weight
    if mode == "concat":
        return torch.cat([dino_feats, low_scaled], dim=-1)
    elif mode == "add":
        if dino_feats.shape[-1] != low_feats.shape[-1]:
            raise ValueError("add fusion requires equal feature dimensions.")
        return dino_feats + low_scaled
    else:
        raise ValueError(f"Unknown fusion mode {mode!r}. Use 'concat' or 'add'.")


# ── Base class ────────────────────────────────────────────────────────────────

class LowLevelFeature(ABC):
    """Abstract base class for low-level feature extractors."""

    @abstractmethod
    def extract(self, image: Tensor) -> Tensor:
        """
        Extract low-level features from an RGB image.

        Parameters
        ----------
        image : Tensor (3, H, W)  – ImageNet-normalised float

        Returns
        -------
        Tensor (H, W, D_low)
        """

    @staticmethod
    def _denorm(image: Tensor) -> np.ndarray:
        """Return HxWx3 uint8 numpy array (undo ImageNet normalisation)."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (image.cpu().float() * std + mean).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── L-1: Color histogram ──────────────────────────────────────────────────────

@_register("color_hist")
class ColorHistogram(LowLevelFeature):
    """
    L-1: Per-patch color histogram in the HSV colour space.

    The image is divided into a regular grid of patches; for each patch a
    histogram over all channels is computed and bilinearly upsampled to full
    resolution.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins per channel (default 8).
    patch_grid : int
        Grid side length (default 14, matching ViT-14).
    """

    def __init__(self, n_bins: int = 8, patch_grid: int = 14) -> None:
        self.n_bins     = n_bins
        self.patch_grid = patch_grid

    def extract(self, image: Tensor) -> Tensor:
        import cv2

        _, H, W = image.shape
        img_np  = self._denorm(image)
        hsv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        G    = self.patch_grid
        D    = 3 * self.n_bins
        hist = np.zeros((G, G, D), dtype=np.float32)

        ph = H // G
        pw = W // G
        for i in range(G):
            for j in range(G):
                patch = hsv[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
                h_h   = np.histogram(patch[:, :, 0], bins=self.n_bins, range=(0, 180))[0]
                h_s   = np.histogram(patch[:, :, 1], bins=self.n_bins, range=(0, 256))[0]
                h_v   = np.histogram(patch[:, :, 2], bins=self.n_bins, range=(0, 256))[0]
                vec   = np.concatenate([h_h, h_s, h_v]).astype(np.float32)
                vec  /= (vec.sum() + 1e-6)
                hist[i, j] = vec

        # Upsample to (H, W, D)
        hist_t  = torch.from_numpy(hist).permute(2, 0, 1).unsqueeze(0)  # (1, D, G, G)
        up      = F.interpolate(hist_t, size=(H, W), mode="bilinear", align_corners=True)
        return up[0].permute(1, 2, 0)                                    # (H, W, D)


# ── L-2: HOG ──────────────────────────────────────────────────────────────────

@_register("hog")
class HOGFeature(LowLevelFeature):
    """
    L-2: Histogram of Oriented Gradients via skimage.

    Parameters
    ----------
    orientations : int  (default 9)
    pixels_per_cell : tuple (default (8, 8))
    cells_per_block : tuple (default (2, 2))
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
    ) -> None:
        self.orientations    = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def extract(self, image: Tensor) -> Tensor:
        from skimage.feature import hog

        _, H, W = image.shape
        img_np  = self._denorm(image)

        feat_map, hog_img = hog(
            img_np,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            channel_axis=-1,
            visualize=True,
        )
        # hog_img is (H, W) – replicate to match convention
        hog_t = torch.from_numpy(hog_img).float().unsqueeze(-1)          # (H, W, 1)
        return hog_t


# ── L-3: LBP ──────────────────────────────────────────────────────────────────

@_register("lbp")
class LBPFeature(LowLevelFeature):
    """
    L-3: Local Binary Patterns.

    Parameters
    ----------
    radius : int  (default 1)
    n_points : int  (default 8)
    """

    def __init__(self, radius: int = 1, n_points: int = 8) -> None:
        self.radius   = radius
        self.n_points = n_points

    def extract(self, image: Tensor) -> Tensor:
        from skimage.feature import local_binary_pattern
        import cv2

        _, H, W = image.shape
        img_np  = self._denorm(image)
        gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        lbp = local_binary_pattern(gray, self.n_points, self.radius, method="uniform")
        lbp_t = torch.from_numpy(lbp.astype(np.float32)).unsqueeze(-1)  # (H, W, 1)
        # Normalise to [0, 1]
        lbp_t = (lbp_t - lbp_t.min()) / (lbp_t.max() - lbp_t.min() + 1e-6)
        return lbp_t


# ── L-4: SLIC superpixel pooling ──────────────────────────────────────────────

@_register("slic_pool")
class SLICPool(LowLevelFeature):
    """
    L-4: Average-pool DINO features into SLIC superpixels, then broadcast back.

    Requires DINO pixel features to be provided via ``dino_features`` kwarg
    to :meth:`extract`.

    Parameters
    ----------
    n_segments : int  (default 200)
    compactness : float  (default 10)
    """

    def __init__(self, n_segments: int = 200, compactness: float = 10.0) -> None:
        self.n_segments  = n_segments
        self.compactness = compactness

    def extract(self, image: Tensor, dino_features: Optional[Tensor] = None) -> Tensor:
        from skimage.segmentation import slic

        _, H, W = image.shape
        img_np  = self._denorm(image)

        segments = slic(
            img_np, n_segments=self.n_segments,
            compactness=self.compactness, sigma=1, start_label=0,
        )

        if dino_features is None:
            # Return segment IDs as a one-hot-encoded feature (D = n_segments)
            n_sp = segments.max() + 1
            out  = np.zeros((H, W, n_sp), dtype=np.float32)
            for sp in range(n_sp):
                out[:, :, sp] = (segments == sp).astype(np.float32)
            return torch.from_numpy(out)

        # Average-pool DINO features per superpixel
        D       = dino_features.shape[-1]
        feat_np = dino_features.cpu().float().numpy()                    # (H, W, D)
        pooled  = np.zeros((H, W, D), dtype=np.float32)
        for sp in np.unique(segments):
            mask        = segments == sp
            pooled[mask] = feat_np[mask].mean(axis=0)
        return torch.from_numpy(pooled)


# ── L-5: Edge-weighted feature averaging ─────────────────────────────────────

@_register("edge_avg")
class EdgeWeightedAveraging(LowLevelFeature):
    """
    L-5: Compute edge magnitude as a 1-channel feature map.

    Returns the Canny/Sobel edge map as a (H, W, 1) feature that can be
    fused with DINO features via :func:`fuse_features`.
    """

    def extract(self, image: Tensor) -> Tensor:
        import cv2

        _, H, W = image.shape
        img_np  = self._denorm(image)
        gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        sobelx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag     = np.sqrt(sobelx ** 2 + sobely ** 2).astype(np.float32)
        mag    /= (mag.max() + 1e-6)

        return torch.from_numpy(mag).unsqueeze(-1)                       # (H, W, 1)


# ── L-6: SAM + DINO ───────────────────────────────────────────────────────────

@_register("sam_dino")
class SAMDINOFeature(LowLevelFeature):
    """
    L-6: Use SAM to generate mask proposals, average DINO embedding per
    proposal, then return a per-pixel proposal-ID feature map.

    Requires:
        - ``segment_anything`` package (``pip install segment-anything``).
        - A SAM ViT-H checkpoint.  Path can be set via the ``sam_checkpoint``
          constructor argument or the ``SAM_CHECKPOINT`` environment variable.

    If SAM is unavailable, falls back to edge-averaging (L-5).
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
    ) -> None:
        import os
        self.sam_checkpoint = sam_checkpoint or os.environ.get("SAM_CHECKPOINT", "")
        self.model_type     = model_type
        self._predictor     = None

    def _load_sam(self):
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as exc:
            raise ImportError(
                "SAM requires 'segment_anything': pip install segment-anything"
            ) from exc
        if not self.sam_checkpoint:
            raise FileNotFoundError(
                "SAM checkpoint not found. Set 'sam_checkpoint' or "
                "the SAM_CHECKPOINT environment variable."
            )
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        return SamAutomaticMaskGenerator(sam)

    def extract(
        self,
        image: Tensor,
        dino_features: Optional[Tensor] = None,
    ) -> Tensor:
        _, H, W = image.shape
        img_np  = self._denorm(image)

        try:
            gen = self._load_sam()
        except (ImportError, FileNotFoundError) as exc:
            warnings.warn(f"SAM unavailable ({exc}). Falling back to EdgeWeightedAveraging.")
            return EdgeWeightedAveraging().extract(image)

        masks = gen.generate(img_np)

        proposal_map = np.full((H, W), -1, dtype=np.int32)
        # Fill in order of descending mask area so smaller masks are on top
        for i, m in enumerate(sorted(masks, key=lambda x: -x["area"])):
            proposal_map[m["segmentation"]] = i

        if dino_features is not None:
            feat_np = dino_features.cpu().float().numpy()                # (H, W, D)
            D       = feat_np.shape[-1]
            out     = np.zeros((H, W, D), dtype=np.float32)
            for i in range(len(masks)):
                mask_px = proposal_map == i
                if mask_px.any():
                    out[mask_px] = feat_np[mask_px].mean(axis=0)
            return torch.from_numpy(out)

        # Fallback: return proposal IDs as normalised single channel
        n_proposals = max(len(masks), 1)
        feat        = (proposal_map / n_proposals).astype(np.float32)
        return torch.from_numpy(feat).unsqueeze(-1)                      # (H, W, 1)


# ── L-7: Watershed over DINO PCA ──────────────────────────────────────────────

@_register("watershed")
class WatershedDINO(LowLevelFeature):
    """
    L-7: Project DINO features to 3 PCA dimensions, treat them as an
    "image", compute the watershed, and return segment IDs as a feature.

    Requires DINO pixel features via ``dino_features`` kwarg.
    """

    def __init__(self, pca_dim: int = 3, compactness: float = 0.0) -> None:
        self.pca_dim     = pca_dim
        self.compactness = compactness

    def extract(
        self,
        image: Tensor,
        dino_features: Optional[Tensor] = None,
    ) -> Tensor:
        import cv2
        from skimage.segmentation import watershed
        from skimage.filters import rank
        from skimage.morphology import disk

        _, H, W = image.shape

        if dino_features is None:
            warnings.warn("WatershedDINO: no dino_features provided. Returning zeros.")
            return torch.zeros(H, W, 1)

        from sklearn.decomposition import PCA
        feat_np = dino_features.reshape(-1, dino_features.shape[-1]).cpu().numpy()
        n_comp  = min(self.pca_dim, feat_np.shape[1])
        pca     = PCA(n_components=n_comp)
        low     = pca.fit_transform(feat_np).reshape(H, W, n_comp)

        # Normalise to [0, 255]
        lo_min, lo_max = low.min(), low.max()
        low_u8 = ((low - lo_min) / (lo_max - lo_min + 1e-6) * 255).astype(np.uint8)

        # Use gradient magnitude as markers
        gray      = cv2.cvtColor(low_u8[:, :, :3] if n_comp >= 3 else
                                 np.repeat(low_u8[:, :, :1], 3, axis=-1),
                                 cv2.COLOR_RGB2GRAY)
        grad      = rank.gradient(gray, disk(2))
        markers   = np.zeros_like(gray, dtype=np.int32)
        markers[grad < 10] = 1
        markers[grad > 50] = 2

        seg = watershed(grad, markers=markers, compactness=self.compactness)
        seg_norm = (seg / (seg.max() + 1e-6)).astype(np.float32)
        return torch.from_numpy(seg_norm).unsqueeze(-1)                  # (H, W, 1)


# ── L-8: Late fusion ──────────────────────────────────────────────────────────

@_register("late_fusion")
class LateFusion(LowLevelFeature):
    """
    L-8: Run two parallel feature extractors and fuse their outputs.

    Parameters
    ----------
    extractor_a : str
        Name of the first low-level feature extractor.
    extractor_b : str
        Name of the second.
    mode : str
        Fusion mode passed to :func:`fuse_features` (default ``"concat"``).
    kwargs_a : dict
    kwargs_b : dict
    """

    def __init__(
        self,
        extractor_a: str = "color_hist",
        extractor_b: str = "hog",
        mode: str = "concat",
        kwargs_a: Optional[Dict[str, Any]] = None,
        kwargs_b: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ext_a = get_lowlevel_feature(extractor_a, **(kwargs_a or {}))
        self.ext_b = get_lowlevel_feature(extractor_b, **(kwargs_b or {}))
        self.mode  = mode

    def extract(
        self,
        image: Tensor,
        dino_features: Optional[Tensor] = None,
    ) -> Tensor:
        fa = self.ext_a.extract(image)
        fb = self.ext_b.extract(image)

        # Ensure same spatial size
        H, W = fa.shape[:2]
        if fb.shape[:2] != (H, W):
            fb_t = fb.permute(2, 0, 1).unsqueeze(0).float()
            fb_t = F.interpolate(fb_t, size=(H, W), mode="bilinear", align_corners=True)
            fb   = fb_t[0].permute(1, 2, 0)

        return fuse_features(fa, fb, mode=self.mode)
