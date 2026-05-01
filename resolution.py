"""
resolution.py – Patch-feature to pixel-feature upsampling methods.

All methods inherit from :class:`ResolutionRecovery` and implement::

    upsample(patch_features, original_image=None) -> pixel_features

where:
    patch_features : Tensor (B, N, D)  – N = H_p * W_p patches, D = embed dim
    original_image : Tensor (B, 3, H, W) or None
    pixel_features : Tensor (B, H, W, D)

Registered methods
------------------
R-1  nearest     – Nearest-neighbour reshape + upsample
R-2  bilinear    – Bilinear interpolation
R-3  bicubic     – Bicubic interpolation
R-4  pca         – Feature-space PCA + bilinear + centroid back-projection
R-5  bilateral   – Guided joint bilateral upsampling using image edges
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {}


def _register(name: str):
    """Class decorator that adds the class to the global registry."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_resolution_method(name: str, **kwargs) -> "ResolutionRecovery":
    """
    Factory function.

    Parameters
    ----------
    name : str
        One of ``"nearest"``, ``"bilinear"``, ``"bicubic"``, ``"pca"``,
        ``"bilateral"``.
    **kwargs
        Forwarded to the class constructor.

    Returns
    -------
    ResolutionRecovery
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown resolution method {name!r}. "
                       f"Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


# ── Base class ────────────────────────────────────────────────────────────────

class ResolutionRecovery(ABC):
    """Abstract base class for resolution recovery methods."""

    @abstractmethod
    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        """
        Upsample patch-level features to pixel-level features.

        Parameters
        ----------
        patch_features : Tensor (B, N, D)
        original_image : Tensor (B, 3, H, W) or None
        target_size : (H, W)

        Returns
        -------
        pixel_features : Tensor (B, H, W, D)
        """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _reshape_to_spatial(patch_features: Tensor) -> Tensor:
        """
        Reshape (B, N, D) → (B, D, H_p, W_p) assuming square patch grid.
        """
        B, N, D = patch_features.shape
        H_p = W_p = int(math.isqrt(N))
        if H_p * W_p != N:
            raise ValueError(
                f"Number of patches {N} is not a perfect square. "
                "Non-square patch grids are not yet supported."
            )
        return patch_features.permute(0, 2, 1).reshape(B, D, H_p, W_p)

    @staticmethod
    def _interp(
        x: Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear",
    ) -> Tensor:
        """Thin wrapper around F.interpolate with align_corners where needed."""
        align = {"bilinear": True, "bicubic": True}.get(mode, None)
        kwargs: Dict[str, Any] = {"size": size, "mode": mode}
        if align is not None:
            kwargs["align_corners"] = align
        return F.interpolate(x.float(), **kwargs)


# ── R-0: No upsampling ────────────────────────────────────────────────────────

@_register("none")
class NoneResolution(ResolutionRecovery):
    """R-0: No upsampling. Returns patch grid as-is (e.g. 14x14)."""

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        # Just reshape to spatial and return. target_size is ignored.
        spatial = self._reshape_to_spatial(patch_features)  # (B, D, H_p, W_p)
        return spatial.permute(0, 2, 3, 1)                  # (B, H_p, W_p, D)


# ── R-1: Nearest neighbour ────────────────────────────────────────────────────

@_register("nearest")
class NearestNeighbour(ResolutionRecovery):
    """R-1: Nearest-neighbour upsampling from patch grid to pixel grid."""

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        spatial = self._reshape_to_spatial(patch_features)              # (B, D, H_p, W_p)
        upsampled = self._interp(spatial, target_size, mode="nearest")  # (B, D, H, W)
        return upsampled.permute(0, 2, 3, 1)                            # (B, H, W, D)


# ── R-2: Bilinear ─────────────────────────────────────────────────────────────

@_register("bilinear")
class BilinearInterpolation(ResolutionRecovery):
    """R-2: Bilinear interpolation upsampling."""

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        spatial   = self._reshape_to_spatial(patch_features)
        upsampled = self._interp(spatial, target_size, mode="bilinear")
        return upsampled.permute(0, 2, 3, 1)


# ── R-3: Bicubic ──────────────────────────────────────────────────────────────

@_register("bicubic")
class BicubicInterpolation(ResolutionRecovery):
    """R-3: Bicubic interpolation upsampling."""

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        spatial   = self._reshape_to_spatial(patch_features)
        upsampled = self._interp(spatial, target_size, mode="bicubic")
        return upsampled.permute(0, 2, 3, 1)


# ── R-4: PCA interpolation ────────────────────────────────────────────────────

@_register("pca")
class PCAInterpolation(ResolutionRecovery):
    """
    R-4: Project patch features to lower dimension *d* via PCA, bilinear
    upsample in that compressed space, then map each pixel back to the nearest
    cluster centroid in the original D-dimensional space.

    Parameters
    ----------
    pca_dim : int
        Target PCA dimensionality (default 32).
    n_centroids : int
        Number of cluster centroids used for back-projection (default 64).
    random_state : int
        Seed for KMeans used to build the centroids.
    """

    def __init__(
        self,
        pca_dim: int = 32,
        n_centroids: int = 64,
        random_state: int = 0,
    ) -> None:
        self.pca_dim       = pca_dim
        self.n_centroids   = n_centroids
        self.random_state  = random_state

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise ImportError("scikit-learn is required for PCAInterpolation.") from exc

        B, N, D = patch_features.shape
        device  = patch_features.device

        results = []
        for b in range(B):
            feats_np = patch_features[b].cpu().float().numpy()          # (N, D)

            # PCA
            n_comp = min(self.pca_dim, N, D)
            pca    = PCA(n_components=n_comp, random_state=self.random_state)
            low    = pca.fit_transform(feats_np)                        # (N, d)

            # Build centroids in full-D space
            n_c  = min(self.n_centroids, N)
            km   = MiniBatchKMeans(n_clusters=n_c, random_state=self.random_state, n_init=3)
            km.fit(feats_np)
            centroids = torch.from_numpy(km.cluster_centers_).float()   # (n_c, D)

            # Upsample low-d features
            H_p = W_p = int(math.isqrt(N))
            low_t = torch.from_numpy(low).float().reshape(1, H_p, W_p, n_comp)
            low_t = low_t.permute(0, 3, 1, 2)                          # (1, d, H_p, W_p)
            low_up = F.interpolate(
                low_t, size=target_size, mode="bilinear", align_corners=True
            )                                                            # (1, d, H, W)
            H, W = target_size
            low_up_np = low_up[0].permute(1, 2, 0).reshape(-1, n_comp).numpy()

            # Reconstruct D-dim via inverse PCA
            high_np = pca.inverse_transform(low_up_np)                  # (H*W, D)
            high_t  = torch.from_numpy(high_np).float()

            # Map to nearest centroid
            dists   = torch.cdist(high_t, centroids)                    # (H*W, n_c)
            labels  = dists.argmin(dim=1)                               # (H*W,)
            pixel_f = centroids[labels].reshape(H, W, D)                # (H, W, D)
            results.append(pixel_f)

        return torch.stack(results, dim=0).to(device)                   # (B, H, W, D)


# ── R-5: Guided joint bilateral upsampling ────────────────────────────────────

@_register("bilateral")
class GuidedBilateralUpsampling(ResolutionRecovery):
    """
    R-5: Guided joint bilateral upsampling using image edges as guidance.

    For each output pixel the upsampled value is a weighted sum of surrounding
    patch-grid values.  The weights combine spatial proximity (Gaussian) and
    intensity similarity of the *guide* image (edge map derived from RGB).

    Parameters
    ----------
    sigma_s : float
        Spatial bandwidth in patch-grid coordinates (default 1.0).
    sigma_r : float
        Range (intensity) bandwidth (default 0.1).
    radius : int
        Patch-grid neighbourhood radius for the bilateral filter (default 2).
    """

    def __init__(
        self,
        sigma_s: float = 1.0,
        sigma_r: float = 0.1,
        radius: int = 2,
    ) -> None:
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.radius  = radius

    def upsample(
        self,
        patch_features: Tensor,
        original_image: Optional[Tensor] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tensor:
        if original_image is None:
            warnings.warn(
                "GuidedBilateralUpsampling requires original_image for edge guidance. "
                "Falling back to bilinear interpolation."
            )
            return BilinearInterpolation().upsample(patch_features, target_size=target_size)

        B, N, D = patch_features.shape
        H, W    = target_size
        H_p = W_p = int(math.isqrt(N))
        device  = patch_features.device

        # Compute grayscale guide at patch resolution
        guide = original_image.mean(dim=1, keepdim=True).float()        # (B, 1, H_img, W_img)
        guide_p = F.adaptive_avg_pool2d(guide, (H_p, W_p))              # (B, 1, H_p, W_p)
        guide_px = F.interpolate(guide, size=(H, W), mode="bilinear", align_corners=True)

        spatial   = self._reshape_to_spatial(patch_features).float()    # (B, D, H_p, W_p)

        results = []
        for b in range(B):
            sp_b  = spatial[b]                    # (D, H_p, W_p)
            gp_b  = guide_p[b, 0]                 # (H_p, W_p)  patch guide
            gpx_b = guide_px[b, 0]                # (H, W)      pixel guide

            # For each pixel in target we compute coords in patch grid
            # and accumulate bilateral weights. For efficiency we loop on
            # a small neighbourhood in patch space.
            # Bilinear up first as base, then apply bilateral correction
            # using a simplified implementation.
            base = F.interpolate(
                sp_b.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True
            )[0]                                   # (D, H, W)

            # Edge map in patch space
            from torch.nn.functional import conv2d
            sobel_x = torch.tensor(
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                device=device
            ).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)
            gp_4 = gp_b.unsqueeze(0).unsqueeze(0)
            ex   = conv2d(gp_4, sobel_x, padding=1)[0, 0]
            ey   = conv2d(gp_4, sobel_y, padding=1)[0, 0]
            edge_mag = (ex ** 2 + ey ** 2).sqrt()  # (H_p, W_p)

            # Upsample edge map and blend
            edge_up = F.interpolate(
                edge_mag.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=True
            )[0, 0]

            # Weight: lower edge → stronger smoothing (bilateral-style)
            alpha = torch.exp(-edge_up / (self.sigma_r + 1e-6))        # (H, W)
            alpha = alpha.unsqueeze(0)                                   # (1, H, W)

            # Nearest-up (blocky but edge-preserving proxy)
            nn_up = F.interpolate(
                sp_b.unsqueeze(0), size=(H, W), mode="nearest"
            )[0]                                    # (D, H, W)

            # Blend: in smooth regions use bilinear, near edges use nearest
            blended = alpha * base + (1 - alpha) * nn_up               # (D, H, W)
            results.append(blended.permute(1, 2, 0))                    # (H, W, D)

        return torch.stack(results, dim=0).to(device)                   # (B, H, W, D)
