"""
postprocess.py – Predicted mask refinement methods.

All methods inherit from :class:`PostProcessor` and implement::

    process(pred_mask, image) -> refined_mask

Registered methods
------------------
P-1  morphology         – Morphological open/close
P-2  connected_comp     – Connected-component filtering (min area)
P-3  dense_crf          – Dense CRF (pydensecrf)
P-4  superpixel         – SLIC superpixel pooling + majority vote
P-5  bilateral_soft     – Bilateral filter on soft (probability) assignments
P-6  graph_cut          – Graph-cut refinement  (optional / stub)
P-7  tta                – Test-time augmentation ensembling
"""

from __future__ import annotations

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


def get_postprocessor(name: str, **kwargs) -> "PostProcessor":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown post-processor {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# ── Base class ────────────────────────────────────────────────────────────────

class PostProcessor(ABC):
    """Abstract base for mask post-processing."""

    @abstractmethod
    def process(
        self,
        pred_mask: Tensor,
        image: Tensor,
    ) -> Tensor:
        """
        Refine a predicted segmentation mask.

        Parameters
        ----------
        pred_mask : Tensor (H, W)  – integer class labels
        image     : Tensor (3, H, W) – normalised RGB image

        Returns
        -------
        refined_mask : Tensor (H, W)
        """

    def __call__(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        return self.process(pred_mask, image)

    @staticmethod
    def _to_numpy_uint8(mask: Tensor) -> np.ndarray:
        return mask.cpu().numpy().astype(np.uint8)

    @staticmethod
    def _denorm_image(image: Tensor) -> np.ndarray:
        """Undo ImageNet normalisation and return HxWx3 uint8 numpy array."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (image.cpu().float() * std + mean).clamp(0, 1)
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── P-1: Morphological open/close ────────────────────────────────────────────

@_register("morphology")
class MorphologyProcessor(PostProcessor):
    """
    P-1: Apply morphological opening then closing to the mask.

    Parameters
    ----------
    kernel_size : int
        Square kernel side length (default 5).
    n_open : int
        Number of erosion-dilation cycles for opening (default 1).
    n_close : int
        Number of dilation-erosion cycles for closing (default 1).
    """

    def __init__(
        self,
        kernel_size: int = 5,
        n_open: int = 1,
        n_close: int = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_open      = n_open
        self.n_close     = n_close

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        import cv2

        mask_np  = self._to_numpy_uint8(pred_mask)
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        classes  = np.unique(mask_np)

        result   = np.zeros_like(mask_np)
        for cls in classes:
            if cls == 255:
                continue
            binary = (mask_np == cls).astype(np.uint8)
            for _ in range(self.n_open):
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            for _ in range(self.n_close):
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            result[binary > 0] = cls

        return torch.from_numpy(result).long()


# ── P-2: Connected-component filtering ───────────────────────────────────────

@_register("connected_comp")
class ConnectedComponentFilter(PostProcessor):
    """
    P-2: Remove small connected components below a minimum area threshold.

    Small components are replaced by the most common label in their spatial
    neighbourhood (determined via dilation).

    Parameters
    ----------
    min_area : int
        Minimum number of pixels for a component to survive (default 200).
    """

    def __init__(self, min_area: int = 200) -> None:
        self.min_area = min_area

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        import cv2

        mask_np = self._to_numpy_uint8(pred_mask)
        result  = mask_np.copy()

        for cls in np.unique(mask_np):
            if cls == 255:
                continue
            binary = (mask_np == cls).astype(np.uint8)
            n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

            for comp_id in range(1, n_comp):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area < self.min_area:
                    # Replace small component with 0 (background)
                    result[labels == comp_id] = 0

        return torch.from_numpy(result).long()


# ── P-3: Dense CRF ───────────────────────────────────────────────────────────

@_register("dense_crf")
class DenseCRF(PostProcessor):
    """
    P-3: DenseCRF-based mask refinement using pydensecrf.

    Falls back gracefully if pydensecrf is not installed.

    Parameters
    ----------
    n_iter : int
        Number of CRF inference iterations (default 10).
    sxy_gaussian : float
        Spatial std for the Gaussian pairwise term (default 3).
    sxy_bilateral : float
        Spatial std for the bilateral term (default 80).
    srgb_bilateral : float
        Color std for the bilateral term (default 13).
    compat : float
        Compatibility weight (default 10).
    """

    def __init__(
        self,
        n_iter: int = 10,
        sxy_gaussian: float = 3.0,
        sxy_bilateral: float = 80.0,
        srgb_bilateral: float = 13.0,
        compat: float = 10.0,
    ) -> None:
        self.n_iter          = n_iter
        self.sxy_gaussian    = sxy_gaussian
        self.sxy_bilateral   = sxy_bilateral
        self.srgb_bilateral  = srgb_bilateral
        self.compat          = compat

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
        except ImportError:
            warnings.warn(
                "pydensecrf is not installed. Skipping DenseCRF post-processing. "
                "Install with: pip install pydensecrf"
            )
            return pred_mask

        H, W      = pred_mask.shape
        img_np    = self._denorm_image(image)               # (H, W, 3) uint8
        mask_np   = self._to_numpy_uint8(pred_mask)

        n_labels  = int(mask_np.max()) + 1
        if n_labels < 2:
            return pred_mask

        unary     = unary_from_labels(mask_np, n_labels, gt_prob=0.7, zero_unsure=False)
        d         = dcrf.DenseCRF2D(W, H, n_labels)
        d.setUnaryEnergy(unary)

        # Gaussian (spatial smoothness)
        d.addPairwiseGaussian(sxy=self.sxy_gaussian, compat=3)

        # Bilateral (appearance + spatial)
        d.addPairwiseBilateral(
            sxy=self.sxy_bilateral,
            srgb=self.srgb_bilateral,
            rgbim=img_np,
            compat=self.compat,
        )

        Q = d.inference(self.n_iter)
        refined = np.argmax(Q, axis=0).reshape(H, W)
        return torch.from_numpy(refined).long()


# ── P-4: Superpixel pooling ───────────────────────────────────────────────────

@_register("superpixel")
class SuperpixelPooling(PostProcessor):
    """
    P-4: SLIC superpixel segmentation + majority vote per superpixel.

    Parameters
    ----------
    n_segments : int
        Approximate number of SLIC superpixels (default 200).
    compactness : float
        SLIC compactness parameter (default 10).
    """

    def __init__(self, n_segments: int = 200, compactness: float = 10.0) -> None:
        self.n_segments  = n_segments
        self.compactness = compactness

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        from skimage.segmentation import slic

        img_np  = self._denorm_image(image)                 # (H, W, 3)
        mask_np = pred_mask.cpu().numpy().astype(int)

        segments = slic(
            img_np,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=1,
            start_label=0,
        )

        result = mask_np.copy()
        for sp_id in np.unique(segments):
            sp_pixels = segments == sp_id
            labels_in_sp = mask_np[sp_pixels]
            valid        = labels_in_sp[labels_in_sp != 255]
            if valid.size > 0:
                vals, cnts           = np.unique(valid, return_counts=True)
                result[sp_pixels]    = vals[cnts.argmax()]

        return torch.from_numpy(result).long()


# ── P-5: Bilateral filter on soft assignments ────────────────────────────────

@_register("bilateral_soft")
class BilateralSoftFilter(PostProcessor):
    """
    P-5: Apply a joint bilateral filter to per-class probability maps, then
    take argmax.  Operates on soft (float) logits if provided; otherwise uses
    one-hot from the hard mask.

    Parameters
    ----------
    d : int
        Pixel neighbourhood diameter (default 9).
    sigma_color : float
        Bilateral filter color std (default 75).
    sigma_space : float
        Bilateral filter spatial std (default 75).
    """

    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
    ) -> None:
        self.d            = d
        self.sigma_color  = sigma_color
        self.sigma_space  = sigma_space

    def process(
        self,
        pred_mask: Tensor,
        image: Tensor,
        soft_mask: Optional[Tensor] = None,     # (C, H, W) float logits
    ) -> Tensor:
        import cv2

        img_np   = self._denorm_image(image)    # (H, W, 3)
        mask_np  = pred_mask.cpu().numpy().astype(int)
        n_cls    = int(mask_np.max()) + 1

        if soft_mask is not None:
            prob_np = F.softmax(soft_mask, dim=0).cpu().numpy()  # (C, H, W)
        else:
            H, W    = mask_np.shape
            prob_np = np.zeros((n_cls, H, W), dtype=np.float32)
            for c in range(n_cls):
                prob_np[c] = (mask_np == c).astype(np.float32)

        filtered = np.zeros_like(prob_np)
        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        for c in range(prob_np.shape[0]):
            ch = (prob_np[c] * 255).astype(np.float32)
            filt = cv2.bilateralFilter(
                ch, self.d, self.sigma_color, self.sigma_space
            )
            filtered[c] = filt / 255.0

        refined = filtered.argmax(axis=0)
        return torch.from_numpy(refined).long()


# ── P-6: Graph-cut refinement (optional stub) ─────────────────────────────────

@_register("graph_cut")
class GraphCutRefinement(PostProcessor):
    """
    P-6: Graph-cut based refinement.

    This is a stub implementation. For a full implementation integrate
    PyMaxflow or similar.  Raises NotImplementedError with a TODO message
    when called.
    """

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        raise NotImplementedError(
            "TODO (P-6): Implement graph-cut refinement using PyMaxflow or GCO. "
            "See https://github.com/pmneila/PyMaxflow for reference."
        )


# ── P-7: Test-time augmentation ensembling ────────────────────────────────────

@_register("tta")
class TTAEnsembling(PostProcessor):
    """
    P-7: Test-time augmentation (TTA) ensembling.

    Applies a sequence of geometric augmentations (horizontal flip, vertical
    flip, 90°/180°/270° rotation) to the image, runs the provided *pipeline_fn*
    on each, averages soft probability maps, then takes argmax.

    Parameters
    ----------
    pipeline_fn : callable
        A function ``f(image: Tensor) -> Tensor (H, W)`` that runs the full
        segmentation pipeline and returns a hard mask.  Must be set before
        calling :meth:`process`.
    n_classes : int
        Total number of classes (for one-hot soft averaging).
    augmentations : list[str]
        Which augmentations to use.  Supported: ``"hflip"``, ``"vflip"``,
        ``"rot90"``, ``"rot180"``, ``"rot270"``.
    """

    def __init__(
        self,
        pipeline_fn: Optional[Callable] = None,
        n_classes: int = 2,
        augmentations: Optional[List[str]] = None,
    ) -> None:
        self.pipeline_fn  = pipeline_fn
        self.n_classes    = n_classes
        self.augmentations = augmentations or ["hflip", "vflip", "rot90", "rot180", "rot270"]

    def _augment(self, img: Tensor, aug: str) -> Tensor:
        if aug == "hflip":   return img.flip(-1)
        if aug == "vflip":   return img.flip(-2)
        if aug == "rot90":   return img.rot90(1, [-2, -1])
        if aug == "rot180":  return img.rot90(2, [-2, -1])
        if aug == "rot270":  return img.rot90(3, [-2, -1])
        return img

    def _deaugment_mask(self, mask: Tensor, aug: str) -> Tensor:
        if aug == "hflip":   return mask.flip(-1)
        if aug == "vflip":   return mask.flip(-2)
        if aug == "rot90":   return mask.rot90(-1, [-2, -1])
        if aug == "rot180":  return mask.rot90(-2, [-2, -1])
        if aug == "rot270":  return mask.rot90(-3, [-2, -1])
        return mask

    def process(self, pred_mask: Tensor, image: Tensor) -> Tensor:
        if self.pipeline_fn is None:
            warnings.warn(
                "TTAEnsembling.pipeline_fn is None. "
                "Set it to a callable before using TTA. Returning original mask."
            )
            return pred_mask

        H, W   = pred_mask.shape
        n_cls  = self.n_classes
        accum  = torch.zeros(n_cls, H, W, dtype=torch.float32)

        # Original
        base_oh = torch.zeros(n_cls, H, W)
        for c in range(n_cls):
            base_oh[c] = (pred_mask == c).float()
        accum += base_oh

        for aug in self.augmentations:
            aug_img  = self._augment(image.unsqueeze(0), aug).squeeze(0)
            aug_pred = self.pipeline_fn(aug_img)                        # (H, W)
            aug_pred = self._deaugment_mask(aug_pred.unsqueeze(0), aug).squeeze(0)
            oh       = torch.zeros(n_cls, H, W)
            for c in range(n_cls):
                oh[c] = (aug_pred == c).float()
            accum += oh

        return accum.argmax(dim=0).long()
