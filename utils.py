"""
utils.py – General helper functions for dinov3_seg.

Functions
---------
load_dino_model       – Load a DINOv2 model from torch.hub
save_results          – Save predicted mask and optionally an overlay image
load_checkpoint       – Load pipeline state from disk
save_checkpoint       – Save pipeline state to disk
visualize_segmentation – Create a colour overlay of pred vs GT
denormalize_image     – Undo ImageNet normalisation
set_seed              – Set global random seeds
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── ImageNet normalisation ────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image: Tensor) -> np.ndarray:
    """
    Undo ImageNet normalisation and return a HxWx3 uint8 NumPy array.

    Parameters
    ----------
    image : Tensor (3, H, W) – normalised float

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    img = (image.cpu().float() * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── DINO model loading ────────────────────────────────────────────────────────

def load_dino_model(
    model_name: str = "dinov2_vitb14",
    device: Union[str, torch.device] = "cpu",
    cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load a DINOv2 model via torch.hub.

    Parameters
    ----------
    model_name : str
        e.g. ``"dinov2_vitb14"``, ``"dinov2_vitl14"``.
    device : str | torch.device
    cache_dir : str, optional
        Override torch.hub cache directory.

    Returns
    -------
    model : torch.nn.Module (eval mode)
    """
    if cache_dir is not None:
        os.environ["TORCH_HOME"] = cache_dir

    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def extract_patch_tokens(
    model: torch.nn.Module,
    image: Tensor,
    device: Union[str, torch.device] = "cpu",
) -> Tensor:
    """
    Extract patch tokens from a DINOv2 model for a single or batched image.

    Parameters
    ----------
    model  : DINOv2 torch.nn.Module
    image  : Tensor (3, H, W) or (B, 3, H, W)
    device : str | torch.device

    Returns
    -------
    Tensor (B, N_patches, D) – patch tokens (CLS token excluded)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    out   = model.forward_features(image)           # dict or Tensor
    if isinstance(out, dict):
        tokens = out.get("x_norm_patchtokens", out.get("patch_tokens"))
        if tokens is None:
            raise KeyError(
                "Unexpected DINOv2 output keys. "
                f"Got: {list(out.keys())}"
            )
    else:
        # Older hub versions return (B, N+1, D) with CLS at index 0
        tokens = out[:, 1:, :]
    return tokens.cpu()


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save an arbitrary dict to disk as a PyTorch checkpoint.

    Parameters
    ----------
    state : dict  – e.g. {"config": ..., "cluster_centroids": ...}
    path  : str | Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    print(f"[utils] Checkpoint saved to {path}")


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path         : str | Path
    map_location : str

    Returns
    -------
    state : dict
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(str(path), map_location=map_location)
    print(f"[utils] Checkpoint loaded from {path}")
    return state


# ── Saving results ────────────────────────────────────────────────────────────

def save_results(
    pred_mask: Tensor,
    output_dir: Union[str, Path],
    filename: str,
    image: Optional[Tensor] = None,
    gt_mask: Optional[Tensor] = None,
    save_overlay: bool = True,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
) -> None:
    """
    Save prediction and optional overlay to ``output_dir/<filename>``.

    Parameters
    ----------
    pred_mask  : Tensor (H, W)
    output_dir : str | Path
    filename   : str  – base name without extension
    image      : Tensor (3, H, W) optional, used for overlay
    gt_mask    : Tensor (H, W) optional, saved alongside pred
    save_overlay : bool  – if True and image is provided, save a colour overlay
    colormap   : list of (R, G, B) tuples for class colours
    """
    try:
        import cv2
    except ImportError:
        print("[utils] cv2 not available; skipping result saving.")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{filename}_pred.png"), pred_np * 128)

    if gt_mask is not None:
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{filename}_gt.png"), gt_np * 128)

    if save_overlay and image is not None:
        overlay = visualize_segmentation(image, pred_mask, gt_mask, colormap)
        cv2.imwrite(
            str(out_dir / f"{filename}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_segmentation(
    image: Tensor,
    pred_mask: Tensor,
    gt_mask: Optional[Tensor] = None,
    colormap: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a colour overlay of predicted (and optionally GT) masks.

    Parameters
    ----------
    image    : Tensor (3, H, W)
    pred_mask : Tensor (H, W)
    gt_mask  : Tensor (H, W) optional
    colormap : list of RGB tuples; auto-generated if None
    alpha    : blend weight for overlay

    Returns
    -------
    np.ndarray (H, W, 3) uint8 – side-by-side if gt_mask provided
    """
    img_np   = denormalize_image(image)
    pred_np  = pred_mask.cpu().numpy().astype(int)
    n_cls    = int(pred_np.max()) + 1

    if colormap is None:
        rng = np.random.default_rng(0)
        colormap = [tuple(rng.integers(50, 255, 3).tolist()) for _ in range(max(n_cls, 10))]

    def _overlay(base: np.ndarray, seg: np.ndarray) -> np.ndarray:
        canvas = base.copy()
        for c in range(len(colormap)):
            mask = seg == c
            if mask.any():
                canvas[mask] = (
                    (1 - alpha) * base[mask] + alpha * np.array(colormap[c])
                ).clip(0, 255).astype(np.uint8)
        return canvas

    pred_overlay = _overlay(img_np, pred_np)

    if gt_mask is not None:
        gt_np        = gt_mask.cpu().numpy().astype(int)
        gt_overlay   = _overlay(img_np, gt_np)
        return np.concatenate([img_np, pred_overlay, gt_overlay], axis=1)

    return np.concatenate([img_np, pred_overlay], axis=1)


# ── Metrics pretty-printing ───────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Print a formatted metrics dict to stdout."""
    width = max(len(k) for k in metrics) + 4
    print(f"\n{'─' * (width + 12)}")
    print(f"  {title}")
    print(f"{'─' * (width + 12)}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<{width}} {v:.4f}")
        else:
            print(f"  {k:<{width}} {v}")
    print(f"{'─' * (width + 12)}\n")


# ── Config serialisation ──────────────────────────────────────────────────────

def save_config(config, path: Union[str, Path]) -> None:
    """Save a PipelineConfig to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"[utils] Config saved to {path}")


def load_config(path: Union[str, Path]):
    """Load a PipelineConfig from a JSON file."""
    from .config import PipelineConfig

    with open(path) as f:
        d = json.load(f)
    return PipelineConfig.from_dict(d)
