"""
metrics.py – Evaluation metrics for segmentation.

Public functions
----------------
compute_miou            – mean Intersection over Union
compute_pixel_accuracy  – overall pixel accuracy
compute_dice            – Dice / F1 coefficient
compute_boundary_f1     – Boundary F1 at configurable trimap widths
compute_per_class       – per-class IoU, accuracy, Dice

Internal diagnostic metrics (debug=True)
-----------------------------------------
resolution_diagnostics  – boundary alignment, feature consistency, rank corr.
clustering_diagnostics  – purity, entropy, NMI, ARI, silhouette
assignment_diagnostics  – per-cluster label entropy, cross-image consistency
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

IGNORE_INDEX = 255


def _valid(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove ignore-index pixels."""
    mask = gt != IGNORE_INDEX
    return pred[mask], gt[mask]


def _to_np(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(int)


# ── mIoU ─────────────────────────────────────────────────────────────────────

def compute_miou(
    pred: Tensor,
    gt: Tensor,
    n_classes: int,
    ignore_index: int = IGNORE_INDEX,
    return_per_class: bool = False,
) -> float | Tuple[float, np.ndarray]:
    """
    Compute mean Intersection over Union.

    Parameters
    ----------
    pred, gt    : Tensor (H, W) or (B, H, W)
    n_classes   : int
    ignore_index : int
    return_per_class : bool

    Returns
    -------
    miou : float  (and optionally per_class_iou : np.ndarray)
    """
    pred_np = _to_np(pred.reshape(-1))
    gt_np   = _to_np(gt.reshape(-1))
    pred_np, gt_np = _valid(pred_np, gt_np)

    ious = []
    for c in range(n_classes):
        p = pred_np == c
        g = gt_np   == c
        inter = (p & g).sum()
        union = (p | g).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)

    ious_arr    = np.array(ious, dtype=np.float64)
    valid_ious  = ious_arr[~np.isnan(ious_arr)]
    miou_val    = float(valid_ious.mean()) if len(valid_ious) > 0 else 0.0

    if return_per_class:
        return miou_val, ious_arr
    return miou_val


# ── Pixel accuracy ────────────────────────────────────────────────────────────

def compute_pixel_accuracy(
    pred: Tensor,
    gt: Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Overall pixel accuracy (excluding ignore pixels)."""
    pred_np = _to_np(pred.reshape(-1))
    gt_np   = _to_np(gt.reshape(-1))
    p, g    = _valid(pred_np, gt_np)
    return float((p == g).mean()) if len(g) > 0 else 0.0


# ── Dice / F1 ─────────────────────────────────────────────────────────────────

def compute_dice(
    pred: Tensor,
    gt: Tensor,
    n_classes: int,
    ignore_index: int = IGNORE_INDEX,
    return_per_class: bool = False,
) -> float | Tuple[float, np.ndarray]:
    """
    Compute mean Dice / F1 coefficient.

    Returns mean Dice (float) and optionally per-class Dice array.
    """
    pred_np = _to_np(pred.reshape(-1))
    gt_np   = _to_np(gt.reshape(-1))
    pred_np, gt_np = _valid(pred_np, gt_np)

    dices = []
    for c in range(n_classes):
        p = pred_np == c
        g = gt_np   == c
        tp = (p & g).sum()
        fp = (p & ~g).sum()
        fn = (~p & g).sum()
        denom = 2 * tp + fp + fn
        if denom == 0:
            dices.append(float("nan"))
        else:
            dices.append(2 * tp / denom)

    arr   = np.array(dices, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    mean  = float(valid.mean()) if len(valid) > 0 else 0.0
    if return_per_class:
        return mean, arr
    return mean


# ── Boundary F1 ───────────────────────────────────────────────────────────────

def compute_boundary_f1(
    pred: Tensor,
    gt: Tensor,
    widths: Tuple[int, ...] = (2, 4, 8),
    ignore_index: int = IGNORE_INDEX,
) -> Dict[int, float]:
    """
    Boundary F1 score at multiple trimap widths.

    Uses cv2 distance transform to determine boundary region of each width.

    Parameters
    ----------
    pred, gt : Tensor (H, W)
    widths   : pixel widths for boundary band

    Returns
    -------
    dict {width: bf1_score}
    """
    try:
        import cv2
    except ImportError:
        warnings.warn("cv2 not available; boundary F1 cannot be computed.")
        return {w: float("nan") for w in widths}

    pred_np = _to_np(pred)
    gt_np   = _to_np(gt)

    def boundary_mask(mask: np.ndarray, width: int) -> np.ndarray:
        """Boolean mask of pixels within `width` pixels of any class boundary."""
        bin_mask = (mask > 0).astype(np.uint8)
        dist     = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
        dist_inv = cv2.distanceTransform(1 - bin_mask, cv2.DIST_L2, 3)
        return ((dist <= width) | (dist_inv <= width))

    results: Dict[int, float] = {}
    for w in widths:
        bnd_pred = boundary_mask(pred_np, w)
        bnd_gt   = boundary_mask(gt_np, w)

        # Restrict to valid pixels
        valid = gt_np != ignore_index
        bnd_pred &= valid
        bnd_gt   &= valid

        prec_denom = bnd_pred.sum()
        rec_denom  = bnd_gt.sum()

        if prec_denom == 0 or rec_denom == 0:
            results[w] = float("nan")
            continue

        precision = (bnd_pred & bnd_gt).sum() / prec_denom
        recall    = (bnd_pred & bnd_gt).sum() / rec_denom
        if precision + recall < 1e-6:
            results[w] = 0.0
        else:
            results[w] = float(2 * precision * recall / (precision + recall))

    return results


# ── Per-class metrics ─────────────────────────────────────────────────────────

def compute_per_class(
    pred: Tensor,
    gt: Tensor,
    n_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, np.ndarray]:
    """
    Return per-class IoU, pixel accuracy, and Dice as a dict of arrays
    of shape (n_classes,).
    """
    _, per_iou   = compute_miou(pred, gt, n_classes, ignore_index, return_per_class=True)
    _, per_dice  = compute_dice(pred, gt, n_classes, ignore_index, return_per_class=True)

    pred_np = _to_np(pred.reshape(-1))
    gt_np   = _to_np(gt.reshape(-1))
    pred_np, gt_np = _valid(pred_np, gt_np)

    per_acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        g_c = gt_np == c
        if g_c.sum() > 0:
            per_acc[c] = (pred_np[g_c] == c).mean()

    return {"iou": per_iou, "accuracy": per_acc, "dice": per_dice}


# ── Summary dict helper ───────────────────────────────────────────────────────

def compute_all_metrics(
    pred: Tensor,
    gt: Tensor,
    n_classes: int,
    boundary_widths: Tuple[int, ...] = (2, 4, 8),
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """
    Convenience wrapper that computes mIoU, pixel accuracy, Dice, and
    boundary F1 in one call.

    Returns
    -------
    dict with keys: ``miou``, ``pixel_acc``, ``dice``,
    ``bf1_w2``, ``bf1_w4``, ``bf1_w8``.
    """
    miou     = compute_miou(pred, gt, n_classes, ignore_index)
    pix_acc  = compute_pixel_accuracy(pred, gt, ignore_index)
    dice     = compute_dice(pred, gt, n_classes, ignore_index)
    bf1      = compute_boundary_f1(pred, gt, boundary_widths, ignore_index)

    result = {
        "miou":      miou,
        "pixel_acc": pix_acc,
        "dice":      dice,
    }
    for w, v in bf1.items():
        result[f"bf1_w{w}"] = v

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Internal diagnostic metrics
# ═══════════════════════════════════════════════════════════════════════════════

# ── Resolution diagnostics ────────────────────────────────────────────────────

def resolution_diagnostics(
    pixel_features: Tensor,
    pred_mask: Tensor,
    gt_mask: Tensor,
    image: Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    Internal diagnostic metrics for the resolution stage.

    Returns
    -------
    dict with keys: ``boundary_alignment``, ``feature_consistency``,
    ``rank_correlation_gt_edges``.
    """
    import cv2

    H, W   = gt_mask.shape
    gt_np  = _to_np(gt_mask)
    pred_np = _to_np(pred_mask)

    # 1. Boundary alignment: Fraction of GT boundary pixels where pred also has boundary
    gt_bin  = (gt_np > 0).astype(np.uint8)
    gt_edge = cv2.Laplacian(gt_bin, cv2.CV_64F)
    gt_bnd  = (np.abs(gt_edge) > 0)

    pred_bin  = (pred_np > 0).astype(np.uint8)
    pred_edge = cv2.Laplacian(pred_bin, cv2.CV_64F)
    pred_bnd  = (np.abs(pred_edge) > 0)

    ba = float((gt_bnd & pred_bnd).sum() / (gt_bnd.sum() + 1e-6))

    # 2. Feature consistency: mean cosine sim between adjacent pixels in same GT region
    feats = pixel_features.reshape(H, W, -1).cpu().float()
    right = feats[:, 1:, :]
    left  = feats[:, :-1, :]
    same  = torch.from_numpy(gt_np[:, 1:] == gt_np[:, :-1]).float()
    cos   = F.cosine_similarity(right.reshape(-1, feats.shape[-1]),
                                 left.reshape(-1, feats.shape[-1]), dim=-1)
    fc    = float((cos * same.reshape(-1)).sum() / (same.sum() + 1e-6))

    # 3. Rank correlation between predicted-feature magnitude and GT edge strength
    if image is not None:
        try:
            img_np = (image.cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            import cv2 as _cv2
            gray   = _cv2.cvtColor(img_np, _cv2.COLOR_RGB2GRAY)
            sobelx = _cv2.Sobel(gray, _cv2.CV_64F, 1, 0)
            sobely = _cv2.Sobel(gray, _cv2.CV_64F, 0, 1)
            edge   = np.sqrt(sobelx ** 2 + sobely ** 2).flatten()
            feat_mag = feats.norm(dim=-1).numpy().flatten()
            from scipy.stats import spearmanr
            rho, _ = spearmanr(feat_mag, edge)
            rc = float(rho)
        except Exception:
            rc = float("nan")
    else:
        rc = float("nan")

    return {
        "boundary_alignment":      ba,
        "feature_consistency":     fc,
        "rank_correlation_gt_edges": rc,
    }


# ── Clustering diagnostics ────────────────────────────────────────────────────

def clustering_diagnostics(
    cluster_labels: Tensor,
    gt_mask: Tensor,
    features: Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    Internal diagnostic metrics for the clustering stage.

    Returns
    -------
    dict with keys: ``purity``, ``entropy``, ``nmi``, ``ari``,
    ``silhouette`` (nan if features not provided).
    """
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
    )
    from scipy.stats import entropy as scipy_entropy

    cl_np = _to_np(cluster_labels.reshape(-1))
    gt_np = _to_np(gt_mask.reshape(-1))
    valid = gt_np != IGNORE_INDEX
    cl, gt = cl_np[valid], gt_np[valid]

    # Purity
    cluster_ids = np.unique(cl)
    purity_sum  = 0.0
    for c in cluster_ids:
        mask = cl == c
        if mask.sum() == 0:
            continue
        _, cnts = np.unique(gt[mask], return_counts=True)
        purity_sum += cnts.max()
    purity = purity_sum / (len(cl) + 1e-6)

    # Entropy (mean per-cluster entropy)
    entropies = []
    for c in cluster_ids:
        mask = cl == c
        _, cnts = np.unique(gt[mask], return_counts=True)
        probs = cnts / cnts.sum()
        entropies.append(float(scipy_entropy(probs)))
    mean_entropy = float(np.mean(entropies)) if entropies else float("nan")

    nmi = float(normalized_mutual_info_score(gt, cl))
    ari = float(adjusted_rand_score(gt, cl))

    sil = float("nan")
    if features is not None and len(np.unique(cl)) > 1:
        try:
            feats_np = features.reshape(-1, features.shape[-1]).cpu().float().numpy()
            feats_np = feats_np[valid]
            if len(feats_np) > 5000:
                idx = np.random.choice(len(feats_np), 5000, replace=False)
                feats_np, cl_sub = feats_np[idx], cl[idx]
            else:
                cl_sub = cl
            sil = float(silhouette_score(feats_np, cl_sub, sample_size=2000))
        except Exception:
            sil = float("nan")

    return {
        "purity":    purity,
        "entropy":   mean_entropy,
        "nmi":       nmi,
        "ari":       ari,
        "silhouette": sil,
    }


# ── Assignment diagnostics ────────────────────────────────────────────────────

def assignment_diagnostics(
    cluster_labels: Tensor,
    gt_mask: Tensor,
    pred_mask: Tensor,
    cross_image_history: Optional[Dict[int, list]] = None,
) -> Dict[str, float]:
    """
    Internal diagnostic metrics for the assignment stage.

    Parameters
    ----------
    cluster_labels : Tensor (H, W)
    gt_mask        : Tensor (H, W)
    pred_mask      : Tensor (H, W)
    cross_image_history : dict {cluster_id: [class_id, ...]} from
                          CrossImageConsistency, or None.

    Returns
    -------
    dict with keys: ``mean_cluster_label_entropy``,
    ``assignment_consistency`` (nan if history not provided).
    """
    from scipy.stats import entropy as scipy_entropy

    cl_np = _to_np(cluster_labels.reshape(-1))
    gt_np = _to_np(gt_mask.reshape(-1))
    valid = gt_np != IGNORE_INDEX
    cl, gt = cl_np[valid], gt_np[valid]

    entropies = []
    for c in np.unique(cl):
        mask = cl == c
        _, cnts = np.unique(gt[mask], return_counts=True)
        probs   = cnts / cnts.sum()
        entropies.append(float(scipy_entropy(probs)))

    mean_ent = float(np.mean(entropies)) if entropies else float("nan")

    consistency = float("nan")
    if cross_image_history is not None:
        scores = []
        for c_id, classes in cross_image_history.items():
            arr        = np.array(classes)
            _, cnts    = np.unique(arr, return_counts=True)
            scores.append(cnts.max() / cnts.sum())
        consistency = float(np.mean(scores)) if scores else float("nan")

    return {
        "mean_cluster_label_entropy": mean_ent,
        "assignment_consistency":     consistency,
    }
# ── Pretty-print metrics ──────────────────────────────────────────────────────

def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Pretty-print the metrics returned by compute_all_metrics().

    Example:
        m = compute_all_metrics(pred, gt, n_classes)
        print_metrics(m)
    """
    print("\n=== Segmentation Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")
    print("============================\n")