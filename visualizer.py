"""
visualizer.py – Rich intermediate visualisations for the DINOv3 segmentation pipeline.

Saves:
  • PCA feature maps (RGB overview + individual components)
  • Cluster label maps with colour-coded overlays
  • Low-level feature maps (one panel per channel)
  • Cosine-similarity map (pixel vs mask_embedding)
  • Side-by-side qualitative grid (image | GT | pred | clusters | PCA)
  • Per-experiment metric bar charts
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor

# ── matplotlib (lazy import so the module loads on headless machines) ──────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ── colour helpers ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denorm(img_tensor: Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normed → HxWx3 uint8."""
    img = img_tensor.cpu().float().permute(1, 2, 0).numpy()
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def labels_to_rgb(labels: np.ndarray, cmap_name: str = "tab20") -> np.ndarray:
    """Integer label map → HxWx3 uint8 using a discrete colormap."""
    import matplotlib.cm as cm
    n = int(labels.max()) + 1 if labels.size else 1
    cmap = cm.get_cmap(cmap_name, max(n, 2))
    ids = (labels % max(n, 2)).astype(int)
    rgb = (cmap(ids)[..., :3] * 255).astype(np.uint8)
    return rgb


def feat_to_rgb_pca(feats: np.ndarray, n_components: int = 3) -> np.ndarray:
    """(H,W,D) feature map → HxWx3 uint8 via PCA."""
    from sklearn.decomposition import PCA
    H, W, D = feats.shape
    X = feats.reshape(-1, D)
    nc = min(n_components, D, X.shape[0] - 1)
    pca = PCA(n_components=nc)
    low = pca.fit_transform(X)          # (H*W, nc)
    # normalise each component to [0,1]
    lo, hi = low.min(0, keepdims=True), low.max(0, keepdims=True)
    low = (low - lo) / (hi - lo + 1e-6)
    # pad to 3 channels
    if nc < 3:
        pad = np.zeros((low.shape[0], 3 - nc), dtype=np.float32)
        low = np.concatenate([low, pad], axis=1)
    return (low[:, :3].reshape(H, W, 3) * 255).astype(np.uint8)


def _scalar_to_rgb(arr: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """(H,W) float → HxWx3 uint8."""
    import matplotlib.cm as cm
    arr = arr.astype(float)
    lo, hi = arr.min(), arr.max()
    normed = (arr - lo) / (hi - lo + 1e-6)
    return (cm.get_cmap(cmap_name)(normed)[..., :3] * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Core visualiser class
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Collects intermediate stage outputs for a single image and writes
    organised visualisation figures to disk.

    Parameters
    ----------
    out_dir : Path
        Experiment output directory. Sub-directories are created automatically.
    dpi : int
    cmap_clusters : str
    cmap_features : str
    cmap_lowlevel : str
    """

    def __init__(
        self,
        out_dir: Path,
        dpi: int = 120,
        cmap_clusters: str = "tab20",
        cmap_features: str = "viridis",
        cmap_lowlevel: str = "plasma",
    ) -> None:
        self.out_dir      = Path(out_dir)
        self.dpi          = dpi
        self.cmap_clusters = cmap_clusters
        self.cmap_features = cmap_features
        self.cmap_lowlevel = cmap_lowlevel

        for sub in ("pca", "clusters", "lowlevel", "cosine", "grids", "lowlevel_channels"):
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── public entry-point ────────────────────────────────────────────────────

    def save_all(
        self,
        stem: str,
        image: Tensor,
        gt_mask: Tensor,
        pred_mask: Tensor,
        stages: Dict[str, Any],
        lowlevel_cfg: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save every visualisation for a single image.

        Parameters
        ----------
        stem        : filename stem (used as prefix for output files)
        image       : (3, H, W) ImageNet-normalised
        gt_mask     : (H, W) int
        pred_mask   : (H, W) int
        stages      : dict returned by pipeline.run(return_stages=True)
        lowlevel_cfg: low-level config dict (for channel labels)
        metrics     : per-image metric dict
        """
        img_np  = denorm(image)                              # HxWx3
        gt_np   = gt_mask.cpu().numpy().astype(int)
        pred_np = pred_mask.cpu().numpy().astype(int)

        pixel_feats     = stages.get("pixel_features")      # (H,W,D)
        fused_feats     = stages.get("fused_features", pixel_feats)
        cluster_labels  = stages.get("cluster_labels")      # (H,W)
        mask_embedding  = stages.get("mask_embedding")      # (D,)

        # 1. PCA of pixel features
        if pixel_feats is not None:
            self._save_pca(stem, pixel_feats, tag="dino")
        if fused_feats is not None and fused_feats is not pixel_feats:
            self._save_pca(stem, fused_feats, tag="fused")

        # 2. Cluster map
        if cluster_labels is not None:
            self._save_clusters(stem, img_np, cluster_labels.cpu().numpy())

        # 3. Low-level feature channels (if present in stages)
        if "fused_features" in stages and pixel_feats is not None:
            # low-level slice = fused[..., D_dino:]
            D_dino = pixel_feats.shape[-1]
            ll_feats = stages["fused_features"][..., D_dino:].cpu().numpy()
            if ll_feats.shape[-1] > 0:
                self._save_lowlevel_channels(stem, ll_feats, lowlevel_cfg)

        # 4. Cosine similarity map
        if mask_embedding is not None and pixel_feats is not None:
            self._save_cosine_map(stem, pixel_feats, mask_embedding)

        # 5. Master qualitative grid
        self._save_grid(stem, img_np, gt_np, pred_np,
                        cluster_labels, pixel_feats, metrics)

    # ── PCA ───────────────────────────────────────────────────────────────────

    def _save_pca(
        self,
        stem: str,
        feats: Tensor,
        tag: str = "dino",
        n_components: int = 6,
    ) -> None:
        plt = _plt()
        import matplotlib.pyplot as mpl_plt

        feat_np = feats.cpu().float().numpy()  # (H,W,D)
        H, W, D = feat_np.shape
        X = feat_np.reshape(-1, D)

        from sklearn.decomposition import PCA
        nc = min(n_components, D, X.shape[0] - 1)
        pca = PCA(n_components=nc)
        components = pca.fit_transform(X).reshape(H, W, nc)  # (H,W,nc)

        # RGB overview (first 3 components)
        rgb_pca = feat_to_rgb_pca(feat_np, n_components=3)

        ncols = min(nc + 1, 7)
        fig, axes = mpl_plt.subplots(1, ncols, figsize=(3 * ncols, 3), dpi=self.dpi)
        if ncols == 1:
            axes = [axes]

        axes[0].imshow(rgb_pca)
        axes[0].set_title("PCA RGB (PC 1-3)", fontsize=7)
        axes[0].axis("off")

        for i in range(1, ncols):
            comp = components[:, :, i - 1]
            lo, hi = comp.min(), comp.max()
            normed = (comp - lo) / (hi - lo + 1e-6)
            axes[i].imshow(normed, cmap=self.cmap_features, vmin=0, vmax=1)
            axes[i].set_title(
                f"PC{i}  ({pca.explained_variance_ratio_[i-1]:.1%})", fontsize=7
            )
            axes[i].axis("off")

        fig.suptitle(f"PCA – {tag} | {stem}", fontsize=8)
        fig.tight_layout()
        out = self.out_dir / "pca" / f"{stem}_{tag}_pca.png"
        fig.savefig(out, bbox_inches="tight")
        mpl_plt.close(fig)

    # ── Clusters ──────────────────────────────────────────────────────────────

    def _save_clusters(
        self,
        stem: str,
        img_np: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        plt = _plt()
        import matplotlib.pyplot as mpl_plt

        cluster_rgb = labels_to_rgb(cluster_labels, self.cmap_clusters)
        H, W = cluster_labels.shape
        K = int(cluster_labels.max()) + 1

        # Overlay: blend cluster colours with original image
        alpha = 0.55
        overlay = (alpha * cluster_rgb + (1 - alpha) * img_np).astype(np.uint8)

        fig, axes = mpl_plt.subplots(1, 3, figsize=(10, 3.5), dpi=self.dpi)
        axes[0].imshow(img_np);       axes[0].set_title("Image");        axes[0].axis("off")
        axes[1].imshow(cluster_rgb);  axes[1].set_title(f"Clusters (K={K})"); axes[1].axis("off")
        axes[2].imshow(overlay);      axes[2].set_title("Overlay");       axes[2].axis("off")

        # Per-cluster pixel count bar
        fig2, ax2 = mpl_plt.subplots(figsize=(max(4, K * 0.4), 2.5), dpi=self.dpi)
        ids, counts = np.unique(cluster_labels, return_counts=True)
        import matplotlib.cm as cm
        colours = [cm.get_cmap(self.cmap_clusters, max(K, 2))(int(i) % max(K, 2)) for i in ids]
        ax2.bar(ids, counts, color=colours, edgecolor="k", linewidth=0.3)
        ax2.set_xlabel("Cluster ID"); ax2.set_ylabel("Pixel count")
        ax2.set_title(f"Cluster sizes – {stem}")
        fig2.tight_layout()

        fig.suptitle(f"Clusters – {stem}", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out_dir / "clusters" / f"{stem}_clusters.png",  bbox_inches="tight")
        fig2.savefig(self.out_dir / "clusters" / f"{stem}_cluster_hist.png", bbox_inches="tight")
        mpl_plt.close(fig); mpl_plt.close(fig2)

    # ── Low-level feature channels ─────────────────────────────────────────────

    def _save_lowlevel_channels(
        self,
        stem: str,
        ll_feats: np.ndarray,          # (H, W, D_low)
        cfg: Optional[Dict] = None,
    ) -> None:
        plt = _plt()
        import matplotlib.pyplot as mpl_plt

        H, W, D = ll_feats.shape
        method = (cfg or {}).get("method", "lowlevel")
        ncols  = min(D, 12)
        nrows  = int(np.ceil(D / ncols))

        fig, axes = mpl_plt.subplots(nrows, ncols,
                                      figsize=(2.5 * ncols, 2.5 * nrows),
                                      dpi=self.dpi, squeeze=False)
        for ch in range(D):
            r, c = divmod(ch, ncols)
            arr  = ll_feats[:, :, ch]
            axes[r, c].imshow(arr, cmap=self.cmap_lowlevel)
            axes[r, c].set_title(f"ch {ch}", fontsize=6)
            axes[r, c].axis("off")
        # hide spare axes
        for ch in range(D, nrows * ncols):
            r, c = divmod(ch, ncols)
            axes[r, c].axis("off")

        fig.suptitle(f"Low-level ({method}) channels – {stem}", fontsize=8)
        fig.tight_layout()
        out = self.out_dir / "lowlevel_channels" / f"{stem}_{method}_channels.png"
        fig.savefig(out, bbox_inches="tight")
        mpl_plt.close(fig)

        # Also save composite (mean across channels)
        mean_map = ll_feats.mean(axis=-1)
        fig2, ax2 = mpl_plt.subplots(figsize=(4, 3.5), dpi=self.dpi)
        im = ax2.imshow(mean_map, cmap=self.cmap_lowlevel)
        ax2.set_title(f"Low-level mean ({method}) – {stem}", fontsize=7)
        ax2.axis("off")
        import matplotlib.pyplot as mpl_plt2
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(
            self.out_dir / "lowlevel" / f"{stem}_{method}_mean.png",
            bbox_inches="tight",
        )
        mpl_plt.close(fig2)

    # ── Cosine similarity map ──────────────────────────────────────────────────

    def _save_cosine_map(
        self,
        stem: str,
        pixel_feats: Tensor,
        mask_embedding: Tensor,
    ) -> None:
        plt = _plt()
        import matplotlib.pyplot as mpl_plt

        feats = pixel_feats.cpu().float()           # (H, W, D)
        me    = mask_embedding.cpu().float()
        me    = me / (me.norm() + 1e-8)

        H, W, D = feats.shape
        flat  = feats.reshape(-1, D)
        norms = flat.norm(dim=1, keepdim=True)
        flat_n = flat / (norms + 1e-8)
        cosine = (flat_n @ me).reshape(H, W).numpy()

        fig, ax = mpl_plt.subplots(figsize=(4.5, 3.5), dpi=self.dpi)
        im = ax.imshow(cosine, cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Cosine sim to mask_emb – {stem}", fontsize=7)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.out_dir / "cosine" / f"{stem}_cosine.png", bbox_inches="tight")
        mpl_plt.close(fig)

    # ── Master qualitative grid ────────────────────────────────────────────────

    def _save_grid(
        self,
        stem: str,
        img_np: np.ndarray,
        gt_np: np.ndarray,
        pred_np: np.ndarray,
        cluster_labels: Optional[Tensor],
        pixel_feats: Optional[Tensor],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        plt = _plt()
        import matplotlib.pyplot as mpl_plt

        panels = [("Image", img_np, None),
                  ("GT mask", labels_to_rgb(gt_np.clip(0, 1)), None),
                  ("Prediction", labels_to_rgb(pred_np.clip(0, 1)), None)]

        if cluster_labels is not None:
            cl_np = cluster_labels.cpu().numpy()
            panels.append(("Clusters", labels_to_rgb(cl_np, self.cmap_clusters), None))

        if pixel_feats is not None:
            pca_rgb = feat_to_rgb_pca(pixel_feats.cpu().numpy(), n_components=3)
            panels.append(("DINO PCA", pca_rgb, None))

        # Overlay pred on image
        alpha = 0.5
        pred_rgb = labels_to_rgb(pred_np.clip(0, 1))
        overlay  = (alpha * pred_rgb + (1 - alpha) * img_np).astype(np.uint8)
        panels.append(("Pred overlay", overlay, None))

        N = len(panels)
        fig, axes = mpl_plt.subplots(1, N, figsize=(3.2 * N, 3.5), dpi=self.dpi)
        for ax, (title, arr, _) in zip(axes, panels):
            ax.imshow(arr)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

        # Metrics subtitle
        if metrics:
            metric_str = "  |  ".join(
                f"{k}={v:.3f}" for k, v in metrics.items()
                if isinstance(v, float) and not np.isnan(v)
            )
            fig.suptitle(f"{stem}\n{metric_str}", fontsize=7)
        else:
            fig.suptitle(stem, fontsize=8)

        fig.tight_layout()
        fig.savefig(self.out_dir / "grids" / f"{stem}_grid.png", bbox_inches="tight")
        mpl_plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment-level summary plots
# ══════════════════════════════════════════════════════════════════════════════

def save_experiment_summary(
    results: List[Dict],
    out_dir: Path,
    metric_key: str = "miou",
    dpi: int = 120,
) -> None:
    """
    Bar chart of all experiments ranked by `metric_key`.

    Parameters
    ----------
    results : list of dicts – each dict must contain 'exp_id' and the metric.
    out_dir : where to save the figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    valid = [r for r in results if metric_key in r and not np.isnan(r[metric_key])]
    if not valid:
        return

    valid.sort(key=lambda r: r[metric_key], reverse=True)
    exp_ids = [r["exp_id"] for r in valid]
    values  = [r[metric_key] for r in valid]

    fig, ax = plt.subplots(figsize=(max(8, len(exp_ids) * 0.35), 6), dpi=dpi)
    bars = ax.bar(range(len(exp_ids)), values, color="steelblue", edgecolor="k", linewidth=0.3)
    ax.set_xticks(range(len(exp_ids)))
    ax.set_xticklabels(exp_ids, rotation=90, fontsize=5)
    ax.set_ylabel(metric_key.upper())
    ax.set_title(f"All experiments ranked by {metric_key}")
    ax.grid(axis="y", alpha=0.3)

    # Annotate top-5
    for i, bar in enumerate(bars[:5]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{values[i]:.3f}", ha="center", va="bottom", fontsize=5, color="red")

    fig.tight_layout()
    out = Path(out_dir) / f"summary_{metric_key}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_axis_comparison(
    results: List[Dict],
    axis: str,
    out_dir: Path,
    metric_key: str = "miou",
    dpi: int = 120,
) -> None:
    """Box-plot comparing metric distributions for each value of one axis."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    groups: Dict[str, List[float]] = {}
    for r in results:
        val = r.get(axis, "?")
        m   = r.get(metric_key, float("nan"))
        if not np.isnan(m):
            groups.setdefault(str(val), []).append(m)

    if not groups:
        return

    labels = sorted(groups.keys())
    data   = [groups[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4), dpi=dpi)
    ax.boxplot(data, labels=labels, patch_artist=True,
               medianprops=dict(color="red", linewidth=2))
    ax.set_xlabel(axis)
    ax.set_ylabel(metric_key.upper())
    ax.set_title(f"{metric_key} by {axis}")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=7)

    fig.tight_layout()
    out = Path(out_dir) / f"axis_{axis}_{metric_key}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(
    results: List[Dict],
    axis_x: str,
    axis_y: str,
    out_dir: Path,
    metric_key: str = "miou",
    dpi: int = 120,
) -> None:
    """2-D heatmap of mean metric for two configuration axes."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    from collections import defaultdict
    cell: Dict[Tuple, List[float]] = defaultdict(list)
    for r in results:
        xv = str(r.get(axis_x, "?"))
        yv = str(r.get(axis_y, "?"))
        m  = r.get(metric_key, float("nan"))
        if not np.isnan(m):
            cell[(xv, yv)].append(m)

    if not cell:
        return

    xs = sorted({k[0] for k in cell})
    ys = sorted({k[1] for k in cell})
    mat = np.full((len(ys), len(xs)), np.nan)
    for (xv, yv), vals in cell.items():
        xi = xs.index(xv)
        yi = ys.index(yv)
        mat[yi, xi] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(5, len(xs) * 1.2), max(4, len(ys) * 0.8)), dpi=dpi)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label=metric_key)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=7)
    ax.set_xlabel(axis_x); ax.set_ylabel(axis_y)
    ax.set_title(f"Mean {metric_key}: {axis_y} vs {axis_x}")
    for yi in range(len(ys)):
        for xi in range(len(xs)):
            v = mat[yi, xi]
            if not np.isnan(v):
                ax.text(xi, yi, f"{v:.3f}", ha="center", va="center", fontsize=5, color="black")

    fig.tight_layout()
    out = Path(out_dir) / f"heatmap_{axis_y}_vs_{axis_x}_{metric_key}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
