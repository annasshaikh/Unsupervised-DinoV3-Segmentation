"""
assignment.py – Mapping cluster IDs to semantic class labels.

All methods inherit from :class:`Assignment` and implement::

    assign(cluster_labels, gt_mask) -> mapping : dict {cluster_id: class_id}

Registered methods
------------------
A-1  majority_vote          – Majority-vote per cluster
A-2  weighted_majority      – Weighted majority vote (soft distance weighting)
A-3  hungarian              – Hungarian matching via IoU cost matrix
A-4  label_propagation      – Soft assignment with label propagation
A-5  abstention             – Majority vote + abstain if confidence < threshold
A-6  cross_image            – Per-image assignment + cross-image consistency metric
A-7  mask_embedding_cosine  – Assign foreground cluster via cosine sim to mask embedding
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, type] = {}


def _register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_assignment_method(name: str, **kwargs) -> "Assignment":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown assignment method {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# ── Base class ────────────────────────────────────────────────────────────────

class Assignment(ABC):
    """
    Abstract base class for cluster-to-class assignment.

    The ``ignore_index`` class is treated as unlabelled ground-truth.
    """

    IGNORE_INDEX: int = 255

    @abstractmethod
    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
    ) -> Dict[int, int]:
        """
        Compute a cluster → class mapping for a single image.

        Parameters
        ----------
        cluster_labels : Tensor (N,) or (H, W)
            Integer cluster indices.
        gt_mask : Tensor (H, W)
            Ground-truth class labels (binary 0/1 or multi-class).
            Pixels == ``IGNORE_INDEX`` are excluded.

        Returns
        -------
        mapping : dict {cluster_id: class_id}
        """

    def apply(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
    ) -> Tensor:
        """
        Convenience: apply the mapping to produce a predicted mask Tensor.
        The output shape matches cluster_labels.shape.
        """
        H, W     = cluster_labels.shape
        flat_cl  = cluster_labels.reshape(-1).long()
        mapping  = self.assign(flat_cl, gt_mask)
        pred     = torch.full((flat_cl.numel(),), self.IGNORE_INDEX, dtype=torch.long)
        for c_id, cls in mapping.items():
            pred[flat_cl == c_id] = cls
        return pred.reshape(H, W)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_both(
        cluster_labels: Tensor,
        gt_mask: Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (flat cluster ids, flat gt labels) arrays with ignore removed."""
        flat_cl  = cluster_labels.reshape(-1).cpu().numpy().astype(int)
        flat_gt  = gt_mask.reshape(-1).cpu().numpy().astype(int)
        valid    = flat_gt != Assignment.IGNORE_INDEX
        return flat_cl[valid], flat_gt[valid]


# ── A-1: Majority vote ────────────────────────────────────────────────────────

@_register("majority_vote")
class MajorityVote(Assignment):
    """A-1: Each cluster takes the most frequent GT label among its pixels."""

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        **kwargs: Any,
    ) -> Dict[int, int]:
        cl, gt = self._flatten_both(cluster_labels, gt_mask)
        mapping: Dict[int, int] = {}
        for c in np.unique(cl):
            mask_c   = cl == c
            gt_c     = gt[mask_c]
            if gt_c.size == 0:
                mapping[int(c)] = 0
                continue
            vals, cnts = np.unique(gt_c, return_counts=True)
            mapping[int(c)] = int(vals[cnts.argmax()])
        return mapping


# ── A-2: Weighted majority vote ───────────────────────────────────────────────

@_register("weighted_majority")
class WeightedMajorityVote(Assignment):
    """
    A-2: Majority vote weighted by inverse distance to cluster centroid.

    If embeddings are not supplied the method degenerates to plain majority
    vote.

    Parameters
    ----------
    embeddings : Tensor (N, D) or None
        Feature embeddings aligned with ``cluster_labels``.
    """

    def __init__(self, embeddings: Optional[Tensor] = None) -> None:
        self.embeddings = embeddings

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        embeddings: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[int, int]:
        emb = embeddings if embeddings is not None else self.embeddings

        cl, gt = self._flatten_both(cluster_labels, gt_mask)
        mapping: Dict[int, int] = {}

        for c in np.unique(cl):
            mask_c = cl == c
            gt_c   = gt[mask_c]
            if gt_c.size == 0:
                mapping[int(c)] = 0
                continue

            if emb is not None:
                flat_emb  = emb.reshape(-1, emb.shape[-1]).cpu().float()
                all_flat  = cluster_labels.reshape(-1).cpu().numpy().astype(int)
                valid_idx = np.where(gt_mask.reshape(-1).cpu().numpy() != self.IGNORE_INDEX)[0]
                emb_valid = flat_emb[valid_idx]
                emb_c     = emb_valid[mask_c]
                centroid  = emb_c.mean(dim=0, keepdim=True)              # (1, D)
                dists     = torch.norm(emb_c - centroid, dim=1)          # (N_c,)
                weights   = 1.0 / (dists.numpy() + 1e-6)
            else:
                weights = np.ones(gt_c.size)

            classes      = np.unique(gt_c)
            class_scores = {int(cls): weights[gt_c == cls].sum() for cls in classes}
            mapping[int(c)] = max(class_scores, key=class_scores.get)

        return mapping


# ── A-3: Hungarian matching ───────────────────────────────────────────────────

@_register("hungarian")
class HungarianMatching(Assignment):
    """
    A-3: Build an IoU cost matrix between clusters and GT classes,
    then solve with the Hungarian algorithm.
    """

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        **kwargs: Any,
    ) -> Dict[int, int]:
        from scipy.optimize import linear_sum_assignment

        cl, gt = self._flatten_both(cluster_labels, gt_mask)

        cluster_ids = np.unique(cl)
        class_ids   = np.unique(gt)
        K           = len(cluster_ids)
        C           = len(class_ids)

        # Build IoU matrix (K, C)
        iou_mat = np.zeros((K, C), dtype=np.float64)
        for i, c_id in enumerate(cluster_ids):
            pred_c = cl == c_id
            for j, cls in enumerate(class_ids):
                gt_c  = gt == cls
                inter = (pred_c & gt_c).sum()
                union = (pred_c | gt_c).sum()
                iou_mat[i, j] = inter / (union + 1e-6)

        # Minimise negative IoU
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        mapping: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            mapping[int(cluster_ids[r])] = int(class_ids[c])

        # Unmatched clusters → majority vote fallback
        matched_clusters = set(row_ind)
        mv = MajorityVote()
        fallback = mv.assign(cluster_labels, gt_mask)
        for i, c_id in enumerate(cluster_ids):
            if i not in matched_clusters:
                mapping[int(c_id)] = fallback.get(int(c_id), 0)

        return mapping


# ── A-4: Soft label propagation ───────────────────────────────────────────────

@_register("label_propagation")
class LabelPropagation(Assignment):
    """
    A-4: Build a soft assignment via label propagation on a K-NN graph in
    embedding space.  Returns a hard mapping after propagation.

    Parameters
    ----------
    k : int
        Number of neighbours.
    alpha : float
        Propagation dampening factor (0 < α < 1).
    n_iter : int
        Number of propagation iterations.
    """

    def __init__(self, k: int = 10, alpha: float = 0.8, n_iter: int = 20) -> None:
        self.k      = k
        self.alpha  = alpha
        self.n_iter = n_iter

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        embeddings: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[int, int]:
        # Fallback to majority vote if no embeddings
        if embeddings is None:
            return MajorityVote().assign(cluster_labels, gt_mask)

        cl, gt = self._flatten_both(cluster_labels, gt_mask)
        flat_emb = embeddings.reshape(-1, embeddings.shape[-1])
        flat_gt_full = gt_mask.reshape(-1).cpu().numpy().astype(int)
        valid_mask   = flat_gt_full != self.IGNORE_INDEX

        X   = flat_emb.cpu().float().numpy()
        N   = X.shape[0]

        classes     = np.unique(gt)
        C           = len(classes)
        class_map   = {cls: i for i, cls in enumerate(classes)}

        # Initialise label matrix
        F = np.zeros((N, C), dtype=np.float32)
        for i, cls in enumerate(classes):
            F[valid_mask & (flat_gt_full == cls), i] = 1.0

        Y = F.copy()  # clamped labels

        # Build sparse K-NN affinity (cosine)
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(X, n_neighbors=min(self.k, N - 1),
                             mode="connectivity", include_self=False)
        # Symmetrise and row-normalise
        A = (A + A.T)
        A.data = np.ones_like(A.data)
        deg   = np.asarray(A.sum(axis=1)).flatten()
        Dinv  = np.diag(1.0 / (deg + 1e-6))
        W     = Dinv @ A.toarray()

        for _ in range(self.n_iter):
            F = self.alpha * W @ F + (1.0 - self.alpha) * Y

        predicted = F.argmax(axis=1)

        # Build cluster → class mapping via majority
        cluster_ids = np.unique(cl)
        flat_cl_all = cluster_labels.reshape(-1).cpu().numpy().astype(int)
        mapping: Dict[int, int] = {}
        for c_id in cluster_ids:
            pixels   = np.where(flat_cl_all == c_id)[0]
            preds_c  = predicted[pixels]
            if preds_c.size == 0:
                mapping[int(c_id)] = int(classes[0])
                continue
            vals, cnts = np.unique(preds_c, return_counts=True)
            best_idx   = int(vals[cnts.argmax()])
            mapping[int(c_id)] = int(classes[best_idx]) if best_idx < len(classes) else int(classes[0])

        return mapping


# ── A-5: Abstention with threshold ────────────────────────────────────────────

@_register("abstention")
class AbstentionVote(Assignment):
    """
    A-5: Majority vote with abstention.

    If the top-vote fraction is below *threshold*, the cluster is assigned
    ``ignore_index`` (255) instead of a class label.

    Parameters
    ----------
    threshold : float
        Minimum vote fraction to commit to a class (default 0.6).
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        **kwargs: Any,
    ) -> Dict[int, int]:
        cl, gt = self._flatten_both(cluster_labels, gt_mask)
        mapping: Dict[int, int] = {}
        for c in np.unique(cl):
            mask_c = cl == c
            gt_c   = gt[mask_c]
            if gt_c.size == 0:
                mapping[int(c)] = self.IGNORE_INDEX
                continue
            vals, cnts = np.unique(gt_c, return_counts=True)
            top_frac   = cnts.max() / cnts.sum()
            if top_frac < self.threshold:
                mapping[int(c)] = self.IGNORE_INDEX
            else:
                mapping[int(c)] = int(vals[cnts.argmax()])
        return mapping


# ── A-6: Cross-image consistency tracker ──────────────────────────────────────

@_register("cross_image")
class CrossImageConsistency(Assignment):
    """
    A-6: Per-image majority-vote assignment that also tracks the consistency
    of cluster→class mapping across multiple images as a diagnostic metric.

    Use :meth:`assign` per image and call :meth:`consistency_score` afterwards.

    Parameters
    ----------
    base_method : str
        Which per-image assignment to use under the hood (default "majority_vote").
    """

    def __init__(self, base_method: str = "majority_vote") -> None:
        self._base = get_assignment_method(base_method)
        self._history: Dict[int, list] = {}   # cluster_id → [class_id, ...]

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        **kwargs: Any,
    ) -> Dict[int, int]:
        mapping = self._base.assign(cluster_labels, gt_mask, **kwargs)
        for c_id, cls in mapping.items():
            self._history.setdefault(c_id, []).append(cls)
        return mapping

    def consistency_score(self) -> float:
        """
        Returns the mean per-cluster consistency: fraction of images where the
        most-frequent class matches.  Range [0, 1]; 1 = perfect consistency.
        """
        if not self._history:
            return float("nan")
        scores = []
        for c_id, classes in self._history.items():
            classes_arr          = np.array(classes)
            vals, cnts           = np.unique(classes_arr, return_counts=True)
            scores.append(cnts.max() / cnts.sum())
        return float(np.mean(scores))

    def reset(self) -> None:
        """Clear the history (call between dataset folds)."""
        self._history = {}


# ── A-7: Mask-embedding cosine similarity assignment ─────────────────────────

@_register("mask_embedding_cosine")
class MaskEmbeddingCosine(Assignment):
    """
    A-7: Assign clusters to foreground / background using cosine similarity
    between each cluster's mean patch embedding and the precomputed mask
    embedding (average of patch features inside the GT mask region).

    The cluster whose mean embedding is *most similar* to the mask embedding
    is assigned **foreground** (class 1); all others are assigned
    **background** (class 0).

    If ``mask_embedding`` is a zero vector (no foreground pixels in image),
    the method falls back to majority-vote.

    This method is **annotation-free at clustering time** — it only uses the
    precomputed ``mask_embedding`` vectors that encode the foreground region's
    appearance in DINO feature space.

    Parameters
    ----------
    n_classes : int
        Number of semantic classes (default 2: background + foreground).
    fallback : str
        Assignment method to use when mask_embedding is zero (default
        ``"majority_vote"``).
    """

    def __init__(
        self,
        n_classes: int = 2,
        fallback: str = "majority_vote",
        global_embedding: Optional[Tensor] = None,
    ) -> None:
        self.n_classes = n_classes
        self._fallback = get_assignment_method(fallback)
        self.global_embedding = global_embedding

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        mask_embedding: Optional[Tensor] = None,
        pixel_features: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[int, int]:
        """
        Parameters
        ----------
        cluster_labels  : Tensor (N,) or (H, W)
        gt_mask         : Tensor (H, W)  – used only for fallback
        mask_embedding  : Tensor (768,)  – precomputed region embedding
        pixel_features  : Tensor (H, W, D) or (N, D)
            Per-pixel DINO features (upsampled patch tokens).
            If None and cluster_labels has a spatial shape, patch-level
            centroids are computed from *cluster_labels* only (less accurate).
        """
        # Priority: 1. Passed mask_embedding, 2. self.global_embedding, 3. fallback
        me = mask_embedding if mask_embedding is not None else self.global_embedding

        if me is None:
            return self._fallback.assign(cluster_labels, gt_mask)

        me = me.float().cpu()

        # Fallback: zero mask embedding means no foreground → use majority vote
        if me.norm() < 1e-6:
            return self._fallback.assign(cluster_labels, gt_mask)

        me = me / (me.norm() + 1e-8)  # L2-normalise

        flat_cl = cluster_labels.reshape(-1).cpu()
        cluster_ids = flat_cl.unique().tolist()

        # ── Compute mean feature per cluster ────────────────────────────────
        if pixel_features is not None:
            feats = pixel_features.reshape(-1, pixel_features.shape[-1]).float().cpu()
            centroids: Dict[int, Tensor] = {}
            for c_id in cluster_ids:
                mask_c = (flat_cl == int(c_id))
                if mask_c.any():
                    centroids[int(c_id)] = feats[mask_c].mean(dim=0)
                else:
                    centroids[int(c_id)] = torch.zeros(feats.shape[1])
        else:
            # No pixel features provided — we cannot compute real centroids.
            # Fallback gracefully to majority vote.
            return self._fallback.assign(cluster_labels, gt_mask)

        # ── Cosine similarity between each centroid and mask_embedding ───────
        sims: Dict[int, float] = {}
        for c_id, centroid in centroids.items():
            norm_c = centroid / (centroid.norm() + 1e-8)
            sims[int(c_id)] = float(torch.dot(norm_c, me))

        # Cluster with highest cosine sim → foreground (class 1)
        fg_cluster = max(sims, key=sims.get)

        mapping: Dict[int, int] = {}
        for c_id in cluster_ids:
            mapping[int(c_id)] = 1 if int(c_id) == fg_cluster else 0

        return mapping

    # ------------------------------------------------------------------
    # Override apply() to thread pixel_features through
    # ------------------------------------------------------------------

    def apply(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        mask_embedding: Optional[Tensor] = None,
        pixel_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply the cosine-similarity mapping and return a predicted mask.
        The output shape matches cluster_labels.shape.

        Parameters
        ----------
        cluster_labels  : Tensor (H, W)
        gt_mask         : Tensor (H_gt, W_gt)
        mask_embedding  : Tensor (768,)
        pixel_features  : Tensor (H, W, D)
        """
        H, W    = cluster_labels.shape
        flat_cl = cluster_labels.reshape(-1).long()
        mapping = self.assign(
            flat_cl, gt_mask,
            mask_embedding=mask_embedding,
            pixel_features=pixel_features,
        )
        pred = torch.full((flat_cl.numel(),), self.IGNORE_INDEX, dtype=torch.long)
        for c_id, cls in mapping.items():
            pred[flat_cl == c_id] = cls
        return pred.reshape(H, W)


# ── A-8: Mask-embedding cosine similarity (notebook / global-reference style) ──

@_register("mask_embedding_cosine_global")
class MaskEmbeddingCosineGlobal(Assignment):
    """
    A-8: Assign the foreground cluster via per-patch cosine similarity to a
    **global** mask-embedding reference built from training data.

    This mirrors the ``PolypIdentifier`` methodology from the research notebook,
    and differs from A-7 (``MaskEmbeddingCosine``) in two important ways:

    1. **Global reference instead of per-image reference.**
       A-7 takes a single ``mask_embedding`` vector computed *for the current
       image* at inference time.  This class instead maintains a
       ``global_ref`` vector that is accumulated across many training images
       by calling :meth:`update_reference`.  At inference the same global
       reference is used for every image, making the method truly
       annotation-free at test time.

    2. **Per-patch similarity averaging instead of centroid similarity.**
       A-7 computes one centroid per cluster and measures cosine similarity
       between that centroid and the reference.  This class computes the
       cosine similarity of **every individual patch token** in the cluster
       to the reference and averages those per-patch similarities — matching
       ``PolypIdentifier.compute_polyp_similarity`` exactly.

    Training workflow
    -----------------
    ::

        method = MaskEmbeddingCosineGlobal()
        for img_id in train_ids:
            mask_emb = load_mask_embedding(img_id)   # (P, D) or (D,)
            method.update_reference(mask_emb)
        method.freeze_reference()                     # normalises the average

    Inference workflow
    ------------------
    ::

        mapping = method.assign(cluster_labels, gt_mask,
                                pixel_features=patch_tokens)

    Parameters
    ----------
    n_classes : int
        Number of semantic classes (default 2: background + foreground).
    fallback : str
        Assignment method to use when no reference has been set or
        ``pixel_features`` are absent (default ``"majority_vote"``).
    """

    def __init__(
        self,
        n_classes: int = 2,
        fallback: str = "majority_vote",
    ) -> None:
        self.n_classes = n_classes
        self._fallback = get_assignment_method(fallback)

        # Accumulated reference state (populated via update_reference / freeze_reference)
        self._ref_accumulator: Optional[np.ndarray] = None
        self._ref_count: int = 0
        self.global_ref: Optional[Tensor] = None   # final normalised reference

    # ------------------------------------------------------------------
    # Reference building (training phase)
    # ------------------------------------------------------------------

    def update_reference(self, mask_embedding: "np.ndarray | Tensor") -> None:
        """
        Accumulate one image's mask embedding into the global reference.

        Parameters
        ----------
        mask_embedding : array-like (P, D) or (D,)
            Raw (un-normalised) mask embedding for a single training image.
            If 2-D, the patches are averaged first (matching the notebook's
            ``mask_emb.mean(axis=0)`` step).
        """
        if isinstance(mask_embedding, Tensor):
            emb = mask_embedding.float().cpu().numpy()
        else:
            emb = np.asarray(mask_embedding, dtype=np.float32)

        if emb.ndim == 2:
            emb = emb.mean(axis=0)   # (P, D) → (D,)

        if self._ref_accumulator is None:
            self._ref_accumulator = emb.copy()
        else:
            self._ref_accumulator += emb
        self._ref_count += 1

    def freeze_reference(self) -> None:
        """
        Finalise the global reference by averaging all accumulated embeddings.
        Must be called once after all :meth:`update_reference` calls.
        """
        if self._ref_accumulator is None or self._ref_count == 0:
            raise RuntimeError(
                "No mask embeddings have been accumulated. "
                "Call update_reference() for each training image first."
            )
        avg = self._ref_accumulator / self._ref_count          # mean across images
        avg_t = torch.from_numpy(avg).float()
        self.global_ref = avg_t / (avg_t.norm() + 1e-8)        # L2-normalise

    def set_reference(self, ref: "np.ndarray | Tensor") -> None:
        """
        Directly set the global reference (e.g. loaded from disk).

        Parameters
        ----------
        ref : array-like (D,)
            Pre-computed global mask embedding.  Will be L2-normalised.
        """
        if isinstance(ref, Tensor):
            r = ref.float().cpu()
        else:
            r = torch.from_numpy(np.asarray(ref, dtype=np.float32))
        self.global_ref = r / (r.norm() + 1e-8)

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def _compute_cluster_similarity(
        self,
        cluster_patches: Tensor,  # (N_c, D)  raw patch features for one cluster
        ref: Tensor,              # (D,)      already normalised
    ) -> float:
        """
        Per-patch cosine similarity averaged over the cluster.

        Mirrors ``PolypIdentifier.compute_polyp_similarity``:
          1. Normalise each patch vector individually.
          2. Dot-product each with the reference.
          3. Return the mean similarity.
        """
        if cluster_patches.shape[0] == 0:
            return 0.0
        norms = cluster_patches.norm(dim=1, keepdim=True) + 1e-8  # (N_c, 1)
        normalised = cluster_patches / norms                        # (N_c, D)
        sims = (normalised @ ref)                                   # (N_c,)
        return float(sims.mean())

    def assign(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        pixel_features: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[int, int]:
        """
        Parameters
        ----------
        cluster_labels  : Tensor (N,) or (H, W)
        gt_mask         : Tensor (H, W)  – used only for fallback
        pixel_features  : Tensor (H, W, D) or (N, D)
            Per-pixel / per-patch DINO features.  Required; falls back to
            majority vote if absent.
        """
        if self.global_ref is None:
            return self._fallback.assign(cluster_labels, gt_mask)

        if pixel_features is None:
            return self._fallback.assign(cluster_labels, gt_mask)

        ref = self.global_ref.float().cpu()   # (D,)  already normalised

        flat_cl   = cluster_labels.reshape(-1).cpu()
        feats     = pixel_features.reshape(-1, pixel_features.shape[-1]).float().cpu()
        cluster_ids = flat_cl.unique().tolist()

        # Per-patch similarity averaged per cluster  (notebook methodology)
        sims: Dict[int, float] = {}
        for c_id in cluster_ids:
            mask_c = flat_cl == int(c_id)
            cluster_patches = feats[mask_c]           # (N_c, D)
            sims[int(c_id)] = self._compute_cluster_similarity(cluster_patches, ref)

        # Cluster with highest mean per-patch cosine sim → foreground (class 1)
        fg_cluster = max(sims, key=sims.__getitem__)

        mapping: Dict[int, int] = {}
        for c_id in cluster_ids:
            mapping[int(c_id)] = 1 if int(c_id) == fg_cluster else 0

        return mapping

    # ------------------------------------------------------------------
    # Override apply() to thread pixel_features through
    # ------------------------------------------------------------------

    def apply(
        self,
        cluster_labels: Tensor,
        gt_mask: Tensor,
        pixel_features: Optional[Tensor] = None,
    ) -> Tensor:
        H, W    = cluster_labels.shape
        flat_cl = cluster_labels.reshape(-1).long()
        mapping = self.assign(flat_cl, gt_mask, pixel_features=pixel_features)
        pred = torch.full((flat_cl.numel(),), self.IGNORE_INDEX, dtype=torch.long)
        for c_id, cls in mapping.items():
            pred[flat_cl == c_id] = cls
        return pred.reshape(H, W)
