"""
clustering.py – Unsupervised clustering of pixel/patch embeddings.

All methods inherit from :class:`Clustering` and implement::

    cluster(features, n_clusters=None, **kwargs) -> labels : Tensor[N]

Registered methods
------------------
C-1  kmeans         – Standard K-Means (sklearn / fast CUDA variant)
C-2  kmeans_pca     – K-Means on PCA-compressed features
C-3  hdbscan        – HDBSCAN density clustering
C-4  spectral        – Spectral clustering with optional spatial regularisation
C-5  hierarchical   – Agglomerative clustering (ward / complete / average)
C-6  ncut           – Normalised Cuts (skimage)
C-7  joint_kmeans   – Joint K-Means across multiple images
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional

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


def get_clustering_method(name: str, **kwargs) -> "Clustering":
    """
    Factory function.

    Parameters
    ----------
    name : str
        One of the registered keys.
    **kwargs
        Forwarded to the class constructor.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown clustering method {name!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


# ── Base class ────────────────────────────────────────────────────────────────

class Clustering(ABC):
    """Abstract base class for clustering methods."""

    @abstractmethod
    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Cluster a set of feature vectors.

        Parameters
        ----------
        features : Tensor (N, D)
            Per-pixel or per-patch embeddings.
        n_clusters : int or None
            Requested number of clusters (ignored for density methods).
        spatial_positions : Tensor (N, 2) or None
            (row, col) for each feature; used by methods with spatial prior.

        Returns
        -------
        labels : Tensor (N,)  – integer cluster indices in [0, K-1];
                                −1 for noise (HDBSCAN).
        """

    @staticmethod
    def _to_numpy(t: Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy()


# ── C-1: K-Means ──────────────────────────────────────────────────────────────

@_register("kmeans")
class KMeans(Clustering):
    """
    C-1: Standard K-Means clustering via sklearn MiniBatchKMeans.

    Parameters
    ----------
    n_clusters : int
        Default K (can be overridden per call).
    random_state : int
    n_init : int
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int = 0,
        n_init: int = 10,
    ) -> None:
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.n_init       = n_init

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        from sklearn.cluster import MiniBatchKMeans

        K = n_clusters if n_clusters is not None else self.n_clusters
        X = self._to_numpy(features)
        km = MiniBatchKMeans(
            n_clusters=K,
            random_state=self.random_state,
            n_init=self.n_init,
            **kwargs,
        )
        labels = km.fit_predict(X)
        return torch.from_numpy(labels).long()


# ── C-2: K-Means on PCA ───────────────────────────────────────────────────────

@_register("kmeans_pca")
class KMeansPCA(Clustering):
    """
    C-2: Apply PCA first, then run K-Means in the reduced space.

    Parameters
    ----------
    n_clusters : int
    pca_dim : int
        Number of PCA components (default 32).
    random_state : int
    """

    def __init__(
        self,
        n_clusters: int = 8,
        pca_dim: int = 32,
        random_state: int = 0,
        n_init: int = 10,
    ) -> None:
        self.n_clusters   = n_clusters
        self.pca_dim      = pca_dim
        self.random_state = random_state
        self.n_init       = n_init

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA

        K   = n_clusters if n_clusters is not None else self.n_clusters
        X   = self._to_numpy(features)
        dim = min(self.pca_dim, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=dim, random_state=self.random_state)
        Xr  = pca.fit_transform(X)
        km  = MiniBatchKMeans(
            n_clusters=K,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        labels = km.fit_predict(Xr)
        return torch.from_numpy(labels).long()


# ── C-3: HDBSCAN ──────────────────────────────────────────────────────────────

@_register("hdbscan")
class HDBSCANClustering(Clustering):
    """
    C-3: HDBSCAN density-based clustering.

    Noise points are labelled −1.  Note that ``n_clusters`` is *ignored*
    for this method (the number of clusters is data-driven).

    Parameters
    ----------
    min_cluster_size : int
    min_samples : int or None
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples      = min_samples

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            try:
                from sklearn.cluster import HDBSCAN as hdbscan_lib  # sklearn >=1.3
            except ImportError as exc:
                raise ImportError(
                    "HDBSCAN requires either 'hdbscan' or sklearn>=1.3."
                ) from exc

        X = self._to_numpy(features)
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            **kwargs,
        )
        labels = clusterer.fit_predict(X)
        return torch.from_numpy(labels).long()


# ── C-4: Spectral clustering ──────────────────────────────────────────────────

@_register("spectral")
class SpectralClustering(Clustering):
    """
    C-4: Spectral clustering with optional spatial regularisation on the
    196-patch graph (14×14 ViT grid).

    Parameters
    ----------
    n_clusters : int
    spatial_weight : float
        How much to weight (row, col) normalised coordinates when building
        the affinity matrix (0 = pure feature similarity).
    random_state : int
    """

    def __init__(
        self,
        n_clusters: int = 8,
        spatial_weight: float = 0.0,
        random_state: int = 0,
    ) -> None:
        self.n_clusters    = n_clusters
        self.spatial_weight = spatial_weight
        self.random_state  = random_state

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        from sklearn.cluster import SpectralClustering as _SC

        K   = n_clusters if n_clusters is not None else self.n_clusters
        X   = self._to_numpy(features)

        if self.spatial_weight > 0 and spatial_positions is not None:
            sp = self._to_numpy(spatial_positions)
            sp_norm = (sp - sp.min(0)) / (sp.max(0) - sp.min(0) + 1e-6)
            X  = np.concatenate(
                [X, self.spatial_weight * sp_norm], axis=1
            )

        sc = _SC(
            n_clusters=K,
            affinity="nearest_neighbors",
            random_state=self.random_state,
            assign_labels="kmeans",
            n_jobs=-1,
            **kwargs,
        )
        labels = sc.fit_predict(X)
        return torch.from_numpy(labels).long()


# ── C-5: Hierarchical agglomerative clustering ────────────────────────────────

@_register("hierarchical")
class HierarchicalClustering(Clustering):
    """
    C-5: Agglomerative clustering with configurable linkage.

    Parameters
    ----------
    n_clusters : int
    linkage : str
        ``"ward"`` (default), ``"complete"``, or ``"average"``.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        linkage: str = "ward",
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage    = linkage

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        from sklearn.cluster import AgglomerativeClustering

        K   = n_clusters if n_clusters is not None else self.n_clusters
        X   = self._to_numpy(features)
        ac  = AgglomerativeClustering(
            n_clusters=K, linkage=self.linkage, **kwargs
        )
        labels = ac.fit_predict(X)
        return torch.from_numpy(labels).long()


# ── C-6: Normalised Cuts ──────────────────────────────────────────────────────

@_register("ncut")
class NormalisedCut(Clustering):
    """
    C-6: Normalised Cuts via skimage's SLIC + RAG or via spectral embedding.

    Uses sklearn SpectralClustering with a cosine-similarity affinity as a
    practical approximation of Ncut.  For a full Ncut implementation on
    image segments see :func:`skimage.graph.cut_normalized`.

    Parameters
    ----------
    n_clusters : int
    random_state : int
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 0) -> None:
        self.n_clusters   = n_clusters
        self.random_state = random_state

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        from sklearn.cluster import SpectralClustering as _SC

        K  = n_clusters if n_clusters is not None else self.n_clusters
        X  = self._to_numpy(features)
        sc = _SC(
            n_clusters=K,
            affinity="cosine",
            random_state=self.random_state,
            assign_labels="discretize",
            n_jobs=-1,
        )
        labels = sc.fit_predict(X)
        return torch.from_numpy(labels).long()


# ── C-7: Joint K-Means (multi-image) ─────────────────────────────────────────

@_register("joint_kmeans")
class JointKMeans(Clustering):
    """
    C-7: Joint K-Means trained on all patch embeddings from the train set,
    then applied image-by-image at inference.

    Call :meth:`fit` with a list of per-image feature tensors before using
    :meth:`cluster`.

    Parameters
    ----------
    n_clusters : int
    random_state : int
    """

    def __init__(self, n_clusters: int = 8, random_state: int = 0) -> None:
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self._km          = None

    def fit(self, feature_list: List[Tensor]) -> "JointKMeans":
        """
        Train K-Means on all features stacked from a list of tensors.

        Parameters
        ----------
        feature_list : list[Tensor (N_i, D)]
        """
        from sklearn.cluster import MiniBatchKMeans

        X = np.concatenate([self._to_numpy(f) for f in feature_list], axis=0)
        self._km = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(X)
        return self

    def cluster(
        self,
        features: Tensor,
        n_clusters: Optional[int] = None,
        spatial_positions: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        if self._km is None:
            raise RuntimeError(
                "JointKMeans.fit() must be called before cluster(). "
                "Provide a list of feature tensors from the train set."
            )
        X      = self._to_numpy(features)
        labels = self._km.predict(X)
        return torch.from_numpy(labels).long()
