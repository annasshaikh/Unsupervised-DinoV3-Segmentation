"""
dinov3_seg: Modular DINO-based unsupervised segmentation pipeline.

Stages:
    - DataLoader     : dataset loading and preprocessing
    - Resolution     : patch-feature upsampling to pixel resolution
    - Clustering     : unsupervised grouping of pixel/patch embeddings
    - Assignment     : mapping cluster IDs to semantic class labels
    - PostProcessing : mask refinement
    - LowLevel       : additional hand-crafted features
    - Pipeline       : end-to-end orchestration
    - Metrics        : evaluation utilities
"""

from .config import PipelineConfig, get_default_config
from .pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Pipeline", "PipelineConfig", "get_default_config"]
