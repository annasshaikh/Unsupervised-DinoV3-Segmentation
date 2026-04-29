"""
dataloader.py – Dataset loading and preprocessing for dinov3_seg.

Each dataset sample is a dict with keys:
    image        : FloatTensor (3, 224, 224)  – ImageNet-normalised
    mask         : FloatTensor (1, 224, 224)  – binary {0, 1}
    patch_tokens : FloatTensor (N_patches, D) – precomputed DINO embeddings
    filename     : str
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ── ImageNet statistics ──────────────────────────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Loads image / mask / patch-token triplets from a structured directory.

    Expected layout::

        dataset_path/
            images/
                train/  <filename>.jpg  (or .png)
                test/   ...
            masks/
                train/  <filename>.png
                test/   ...
            embeddings/
                train/  <filename>.npy   # shape (N_patches, D)
                test/   ...

    Parameters
    ----------
    dataset_path : str | Path
        Root directory of the dataset.
    split : str
        ``"train"`` or ``"test"``.
    img_transform : callable, optional
        Override the default image transform.
    mask_transform : callable, optional
        Override the default mask transform.
    binary_mask : bool
        If ``True`` (default), convert mask pixel values > 0 to 1.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        img_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        binary_mask: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_path)
        self.split = split
        self.binary_mask = binary_mask
        self.img_transform  = img_transform  or _IMG_TRANSFORM
        self.mask_transform = mask_transform or _MASK_TRANSFORM

        self.img_dir   = self.root / split / "images"      
        self.mask_dir  = self.root / split / "masks"      
        self.emb_dir   = self.root   / split / "embeddings"

        self.filenames = self._collect_filenames()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _collect_filenames(self) -> List[str]:
        """Return sorted list of base names (no extension) present in img_dir."""
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        names: List[str] = []
        for p in sorted(self.img_dir.iterdir()):
            if p.suffix.lower() in exts:
                names.append(p.stem)
        if not names:
            raise FileNotFoundError(
                f"No images found in {self.img_dir}. "
                "Check dataset_path and split argument."
            )
        return names

    def _find_image(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
            p = self.img_dir / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found for stem={stem!r} in {self.img_dir}")

    def _find_mask(self, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp"):
            p = self.mask_dir / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"Mask not found for stem={stem!r} in {self.mask_dir}")

    def _load_patch_tokens(self, stem: str) -> torch.Tensor:
        """Load precomputed patch embeddings from a .npy file."""
        npy_path = self.emb_dir / (stem + ".npy")
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Embedding not found: {npy_path}. "
                "Ensure embeddings are precomputed and placed in the 'embeddings/' folder."
            )
        arr = np.load(str(npy_path))           # (N_patches, D)
        return torch.from_numpy(arr).float()

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        stem = self.filenames[idx]

        # ── image ──
        img = Image.open(self._find_image(stem)).convert("RGB")
        image_tensor: torch.Tensor = self.img_transform(img)           # (3, 224, 224)

        # ── mask ──
        mask_pil = Image.open(self._find_mask(stem)).convert("L")
        mask_tensor: torch.Tensor = self.mask_transform(mask_pil)      # (1, 224, 224)
        if self.binary_mask:
            mask_tensor = (mask_tensor > 0).float()

        # ── patch tokens ──
        patch_tokens = self._load_patch_tokens(stem)                   # (N, D)

        return {
            "image":        image_tensor,
            "mask":         mask_tensor,
            "patch_tokens": patch_tokens,
            "filename":     stem,
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def get_dataloaders(
    dataset_path: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    binary_mask: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for the given dataset path.

    Parameters
    ----------
    dataset_path : str | Path
        Root directory (see :class:`SegmentationDataset` for layout).
    batch_size : int
    num_workers : int
    binary_mask : bool
    pin_memory : bool

    Returns
    -------
    train_loader, test_loader : tuple[DataLoader, DataLoader]
        Each loader yields dicts with keys
        ``image``, ``mask``, ``patch_tokens``, ``filename``.
    """
    train_ds = SegmentationDataset(dataset_path, split="train", binary_mask=binary_mask)
    test_ds  = SegmentationDataset(dataset_path, split="test",  binary_mask=binary_mask)

    def _collate(batch: List[Dict]) -> Dict:
        """Custom collate to handle variable-length filename strings."""
        keys = batch[0].keys()
        out: Dict = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals          # filenames → list[str]
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
    return train_loader, test_loader
