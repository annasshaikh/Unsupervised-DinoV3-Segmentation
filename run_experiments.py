"""
run_experiments.py – Robust combinatorial experiment runner.

Features
--------
* Reads experiments.yaml and builds Cartesian product of all axes.
* Skips already-completed experiments (resume from results.jsonl).
* Caches pixel features per (stem, resolution) to avoid recomputation.
* Saves per-image visualisations, cluster maps, PCA maps, low-level maps,
  cosine-similarity maps, and qualitative grids.
* Robust: exceptions in any experiment are caught, logged, and skipped.
* Writes results.jsonl (one JSON per line) and summary.csv at the end.
* Generates aggregate summary charts (bar, boxplot, heatmap).

Usage
-----
    python run_experiments.py --config experiments.yaml
    python run_experiments.py --config experiments.yaml --quick-test
    python run_experiments.py --config experiments.yaml --filter-resolution bilinear none
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ── Imports ─────────────────────────────────────────────────────────────────────
# The package directory name has hyphens, which Python cannot import directly.
# Strategy: create a short alias in sys.modules that points to the real folder
# by inserting a minimal finder/loader shim, then import normally.
_HERE     = Path(__file__).resolve().parent          # .../Unsupervised-DinoV3-Segmentation
_PKG_REAL = _HERE.name                               # "Unsupervised-DinoV3-Segmentation"
_PKG_SAFE = "_dinov3_seg"                            # importable alias (no hyphens)

# Register a simple path-based package under the safe name
import importlib.util as _ilu, types as _types

# Create a package stub so relative imports inside submodules resolve
_pkg_mod = _types.ModuleType(_PKG_SAFE)
_pkg_mod.__path__    = [str(_HERE)]
_pkg_mod.__package__ = _PKG_SAFE
_pkg_mod.__spec__    = _ilu.spec_from_file_location(
    _PKG_SAFE, _HERE / "__init__.py",
    submodule_search_locations=[str(_HERE)]
)
sys.modules[_PKG_SAFE] = _pkg_mod

# Also register each sub-module with the alias so `from .x import y` resolves
def _load_sub(name: str):
    full = f"{_PKG_SAFE}.{name}"
    spec = _ilu.spec_from_file_location(
        full, _HERE / f"{name}.py",
        submodule_search_locations=[str(_HERE)]
    )
    mod               = _ilu.module_from_spec(spec)
    mod.__package__   = _PKG_SAFE
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    setattr(_pkg_mod, name, mod)
    return mod

_cfg_mod  = _load_sub("config")
_dl_mod   = _load_sub("dataloader")
_pl_mod   = _load_sub("pipeline")
_met_mod  = _load_sub("metrics")
_vis_mod  = _load_sub("visualizer")

PipelineConfig          = _cfg_mod.PipelineConfig
SegmentationDataset     = _dl_mod.SegmentationDataset
Pipeline                = _pl_mod.Pipeline
compute_all_metrics     = _met_mod.compute_all_metrics
Visualizer              = _vis_mod.Visualizer
save_experiment_summary = _vis_mod.save_experiment_summary
save_axis_comparison    = _vis_mod.save_axis_comparison
save_heatmap            = _vis_mod.save_heatmap
denorm                  = _vis_mod.denorm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_dict(d: Dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _exp_id(res: Dict, clust: Dict, ll: Dict, assign: Dict, ds: Optional[Dict] = None) -> str:
    base = f"R-{res['id']}__C-{clust['id']}__L-{ll['id']}__A-{assign['id']}"
    if ds is not None:
        return f"D-{ds['name']}__" + base
    return base


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_completed(results_path: Path) -> set:
    """Return set of already-completed experiment IDs."""
    done = set()
    if results_path.exists():
        with open(results_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        done.add(d["exp_id"])
                    except Exception:
                        pass
    return done


def _append_result(results_path: Path, record: Dict) -> None:
    with open(results_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureCache:
    """Disk-backed cache for pixel features keyed by (stem, resolution_id)."""

    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled   = enabled
        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, stem: str, res_id: str) -> Path:
        return self.cache_dir / f"{stem}__{res_id}.pt"

    def get(self, stem: str, res_id: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        p = self._key(stem, res_id)
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                return None
        return None

    def put(self, stem: str, res_id: str, tensor: torch.Tensor) -> None:
        if not self.enabled:
            return
        try:
            torch.save(tensor.cpu(), self._key(stem, res_id))
        except Exception as e:
            warnings.warn(f"Cache write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Config builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline_config(
    dataset_cfg: Dict,
    runtime_cfg: Dict,
    res_cfg: Dict,
    clust_cfg: Dict,
    ll_cfg: Dict,
    assign_cfg: Dict,
) -> PipelineConfig:
    """Build a PipelineConfig from four axis dicts."""

    res_dict = {"method": res_cfg["method"]}
    res_dict.update(res_cfg.get("kwargs", {}) or {})

    clust_dict = {"method": clust_cfg["method"]}
    clust_dict.update(clust_cfg.get("kwargs", {}) or {})

    assign_dict = {"method": assign_cfg["method"]}
    assign_dict.update(assign_cfg.get("kwargs", {}) or {})

    ll_dict = None
    if ll_cfg.get("method") is not None:
        ll_dict = {
            "method":      ll_cfg["method"],
            "fusion_mode": ll_cfg.get("fusion_mode", "concat"),
            "low_weight":  ll_cfg.get("low_weight", 1.0),
        }
        ll_dict.update(ll_cfg.get("kwargs", {}) or {})

    return PipelineConfig(
        dataset=dataset_cfg.get("name", "HumanSeg"),
        dataset_path=dataset_cfg["path"],
        n_classes=dataset_cfg.get("n_classes", 2),
        device=runtime_cfg.get("device", "cuda"),
        batch_size=runtime_cfg.get("batch_size", 1),
        num_workers=runtime_cfg.get("num_workers", 2),
        seed=runtime_cfg.get("seed", 42),
        resolution=res_dict,
        clustering=clust_dict,
        assignment=assign_dict,
        postprocess=[],
        lowlevel=ll_dict,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Single experiment runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_experiment(
    exp_id:     str,
    cfg:        PipelineConfig,
    res_cfg:    Dict,
    ll_cfg:     Dict,
    dataset_cfg: Dict,
    out_cfg:    Dict,
    vis_cfg:    Dict,
    cache:      FeatureCache,
    exp_out_dir: Path,
    max_samples: Optional[int],
    verbose:    bool = True,
) -> Dict:
    """
    Run one full experiment, save visualisations, return aggregated metrics dict.
    """
    t0 = time.time()

    # ── Dataloader ─────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader, Subset
    ds = SegmentationDataset(
        cfg.dataset_path,
        split=dataset_cfg.get("split", "test"),
        binary_mask=True,
    )
    if max_samples and max_samples < len(ds):
        indices = list(range(max_samples))
        ds = Subset(ds, indices)

    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=False)

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline(cfg)

    # ── Visualiser ─────────────────────────────────────────────────────────────
    vis_enabled = out_cfg.get("save_visualisations", True)
    vis_every   = out_cfg.get("vis_every_n_images", 5)
    vis_max     = out_cfg.get("vis_max_images", 20)
    vis = Visualizer(
        exp_out_dir / "vis",
        dpi=vis_cfg.get("dpi", 120),
        cmap_clusters=vis_cfg.get("colormap_clusters", "tab20"),
        cmap_features=vis_cfg.get("colormap_features", "viridis"),
        cmap_lowlevel=vis_cfg.get("colormap_lowlevel", "plasma"),
    ) if vis_enabled else None

    all_metrics: List[Dict] = []
    per_image_records: List[Dict] = []
    vis_count = 0

    for sample_idx, batch in enumerate(loader):
        stem      = batch["filename"][0] if isinstance(batch["filename"], list) else batch["filename"]
        image     = batch["image"][0]       # (3, H, W)
        gt_mask   = batch["mask"][0, 0].long()  # (H, W)
        patch_tok = batch["patch_tokens"][0]    # (196, D)
        mask_emb  = batch["mask_embedding"][0]  # (D,)
        cls_emb   = batch["cls_embedding"][0]   # (D,)

        try:
            # ── Check cache for pixel features ────────────────────────────────
            cached_pf = cache.get(stem, res_cfg["id"])

            pred_mask, stages = pipeline.run(
                image,
                patch_tokens=patch_tok,
                gt_mask=gt_mask,
                mask_embedding=mask_emb,
                cls_embedding=cls_emb,
                return_stages=True,
            )

            # Cache pixel features from this resolution
            if cached_pf is None and "pixel_features" in stages:
                cache.put(stem, res_cfg["id"], stages["pixel_features"])

            # ── Align prediction to GT size ────────────────────────────────────
            import torch.nn.functional as F
            if pred_mask.shape != gt_mask.shape:
                pred_mask = F.interpolate(
                    pred_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=gt_mask.shape, mode="nearest",
                )[0, 0].long()

            # ── Metrics ───────────────────────────────────────────────────────
            m = compute_all_metrics(pred_mask, gt_mask, cfg.n_classes)
            all_metrics.append(m)
            per_image_records.append({"stem": stem, **m})

            # ── Visualisation ─────────────────────────────────────────────────
            do_vis = (
                vis is not None
                and sample_idx % vis_every == 0
                and vis_count < vis_max
            )
            if do_vis:
                try:
                    vis.save_all(
                        stem=stem,
                        image=image,
                        gt_mask=gt_mask,
                        pred_mask=pred_mask,
                        stages=stages,
                        lowlevel_cfg=ll_cfg,
                        metrics=m,
                    )
                    vis_count += 1
                except Exception as ve:
                    warnings.warn(f"[vis] {stem}: {ve}")

        except Exception as e:
            warnings.warn(f"[{exp_id}] sample {stem} failed: {e}")
            traceback.print_exc()
            continue

    # ── Aggregate ──────────────────────────────────────────────────────────────
    if not all_metrics:
        return {"exp_id": exp_id, "n_samples": 0, "error": "all_samples_failed"}

    keys = all_metrics[0].keys()
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m.get(k, float("nan")))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")

    elapsed = time.time() - t0

    # Save per-image metrics
    if out_cfg.get("save_per_image_metrics", True):
        pim_path = exp_out_dir / "per_image_metrics.json"
        with open(pim_path, "w") as f:
            json.dump(per_image_records, f, indent=2)

    if verbose:
        print(
            f"  ✓ [{exp_id}]  mIoU={agg.get('miou', float('nan')):.4f}"
            f"  dice={agg.get('dice', float('nan')):.4f}"
            f"  pix_acc={agg.get('pixel_acc', float('nan')):.4f}"
            f"  ({elapsed:.1f}s, n={len(all_metrics)})"
        )

    return {
        "exp_id":    exp_id,
        "n_samples": len(all_metrics),
        "elapsed_s": elapsed,
        **agg,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Combination builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_combinations(cfg: Dict) -> List[Tuple[str, Dict, Dict, Dict, Dict, Dict]]:
    """Return list of (exp_id, ds, res, clust, ll, assign) config dicts."""
    qt  = cfg.get("quick_test", {})
    qte = qt.get("enabled", False)

    def _ids(section: str) -> List[str]:
        if qte:
            return qt.get(section, [])
        filt = (cfg.get("filter") or {}).get(f"enabled_{section}", [])
        return filt  # empty = all

    def _filter(items: List[Dict], ids: List[str]) -> List[Dict]:
        if not ids:
            return items
        return [x for x in items if x["id"] in ids]

    res_list    = _filter(cfg["resolution"],  _ids("resolution"))
    clust_list  = _filter(cfg["clustering"],  _ids("clustering"))
    ll_list     = _filter(cfg["lowlevel"],    _ids("lowlevel"))
    assign_list = _filter(cfg["assignment"],  _ids("assignment"))

    ds_list = cfg.get("datasets")
    if ds_list is None:
        ds_list = [cfg.get("dataset")]
        use_ds_in_id = False
    else:
        use_ds_in_id = True

    skip = set((cfg.get("filter") or {}).get("skip_ids", []))
    combos = []
    for ds, res, clust, ll, assign in itertools.product(ds_list, res_list, clust_list, ll_list, assign_list):
        eid = _exp_id(res, clust, ll, assign, ds if use_ds_in_id else None)
        if eid not in skip:
            combos.append((eid, ds, res, clust, ll, assign))
    return combos


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv3 segmentation experiment runner")
    parser.add_argument("--config",  default="experiments.yaml", help="Path to experiments.yaml")
    parser.add_argument("--quick-test", action="store_true",  help="Override quick_test.enabled=true")
    parser.add_argument("--filter-resolution", nargs="*",     help="Restrict resolution IDs")
    parser.add_argument("--filter-clustering",  nargs="*",    help="Restrict clustering IDs")
    parser.add_argument("--filter-lowlevel",    nargs="*",    help="Restrict lowlevel IDs")
    parser.add_argument("--filter-assignment",  nargs="*",    help="Restrict assignment IDs")
    parser.add_argument("--max-samples",        type=int,     help="Override max_samples")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = _load_yaml(args.config)

    if args.quick_test:
        cfg.setdefault("quick_test", {})["enabled"] = True

    # CLI overrides
    filt = cfg.setdefault("filter", {})
    if args.filter_resolution: filt["enabled_resolution"] = args.filter_resolution
    if args.filter_clustering:  filt["enabled_clustering"]  = args.filter_clustering
    if args.filter_lowlevel:    filt["enabled_lowlevel"]    = args.filter_lowlevel
    if args.filter_assignment:  filt["enabled_assignment"]  = args.filter_assignment

    # Handle single dataset fallback for extracting some default config
    default_dataset_cfg = cfg.get("datasets", [{}])[0] if cfg.get("datasets") else cfg.get("dataset", {})

    runtime_cfg  = cfg["runtime"]
    out_cfg      = cfg["output"]
    vis_cfg      = cfg.get("visualisation", {})
    cache_cfg    = cfg.get("cache", {})

    qt_enabled   = cfg.get("quick_test", {}).get("enabled", False)
    max_samples  = (
        args.max_samples
        or (cfg["quick_test"].get("max_samples") if qt_enabled else None)
        or default_dataset_cfg.get("max_samples")
    )

    # ── Output setup ──────────────────────────────────────────────────────────
    out_root = Path(out_cfg["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.jsonl"
    summary_path = out_root / "summary.csv"
    summary_vis  = out_root / "summary_plots"
    summary_vis.mkdir(exist_ok=True)

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache = FeatureCache(
        Path(cache_cfg.get("dir", "./experiment_cache")),
        enabled=cache_cfg.get("enabled", True),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    done = _load_completed(results_path)
    if done:
        print(f"\n[Resume] {len(done)} experiments already completed – skipping them.\n")

    # ── Build combinations ────────────────────────────────────────────────────
    combos = build_combinations(cfg)
    total  = len(combos)
    print(f"\n{'='*70}")
    print(f"  DINOv3 Segmentation Experiment Runner")
    print(f"  Total combinations : {total}")
    print(f"  Already done       : {len(done)}")
    print(f"  Remaining          : {total - len([c for c in combos if c[0] in done])}")
    print(f"  Output             : {out_root}")
    print(f"{'='*70}\n")

    all_results: List[Dict] = []

    # Load already-completed results for summary charts
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except Exception:
                        pass

    for combo_idx, (exp_id, ds_cfg, res, clust, ll, assign) in enumerate(combos):

        # Resume check
        if exp_id in done:
            continue

        print(f"\n[{combo_idx+1}/{total}] {exp_id}")

        exp_out_dir = out_root / exp_id
        exp_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipe_cfg = build_pipeline_config(
                ds_cfg, runtime_cfg,
                res, clust, ll, assign,
            )

            record = run_one_experiment(
                exp_id=exp_id,
                cfg=pipe_cfg,
                res_cfg=res,
                ll_cfg=ll,
                dataset_cfg=ds_cfg,
                out_cfg=out_cfg,
                vis_cfg=vis_cfg,
                cache=cache,
                exp_out_dir=exp_out_dir,
                max_samples=max_samples,
                verbose=args.verbose,
            )

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            record = {
                "exp_id":     exp_id,
                "n_samples":  0,
                "error":      str(e),
                "resolution": res["id"],
                "clustering": clust["id"],
                "lowlevel":   ll["id"],
                "assignment": assign["id"],
                "dataset":    ds_cfg["name"],
            }

        # Tag axes onto record
        record.update({
            "dataset":    ds_cfg["name"],
            "resolution": res["id"],
            "clustering": clust["id"],
            "lowlevel":   ll["id"],
            "assignment": assign["id"],
        })

        # Save experiment-level config snapshot
        try:
            with open(exp_out_dir / "config.json", "w") as f:
                json.dump({
                    "dataset":    ds_cfg,
                    "resolution": res,
                    "clustering": clust,
                    "lowlevel":   ll,
                    "assignment": assign,
                    "runtime":    runtime_cfg,
                }, f, indent=2)
        except Exception:
            pass

        # Persist result
        _append_result(results_path, record)
        done.add(exp_id)
        all_results.append(record)

    # ═══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  All experiments complete. Writing summary…")

    # CSV
    if out_cfg.get("save_summary_csv", True) and all_results:
        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            df.to_csv(summary_path, index=False)
            print(f"  CSV  → {summary_path}")

            # Top-5 by mIoU
            if "miou" in df.columns:
                top = df.nlargest(5, "miou")[["exp_id", "miou", "dice", "pixel_acc"]]
                print("\n  ── Top-5 by mIoU ──")
                print(top.to_string(index=False))
        except ImportError:
            pass

    # Summary visualisations
    for metric in ("miou", "dice", "pixel_acc"):
        try:
            save_experiment_summary(all_results, summary_vis, metric_key=metric)
        except Exception as e:
            warnings.warn(f"summary plot failed ({metric}): {e}")

    for axis in ("resolution", "clustering", "lowlevel", "assignment"):
        for metric in ("miou", "dice"):
            try:
                save_axis_comparison(all_results, axis, summary_vis, metric_key=metric)
            except Exception as e:
                warnings.warn(f"axis plot failed ({axis}/{metric}): {e}")

    for ax_x, ax_y in [("resolution", "clustering"), ("lowlevel", "assignment"),
                        ("clustering", "lowlevel"), ("resolution", "assignment")]:
        try:
            save_heatmap(all_results, ax_x, ax_y, summary_vis, metric_key="miou")
        except Exception as e:
            warnings.warn(f"heatmap failed ({ax_x},{ax_y}): {e}")

    print(f"  Summary plots → {summary_vis}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
