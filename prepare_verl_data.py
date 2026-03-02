"""Prepare verl-compatible parquet data from CSV datasets for DeepSeek-OCR-2.

Reads CSV files containing SMILES and image paths, resolves images on disk,
and writes train/val parquet splits in the format expected by verl's
ChemSeekOCRDataset.

Supported data_mode values:
  - pre_rendered: use existing images from a directory
  - realistic:    use file_path column to locate existing images
  - dynamic:      render SMILES to images on the fly via RDKit (dataset.py)

Output columns (verl format):
  - prompt:       JSON-encoded list of chat messages
  - image_path:   absolute path to the molecule image
  - data_source:  str identifier
  - reward_model: JSON-encoded ground truth dict

Usage:
    python prepare_verl_data.py --config gspo_rl_verl_config.yaml
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

_FIND_SMILES_COLUMNS = [
    "SMILES", "smiles", "Smiles", "canonical_smiles",
    "canon_smiles", "smi", "isosmiles", "input",
]

_ID_CANDIDATE_COLUMNS = [
    "image_id", "id", "index", "file_name", "image_file",
    "sample_id", "mol_id", "compound_id",
]


# =========================================================================
# Configuration
# =========================================================================

def _resolve_path(path_str: str, base_dir: Optional[Path] = None) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir or SCRIPT_DIR) / p
    return str(p.resolve())


def load_yaml_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = config_path.parent
    if "model" in cfg:
        cfg["model"]["path"] = _resolve_path(cfg["model"].get("path", ""), base)
    if "output_dir" in cfg:
        cfg["output_dir"] = _resolve_path(cfg["output_dir"], base)
    for ts in cfg.get("data", {}).get("train_sets", []):
        ts["train_csv"] = _resolve_path(ts["train_csv"], base)
        if ts.get("pre_rendered_image_dir"):
            ts["pre_rendered_image_dir"] = _resolve_path(ts["pre_rendered_image_dir"], base)
        if ts.get("realistic_image_root"):
            ts["realistic_image_root"] = _resolve_path(ts["realistic_image_root"], base)
    cfg.setdefault("data", {})
    cfg["data"]["output_dir"] = _resolve_path(
        cfg["data"].get("output_dir", "./verl_data"), base
    )
    save_cfg = cfg["data"].get("save", {}) or {}
    if save_cfg.get("train_file"):
        save_cfg["train_file"] = _resolve_path(save_cfg["train_file"], base)
    if save_cfg.get("val_file"):
        save_cfg["val_file"] = _resolve_path(save_cfg["val_file"], base)
    cfg["data"]["save"] = save_cfg
    return cfg


# =========================================================================
# Helpers
# =========================================================================

def _find_smiles_col(columns: list) -> str:
    for c in _FIND_SMILES_COLUMNS:
        if c in columns:
            return c
    raise ValueError(f"Cannot find SMILES column in: {columns}")


def _sanitize_filename(raw: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(raw))
    return safe.strip("_") or "sample"


def _resolve_sample_id(row, fallback_idx: int) -> str:
    import pandas as pd
    for col in _ID_CANDIDATE_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, float) and val.is_integer():
                raw = str(int(val))
            else:
                raw = str(val).strip()
            if raw:
                return _sanitize_filename(raw)
    return str(fallback_idx)


def _resolve_with_alt_ext(path: str) -> Optional[str]:
    if os.path.isfile(path):
        return path
    stem, _ = os.path.splitext(path)
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp"):
        candidate = f"{stem}{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


# =========================================================================
# Data Preparation
# =========================================================================

def _build_style_config_dict(ts: dict) -> dict:
    """Extract MoleculeStyleConfig kwargs from a train_set config entry."""
    return {
        "render_style": ts.get("style", "molscribe_default"),
        "mol_augment": ts.get("mol_augment", True),
        "include_condensed": ts.get("include_condensed", True),
    }


_WORKER_RENDERER = None


def _init_render_worker(style_config_dict: dict) -> None:
    """Initializer for each worker process in the pool."""
    global _WORKER_RENDERER
    from dataset import MoleculeStyleConfig, MoleculeStyleRenderer
    _WORKER_RENDERER = MoleculeStyleRenderer(MoleculeStyleConfig(**style_config_dict))


def _render_one(args: tuple) -> dict:
    """Render a single SMILES to an image file. Runs inside a worker process."""
    global _WORKER_RENDERER
    sid, smiles, img_out = args
    if os.path.isfile(img_out):
        return {"sid": sid, "smiles": smiles, "image_path": img_out, "ok": True}
    try:
        img, _ = _WORKER_RENDERER.render(smiles)
        img.save(img_out)
        return {"sid": sid, "smiles": smiles, "image_path": img_out, "ok": True}
    except Exception as e:
        return {"sid": sid, "smiles": smiles, "image_path": None, "ok": False, "err": str(e)}


def prepare_data(cfg: dict, num_workers: int = 1) -> None:
    """Convert CSV datasets to parquet format for verl GSPO training."""
    import pandas as pd

    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_cfg = cfg.get("data", {}).get("save", {}) or {}
    train_path = Path(save_cfg.get("train_file") or (output_dir / "train.parquet"))
    val_path = Path(save_cfg.get("val_file") or (output_dir / "val.parquet"))
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("training", {}).get("seed", 3407)
    instruction = cfg.get("data", {}).get(
        "instruction", "<image>\n Give me the SMILES of the molecule. "
    )

    all_rows: List[Dict[str, str]] = []
    for ts in cfg["data"].get("train_sets", []):
        csv_path = ts["train_csv"]
        data_mode = ts.get("data_mode", "pre_rendered")
        sample_num = ts.get("sample_num")

        df = pd.read_csv(csv_path)
        if sample_num and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

        smiles_col = _find_smiles_col(list(df.columns))
        found, skipped = 0, 0

        if data_mode == "dynamic":
            from concurrent.futures import ProcessPoolExecutor

            style_dict = _build_style_config_dict(ts)
            style = style_dict["render_style"]
            csv_stem = Path(csv_path).stem
            dynamic_img_dir = str(output_dir / f"dynamic_{csv_stem}_{style}")
            os.makedirs(dynamic_img_dir, exist_ok=True)

            tasks = []
            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                smiles_val = str(row[smiles_col]).strip()
                sid = _resolve_sample_id(row, row_idx)
                img_out = os.path.join(dynamic_img_dir, f"{sid}.png")
                tasks.append((sid, smiles_val, img_out))

            n_workers = max(1, num_workers)
            print(f"  [dynamic] Rendering {len(tasks)} images with "
                  f"{n_workers} workers, style={style}, output={dynamic_img_dir}")

            err_shown = 0
            if n_workers == 1:
                _init_render_worker(style_dict)
                for task in tasks:
                    result = _render_one(task)
                    if result["ok"]:
                        all_rows.append({
                            "image_path": result["image_path"],
                            "ground_truth": result["smiles"],
                        })
                        found += 1
                        if found % 2000 == 0:
                            print(f"    rendered {found}/{len(tasks)} ...")
                    else:
                        if err_shown < 5:
                            print(f"    render failed ({result['sid']}): {result.get('err', '')}")
                            err_shown += 1
                        skipped += 1
            else:
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_render_worker,
                    initargs=(style_dict,),
                ) as pool:
                    for result in pool.map(_render_one, tasks, chunksize=32):
                        if result["ok"]:
                            all_rows.append({
                                "image_path": result["image_path"],
                                "ground_truth": result["smiles"],
                            })
                            found += 1
                            if found % 2000 == 0:
                                print(f"    rendered {found}/{len(tasks)} ...")
                        else:
                            if err_shown < 5:
                                print(f"    render failed ({result['sid']}): {result.get('err', '')}")
                                err_shown += 1
                            skipped += 1

        else:
            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                image_path: Optional[str] = None

                if data_mode == "pre_rendered":
                    img_dir = ts.get("pre_rendered_image_dir", "")
                    sid = _resolve_sample_id(row, row_idx)
                    raw = os.path.join(img_dir, f"{sid}.png")
                    image_path = _resolve_with_alt_ext(raw)

                elif data_mode == "realistic":
                    img_root = ts.get("realistic_image_root", "")
                    if "file_path" in row.index:
                        rel = str(row["file_path"]).strip()
                        raw = rel if os.path.isabs(rel) else os.path.join(img_root, rel)
                        image_path = _resolve_with_alt_ext(raw)

                if image_path is None:
                    skipped += 1
                    continue

                gold_smiles = str(row[smiles_col])
                all_rows.append({
                    "image_path": image_path,
                    "ground_truth": gold_smiles,
                })
                found += 1

        print(f"  [{data_mode}] {csv_path}: found={found}, skipped={skipped}")

    if not all_rows:
        print("Error: No valid samples found. Check image paths and data_mode.")
        return

    random.seed(seed)
    random.shuffle(all_rows)

    prompts = []
    image_paths = []
    data_sources = []
    ground_truths = []

    data_cfg = cfg.get("data", {})
    val_split_cfg = data_cfg.get("val_split", {})
    val_split_enabled = bool(val_split_cfg.get("enabled", True))
    val_ratio = float(val_split_cfg.get("ratio", 0.02))
    val_min_samples = int(val_split_cfg.get("min_samples", 100))

    if val_split_enabled and len(all_rows) > 1:
        n_val = max(val_min_samples, int(len(all_rows) * val_ratio))
        n_val = min(n_val, len(all_rows) - 1)
        train_rows = all_rows[:-n_val]
        val_rows = all_rows[-n_val:]
    else:
        train_rows = all_rows
        val_rows = []

    for r in train_rows:
        prompts.append(json.dumps(
            [{"role": "user", "content": instruction}]
        ))
        image_paths.append(r["image_path"])
        data_sources.append("chemseek_ocr")
        ground_truths.append(json.dumps({"ground_truth": r["ground_truth"]}))

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "prompt": prompts,
        "image_path": image_paths,
        "data_source": data_sources,
        "reward_model": ground_truths,
    })

    pq.write_table(table, str(train_path))

    if val_rows:
        val_table = pa.table({
            "prompt": [json.dumps([{"role": "user", "content": instruction}]) for _ in val_rows],
            "image_path": [r["image_path"] for r in val_rows],
            "data_source": ["chemseek_ocr"] * len(val_rows),
            "reward_model": [json.dumps({"ground_truth": r["ground_truth"]}) for r in val_rows],
        })
        pq.write_table(val_table, str(val_path))
        print(
            f"Saved {len(train_rows)} train samples to {train_path}; "
            f"{len(val_rows)} val samples to {val_path} (ratio={val_ratio})"
        )
    else:
        if val_path.exists():
            val_path.unlink()
        print(f"Saved {len(train_rows)} train samples to {train_path}; validation split disabled")


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare verl-compatible parquet data for DeepSeek-OCR-2"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(SCRIPT_DIR / "gspo_rl_verl_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for dynamic rendering "
             "(default: from config data.num_workers, or 4)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    n_workers = args.workers or cfg.get("data", {}).get("num_workers", 4)
    prepare_data(cfg, num_workers=n_workers)


if __name__ == "__main__":
    main()
