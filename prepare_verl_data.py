"""Prepare verl-compatible parquet data from CSV datasets for DeepSeek-OCR-2.

Reads CSV files containing SMILES and image paths, resolves images on disk,
and writes train/val parquet splits in the format expected by verl's
ChemSeekOCRDataset.

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

def prepare_data(cfg: dict) -> None:
    """Convert CSV datasets to parquet format for verl GSPO training."""
    import pandas as pd

    output_dir = Path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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

            elif data_mode == "dynamic":
                print(
                    "Warning: dynamic mode requires chemseek-ocr env with rdkit renderer. "
                    "Please pre-render images first, then use pre_rendered mode."
                )
                continue

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

    for r in all_rows:
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

    train_path = output_dir / "train.parquet"
    pq.write_table(table, str(train_path))

    n_val = max(100, len(all_rows) // 50)
    val_rows = all_rows[-n_val:]
    val_table = pa.table({
        "prompt": [json.dumps([{"role": "user", "content": instruction}]) for _ in val_rows],
        "image_path": [r["image_path"] for r in val_rows],
        "data_source": ["chemseek_ocr"] * len(val_rows),
        "reward_model": [json.dumps({"ground_truth": r["ground_truth"]}) for r in val_rows],
    })
    val_path = output_dir / "val.parquet"
    pq.write_table(val_table, str(val_path))

    print(f"Saved {len(all_rows)} train samples to {train_path}")
    print(f"Saved {len(val_rows)} val samples to {val_path}")


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
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    prepare_data(cfg)


if __name__ == "__main__":
    main()
