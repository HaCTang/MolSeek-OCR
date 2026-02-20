import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from DeepSeek_OCR_2 import apply_transformers_compat_shims, resolve_local_model_path


def merge_and_save_model(
    pretrained_weight_path: str,
    checkpoint_path: str,
    merged_model_dir: str,
    full_or_lora: str = "lora",
) -> str:
    checkpoint_name = Path(checkpoint_path).name
    merged_dir = Path(merged_model_dir) / f"{full_or_lora}_{checkpoint_name}"
    marker_file = merged_dir / ".merge_complete"

    if marker_file.is_file():
        print(f"Using cached merged model: {merged_dir}")
        return str(merged_dir)

    if full_or_lora == "lora":
        print(f"Merging LoRA weights from {checkpoint_path} into base model...")
    else:
        print(f"Preparing full-finetuned model for vLLM from {checkpoint_path}...")
    apply_transformers_compat_shims()

    if full_or_lora == "lora":
        model_path = resolve_local_model_path(Path(pretrained_weight_path).resolve())
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_model = peft_model.merge_and_unload()

        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

        for src_file in model_path.glob("*.json"):
            dest = merged_dir / src_file.name
            if not dest.exists():
                shutil.copy2(str(src_file), str(dest))
    else:
        source_dir = Path(checkpoint_path).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Full checkpoint directory not found: {source_dir}")
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        shutil.copytree(source_dir, merged_dir)

    config_json_path = merged_dir / "config.json"
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    cfg_data["model_type"] = "deepseek_vl_v2"
    cfg_data.pop("auto_map", None)
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)

    for py_file in merged_dir.glob("*.py"):
        py_file.unlink()

    marker_file.touch()
    print(f"Saved merged model to: {merged_dir}")

    if full_or_lora == "lora":
        del merged_model, peft_model, base_model
        torch.cuda.empty_cache()

    return str(merged_dir)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint with base model for vLLM usage.")
    parser.add_argument(
        "--pretrained_weight_path",
        type=str,
        default=str((workspace / "../DeepSeek-OCR-2").resolve()),
        help="Path to base DeepSeek-OCR-2 model root.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory (LoRA adapter or full checkpoint).",
    )
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        default=str((workspace / "merged_models").resolve()),
        help="Directory to store merged models.",
    )
    parser.add_argument(
        "--full_or_lora",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Input checkpoint type: lora adapter or full model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_path = merge_and_save_model(
        pretrained_weight_path=str(Path(args.pretrained_weight_path).resolve()),
        checkpoint_path=str(Path(args.checkpoint_path).resolve()),
        merged_model_dir=str(Path(args.merged_model_dir).resolve()),
        full_or_lora=str(args.full_or_lora).strip().lower(),
    )
    print(f"Merged model path: {merged_path}")


if __name__ == "__main__":
    main()
