import argparse
import os

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR inference with HF model id or local checkpoint folder."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR-2",
        help=(
            "Model source path. Supports Hugging Face model id or local folder "
            "such as weight_progressive_sft/checkpoint-2000."
        ),
    )
    parser.add_argument(
        "--image-file",
        type=str,
        default="test_img/penicillin.jpg",
        help="Input image path for inference.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\n Give me the smiles of the molecule. ",
        help="Text prompt. Keep <image> token for image-conditioned inference.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output",
        help="Directory to save inference artifacts.",
    )
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument(
        "--crop-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable crop_mode in model.infer.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save visualized inference outputs.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES before loading the model, e.g. '0'.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        _attn_implementation="flash_attention_2",
        use_safetensors=True,
        trust_remote_code=True,
    )
    return model, tokenizer


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    os.makedirs(args.output_path, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.eval().cuda().to(dtype)

    result = model.infer(
        tokenizer,
        prompt=args.prompt,
        image_file=args.image_file,
        output_path=args.output_path,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        save_results=args.save_results,
    )
    print(result)


if __name__ == "__main__":
    main()
