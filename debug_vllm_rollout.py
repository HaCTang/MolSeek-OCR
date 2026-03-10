"""Minimal diagnostic: test vLLM DeepseekOCR2 with prompt_token_ids vs prompt string.

Usage:
    cd /work/hat170/ChemLLM/ChemSeek-OCR
    CUDA_VISIBLE_DEVICES=0 python debug_vllm_rollout.py
"""
import os
import sys
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "weight_progressive_sft", "checkpoint-2000")
VLLM_CODE_DIR = os.path.join(
    SCRIPT_DIR, "..", "DeepSeek-OCR-2", "DeepSeek-OCR2-master", "DeepSeek-OCR2-vllm"
)
INSTRUCTION = "<image>\n Give me the SMILES of the molecule. "

os.environ["VLLM_USE_V1"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- inject config module (same as evaluation.py) ----
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
config_mod = types.ModuleType("config")
config_mod.BASE_SIZE = 1024
config_mod.IMAGE_SIZE = 768
config_mod.CROP_MODE = True
config_mod.MIN_CROPS = 2
config_mod.MAX_CROPS = 6
config_mod.MAX_CONCURRENCY = 100
config_mod.NUM_WORKERS = 64
config_mod.PRINT_NUM_VIS_TOKENS = False
config_mod.SKIP_REPEAT = True
config_mod.MODEL_PATH = MODEL_PATH
config_mod.INPUT_PATH = ""
config_mod.OUTPUT_PATH = ""
config_mod.PROMPT = INSTRUCTION
config_mod.TOKENIZER = tokenizer
config_mod._chemseek_injected = True
sys.modules["config"] = config_mod

if VLLM_CODE_DIR not in sys.path:
    sys.path.insert(0, VLLM_CODE_DIR)

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from PIL import Image

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    block_size=256,
    enforce_eager=True,
    trust_remote_code=True,
    max_model_len=4096,
    swap_space=0,
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

# Pick a test image from the training data
import pandas as pd

parquet_path = os.path.join(SCRIPT_DIR, "verl_data", "train.parquet")
df = pd.read_parquet(parquet_path)
test_row = df.iloc[0]
image_path = test_row.get("image_path", "")
print(f"Test image: {image_path}")
print(f"Gold SMILES: {test_row.get('reward_model', '???')}")

image = Image.open(image_path).convert("RGB")
processor = DeepseekOCR2Processor()
tokenized = processor.tokenize_with_images(
    images=[image], bos=True, eos=True, cropping=True
)

# ---- Test 1: prompt string (same as evaluation.py) ----
print("\n=== Test 1: prompt STRING (like evaluation.py) ===")
inputs_str = [{"prompt": INSTRUCTION, "multi_modal_data": {"image": tokenized}}]
outputs_str = llm.generate(inputs_str, sampling_params)
pred_str = outputs_str[0].outputs[0].text.strip()
final_ids_str = outputs_str[0].prompt_token_ids
print(f"  pred: {pred_str[:200]}")
print(f"  prompt_token_ids len: {len(final_ids_str)}")
print(f"  first 10 ids: {final_ids_str[:10]}")
print(f"  <image> count: {final_ids_str.count(128815)}")

# ---- Test 2: prompt_token_ids WITHOUT bos (like current gspo) ----
print("\n=== Test 2: prompt_token_ids NO BOS (current gspo) ===")
raw_prompt_ids = tokenizer.encode(INSTRUCTION, add_special_tokens=False)
print(f"  raw_prompt_ids: {raw_prompt_ids}")
print(f"  length: {len(raw_prompt_ids)}")
inputs_ids = [
    {"prompt_token_ids": raw_prompt_ids, "multi_modal_data": {"image": tokenized}}
]
outputs_ids = llm.generate(inputs_ids, sampling_params)
pred_ids = outputs_ids[0].outputs[0].text.strip()
final_ids_2 = outputs_ids[0].prompt_token_ids
print(f"  pred: {pred_ids[:200]}")
print(f"  final prompt_token_ids len: {len(final_ids_2)}")
print(f"  first 10 ids: {final_ids_2[:10]}")
print(f"  <image> count: {final_ids_2.count(128815)}")

# ---- Test 3: prompt_token_ids WITH bos ----
print("\n=== Test 3: prompt_token_ids WITH BOS ===")
bos_id = tokenizer.bos_token_id
print(f"  bos_token_id: {bos_id}")
raw_prompt_ids_bos = [bos_id] + raw_prompt_ids if bos_id is not None else raw_prompt_ids
print(f"  raw_prompt_ids_bos: {raw_prompt_ids_bos}")
inputs_bos = [
    {"prompt_token_ids": raw_prompt_ids_bos, "multi_modal_data": {"image": tokenized}}
]
outputs_bos = llm.generate(inputs_bos, sampling_params)
pred_bos = outputs_bos[0].outputs[0].text.strip()
final_ids_3 = outputs_bos[0].prompt_token_ids
print(f"  pred: {pred_bos[:200]}")
print(f"  final prompt_token_ids len: {len(final_ids_3)}")
print(f"  first 10 ids: {final_ids_3[:10]}")
print(f"  <image> count: {final_ids_3.count(128815)}")

# ---- Test 4: use tokenize_with_images input_ids directly ----
print("\n=== Test 4: tokenize_with_images input_ids as prompt_token_ids ===")
import torch
mm_input_ids = tokenized[0][0]
if isinstance(mm_input_ids, torch.Tensor):
    mm_input_ids = mm_input_ids.tolist()
print(f"  tokenize_with_images input_ids len: {len(mm_input_ids)}")
print(f"  first 10: {mm_input_ids[:10]}")
print(f"  <image> count: {mm_input_ids.count(128815)}")
inputs_mm = [
    {"prompt_token_ids": mm_input_ids, "multi_modal_data": {"image": tokenized}}
]
outputs_mm = llm.generate(inputs_mm, sampling_params)
pred_mm = outputs_mm[0].outputs[0].text.strip()
final_ids_4 = outputs_mm[0].prompt_token_ids
print(f"  pred: {pred_mm[:200]}")
print(f"  final prompt_token_ids len: {len(final_ids_4)}")
print(f"  <image> count: {final_ids_4.count(128815)}")

# ---- Compare ----
print(f"\n=== Results Summary ===")
print(f"  Test 1 (string):             {pred_str[:80]}")
print(f"  Test 2 (token_ids, no BOS):  {pred_ids[:80]}")
print(f"  Test 3 (token_ids, BOS):     {pred_bos[:80]}")
print(f"  Test 4 (mm input_ids):       {pred_mm[:80]}")
