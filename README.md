# ChemSeek-OCR

## Downloading Source Code

```bash
mkdir llm4chem
cd llm4chem
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
git clone https://github.com/HaCTang/ChemSeek-OCR.git
```

## Environment Setting
First build a conda environment following https://github.com/deepseek-ai/DeepSeek-OCR-2.

```bash
conda create -n chemseek-ocr python=3.12.9 -y
conda activate chemseek-ocr
```

Download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5). The file is at the bottom of the page.

```bash
cd DeepSeek-OCR-2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
pip install peft accelerate wandb
pip install matplotlib albumentations opencv-python rdkit SmilesPE pandas
```

**Note:** if you want vLLM and transformers codes to run in the same environment, you don't need to worry about this installation error like: vllm 0.8.5+cu118 requires transformers>=4.51.1

## Download Weight and Test DeepSeek-OCR-2

Assume that you are still in the directory of DeepSeek-OCR-2
```bash
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export HF_DATASETS_CACHE=$PWD/hf_cache
export HF_HUB_DISABLE_XET=1
```

You can do transformer-based inference and download the weight of DeepSeek-OCR-2 at the same time:
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_name = 'deepseek-ai/DeepSeek-OCR-2'
# model_name = 'model_file'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', use_safetensors=True, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Give me the smiles of the molecule. "
prompt = "<image>\n Give me the smiles of the molecule. "
image_file = 'test_img/penicillin.jpg'
output_path = 'output'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = True)
```

## Training Datasets
```bash
cd ../ChemSeek-OCR
```
Download the following benchmarks and unzip them into the folder **training_data**.
[pubchem](https://huggingface.co/yujieq/MolScribe/blob/main/pubchem.zip)
[uspto-mol](https://huggingface.co/yujieq/MolScribe/blob/main/uspto_mol.zip)

You can choose to use **dataset.py** to pre-render pictures in **pubchem**.

## Evaluation Checkpoint

### Download Benchmarks

Download the following benchmarks and unzip them into the folder **benchmark**.
[Synthetic](https://huggingface.co/yujieq/MolScribe/blob/main/synthetic.zip)
[Realistic](https://huggingface.co/yujieq/MolScribe/blob/main/real.zip)
[Perturbed](https://huggingface.co/yujieq/MolScribe/blob/main/perturb.zip)

### Run Evaluation

Please use **chemseek-ocr** for evaluation! Don't use **chemseek-ocr-verl**

Check **evaluation.py** and **evaluation_config.yaml**.

```bash
python evaluation.py --config evaluation_config.yaml
python gspo/evaluation_gspo.py --config gspo/evaluation_gspo_config.yaml
```

## Progressive SFT

The training of Progressive SFT consists of three parts: cold-start LORR SFT in a few-sample scenario, then merging the LoRA weights into the base model, and finally performing full-parameter SFT.

### LoRA SFT

See File **lora_sft.py** and **lora_sft_config.yaml**.

Default setting using 3*64k datapoints to do cold-start LoRA.

For training dataset setting in **lora_sft_config.yaml**, uspto-mol uses **realistic** data mode, while pubchem uses **dynamic** data mode. And pubchem can actually use augmentation. For example:

 ```yaml
 train_sets:
  - train_csv: ./training_data/pubchem/train_200k.csv
    data_mode: dynamic
    pre_rendered_image_dir: null
    realistic_image_root: null
    instruction: "<image>\n Give me the SMILES of the molecule. "
    style: molscribe_default # molscribe_default / chemdraw_like
    mol_augment: true
    include_condensed: False
    max_samples: null
    sample_num: 64000
  - train_csv: ./training_data/pubchem/train_200k.csv
    data_mode: dynamic
    pre_rendered_image_dir: null
    realistic_image_root: null
    instruction: "<image>\n Give me the SMILES of the molecule. "
    style: chemdraw_like # molscribe_default / chemdraw_like
    mol_augment: false
    include_condensed: False
    max_samples: null
    sample_num: 64000
  - train_csv: ./training_data/uspto_mol/train_200k.csv
    data_mode: realistic
    realistic_image_root: ./training_data
    instruction: "<image>\n Give me the SMILES of the molecule. "
    sample_num: 64000
```

For other parameters, you can change:

 ```yaml
...
batch_size: 4 -> 32
grad_accum: 8 -> 1
...
accelerate_num_processes: 4 -> 8
accelerate_gpu_ids: "0,1,2,3" -> "0,1,2,3,4,5,6,7,8"
...
```

You can run **lora_sft.py** with:

```bash
python lora_sft.py --config lora_sft_config.yaml
```

### Merging LoRA weight

```bash
python merge_lora_weight.py \
  --pretrained_weight_path ../DeepSeek-OCR-2 \
  --checkpoint_path ./weight/checkpoint-1500 \
  --merged_model_dir ./merged_models \
  --full_or_lora lora
```

### Full parameter SFT

Similar to LoRA SFT session, for training dataset setting in **lora_sft_config.yaml**, uspto-mol uses **realistic** data mode, while pubchem uses **dynamic** data mode. And pubchem can actually use augmentation. For example:

 ```yaml
 train_sets:
  - train_csv: ./training_data/pubchem/train_1m.csv
    data_mode: dynamic
    pre_rendered_image_dir: null
    realistic_image_root: null
    instruction: "<image>\n Give me the SMILES of the molecule. "
    style: molscribe_default # molscribe_default / chemdraw_like
    mol_augment: true
    include_condensed: False
    max_samples: null
    sample_num: null
  - train_csv: ./training_data/pubchem/train_1m.csv
    data_mode: dynamic
    pre_rendered_image_dir: null
    realistic_image_root: null
    instruction: "<image>\n Give me the SMILES of the molecule. "
    style: chemdraw_like # molscribe_default / chemdraw_like
    mol_augment: false
    include_condensed: False
    max_samples: null
    sample_num: null
  - train_csv: ./training_data/uspto_mol/train_680k.csv
    data_mode: realistic
    realistic_image_root: ./training_data
    instruction: "<image>\n Give me the SMILES of the molecule. "
    sample_num: null
```

The total batch size can be a little bit larger, around 512-2048.
Learning rate is around 1/10 to 1/100 compared with previous LoRA, like 5e-6.
Run 1-2 epochs (you should evaluate the steps).

You can run **progressive_sft.py** with:

```bash
python progressive_sft.py --config progressive_sft_config.yaml
```

## RL

The RL phase is divided into three progressive stages to ensure the stability of the MoE architecture while optimizing for chemical SMILES accuracy.

Download the verl-0.6.1 [Source Code](https://github.com/verl-project/verl/releases/tag/v0.6.1). The file is at the bottom of the page.

```bash
conda create -n chemseek-ocr-verl python=3.12
conda activate chemseek-ocr-verl
unzip verl-0.6.1.zip
cd verl-0.6.1
```
 
 In order to be compatible with the fine-tuning code of DeepSeek-OCR-2, please open **verl-0.6.1/setup.py**. Set "transformers==4.57" in the **install_requires**, and set ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm==0.8.5"] in the **VLLM_REQUIRES**.

```bash
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 #You can run this if error says libcudart.so.11.0
pip install -e .[vllm]
pip install flash-attn==2.7.3 --no-build-isolation
pip install matplotlib albumentations rdkit SmilesPE pandas addict
```

### GSPO RL

GSPO (Group Sequence Policy Optimization) uses tight symmetric clipping (`clip_ratio_low`/`clip_ratio_high`) instead of KL regularization, providing more stable policy updates. The implementation is built on the [verl](https://github.com/volcengine/verl) framework and is now organized under the `gspo/` directory:

| File | Description |
|------|-------------|
| **gspo/prepare_verl_data.py** | Reads CSV datasets, resolves image paths, and writes train/val parquet splits in verl-compatible format. |
| **gspo/gspo_rl_verl.py** | Defines the custom reward function (`compute_score`), multimodal dataset (`ChemSeekOCRDataset`), and assembles verl launch command with GSPO-specific Hydra overrides. |
| **gspo/gspo_rl_verl_config.yaml** | All hyperparameters: model path, data sources, GSPO clipping, reward weights, GPU/FSDP/vLLM settings, and logging. |
| **gspo/evaluation_gspo.py** | Merges verl FSDP checkpoints and evaluates them on configured OCR benchmarks. |
| **gspo/evaluation_gspo_config.yaml** | Benchmark selection and checkpoint/evaluation settings for GSPO models. |

Key GSPO parameters in the config (`gspo:` section):

- `clip_ratio_low` / `clip_ratio_high`: Tight symmetric clipping bounds (default 3e-4 / 4e-4) that replace KL penalty.
- `clip_ratio_c`: Upper clip bound for the importance-sampling ratio (default 10.0).
- `loss_agg_mode: seq-mean-token-mean`: Sequence-level then token-level mean aggregation.
- `use_kl_loss: false` / `kl_loss_coef: 0.0`: KL loss is disabled; tight clipping suffices.
- `group_size: 8`: Number of responses sampled per prompt for advantage estimation.
- `use_dynamic_bsz: true`: Dynamic batch sizing based on token budget.

The reward function computes five SMILES-based components (validity, tanimoto, canon_smiles, graph, chiral) with configurable weights, normalized to [0, 1] by dividing by the weight sum.

```bash
# Step 1: Prepare parquet data
python gspo/prepare_verl_data.py --config gspo/gspo_rl_verl_config.yaml --workers 8

# Step 2: Launch GSPO training
python gspo/gspo_rl_verl.py --config gspo/gspo_rl_verl_config.yaml

# Step 3: Evaluate a GSPO checkpoint
python gspo/evaluation_gspo.py --config gspo/evaluation_gspo_config.yaml
```

### ReFT

ReFT (Rejection sampling Fine-Tuning) is organized under the `reft/` directory and provides a two-phase pipeline: best-of-N generation with vLLM followed by SFT on the curated high-reward samples.

| File | Description |
|------|-------------|
| **reft/reft.py** | Runs the generation, scoring, filtering, and iterative fine-tuning pipeline. |
| **reft/reft_config.yaml** | Controls model path, sampling, reward thresholds, vLLM settings, and SFT hyperparameters. |

```bash
# Run the full ReFT pipeline
python reft/reft.py --config reft/reft_config.yaml

# Only generate best-of-N candidates
python reft/reft.py --config reft/reft_config.yaml --phase generate

# Only train a specific iteration
python reft/reft.py --config reft/reft_config.yaml --phase train --iteration 0
```

## Inference

For **transformer** inference, please use conda environment **chemseek-ocr**: 

```yaml
python transformer_infer_case.py --model-path ./weight_progressive_sft/checkpoint-2000 --image-file ./test_img/penicillin.jpg
```

## Citation

```
@misc{tang2026finetuningdeepseekocr2molecularstructure,
      title={Fine-tuning DeepSeek-OCR-2 for Molecular Structure Recognition}, 
      author={Haocheng Tang and Xingyu Dang and Junmei Wang},
      year={2026},
      eprint={2604.03476},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.03476}, 
}
```
