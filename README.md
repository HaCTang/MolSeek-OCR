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

Check **evaluation.py** and **evaluation_config.yaml**.

```bash
python evaluation.py --config evaluation_config.yaml
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
  - train_csv: ./training_data/uspto_mol/train_200k.csv
    data_mode: realistic
    realistic_image_root: ./training_data
    instruction: "<image>\n Give me the SMILES of the molecule. "
    sample_num: null
```

The total batch size can be a little bit larger, around 512-2048.
Learning rate is around 1/10 to 1/100 compared with previous LoRA, like 5e-6.
Run 1-2 epochs (you should evaluate the steps).

You can run **full_sft.py** with:

```bash
python full_sft.py --config full_sft_config.yaml
```

## RL

The RL phase is divided into three progressive stages to ensure the stability of the MoE architecture while optimizing for chemical SMILES accuracy.

### Routing Replay RL

· Freeze router
· Replay routing
· Train expert only
Goal: Make experts sensitive to reward signals.

### Soft Replay RL

A trade-off approach:
· Forward: Use current router.
· Backward: Apply stop-gradient to the router.

### Full Parameter RL

After experts stabilize:
· Disable replay.
· Unfreeze router.
· Set $lr_{router} = 0.01 \sim 0.1 \times lr_{main}$.
· Strengthen balance + entropy regularization.
