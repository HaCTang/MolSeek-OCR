# ChemSeek-OCR

## Downloading Source Code##

```bash
mkdir llm4chem
cd llm4chem
git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
git clone https://github.com/HaCTang/ChemSeek-OCR.git
```

## Environment Setting##
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
pip install matplotlib albumentations opencv-python rdkit SmilesPE
```
