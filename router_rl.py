import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from calc_accuracy import canonicalize_smiles, tanimoto_similarity
from dataset import DEFAULT_INSTRUCTION
from DeepSeek_OCR_2 import (
    _apply_subset_sampling,
    _build_conversation_dataset,
    _resolve_path,
    apply_deepseek_runtime_patches,
    apply_transformers_compat_shims,
    bf16_supported,
    is_main_process,
    load_modeling_utils_from_loaded_model,
    maybe_init_wandb,
    resolve_local_model_path,
    set_seed,
)


@dataclass
class TrainSetConfig:
    train_csv: str
    data_mode: str
    pre_rendered_image_dir: str | None
    realistic_image_root: str | None
    instruction: str
    style: str | None
    mol_augment: bool | None
    include_condensed: bool | None
    max_samples: int | None
    sample_num: int | None


@dataclass
class RestartConfig:
    enable: bool
    dir: str
    auto_resume_from_latest_checkpoint: bool
    resume_from_checkpoint: str | None
    wandb_resume_same_run: bool


@dataclass
class RewardWeightConfig:
    validity: float
    tanimoto: float
    canon_smiles: float
    graph: float
    chiral: float


@dataclass
class RouterRLConfig:
    pretrained_weight_path: str
    train_sets: List[TrainSetConfig]
    seed: int
    output_dir: str
    batch_size: int
    grad_accum: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    allow_tf32: bool
    load_in_4bit: bool
    attn_implementation: str
    image_size: int
    base_size: int
    crop_mode: bool
    dataloader_num_workers: int
    dataloader_persistent_workers: bool
    enable_gradient_checkpointing: bool
    train_on_responses_only: bool
    generation_num: int
    generation_max_new_tokens: int
    generation_temperature: float
    generation_top_p: float
    grpo_beta: float
    grpo_adv_epsilon: float
    use_reference_model: bool
    reward_weights: RewardWeightConfig
    chiral_no_annotation_reward: float
    expert_trainable_patterns: List[str]
    use_accelerate: bool
    accelerate_num_processes: int
    accelerate_gpu_ids: str | None
    restart: RestartConfig
    save_steps: int
    log_steps: int
    wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_api_key: str | None


class RouterRLDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        messages = item["messages"]
        if len(messages) < 2:
            raise ValueError("Expected at least 2 messages (user + assistant) in base dataset sample.")
        user_message = dict(messages[0])
        gold_smiles = str(messages[-1]["content"]).strip()
        return {"user_message": user_message, "gold_smiles": gold_smiles}


class PromptOnlyDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return features


class DeepSeekPromptBuilder:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        text_encode: Any,
        basic_image_transform_cls: Any,
        dynamic_preprocess_fn: Any,
        image_size: int,
        base_size: int,
        crop_mode: bool,
        train_on_responses_only: bool,
    ):
        from DeepSeek_OCR_2 import DeepSeekOCR2DataCollator

        self.collator = DeepSeekOCR2DataCollator(
            tokenizer=tokenizer,
            model=model,
            text_encode=text_encode,
            basic_image_transform_cls=basic_image_transform_cls,
            dynamic_preprocess_fn=dynamic_preprocess_fn,
            image_size=image_size,
            base_size=base_size,
            crop_mode=crop_mode,
            train_on_responses_only=train_on_responses_only,
        )

    def build_prompt_inputs(self, user_message: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        processed = self.collator.process_single_sample([user_message])
        input_ids = processed["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        images_seq_mask = processed["images_seq_mask"].unsqueeze(0).to(device)
        images_spatial_crop = processed["images_spatial_crop"].to(device)
        images = [(processed["images_crop"].to(device), processed["images_ori"].to(device))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

    def build_completion_inputs(
        self, user_message: Dict[str, Any], completion_text: str, device: torch.device
    ) -> Dict[str, Any]:
        processed = self.collator.process_single_sample(
            [user_message, {"role": "<|Assistant|>", "content": completion_text}]
        )
        input_ids = processed["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        images_seq_mask = processed["images_seq_mask"].unsqueeze(0).to(device)
        images_spatial_crop = processed["images_spatial_crop"].to(device)
        images = [(processed["images_crop"].to(device), processed["images_ori"].to(device))]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "prompt_token_count": int(processed["prompt_token_count"]),
        }


def _is_distributed_launched() -> bool:
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("RANK") is not None:
        return True
    if os.environ.get("ACCELERATE_PROCESS_INDEX") is not None:
        return True
    return False


def _maybe_launch_with_accelerate(cfg: RouterRLConfig, cli_args: argparse.Namespace) -> None:
    if not cfg.use_accelerate:
        return
    if _is_distributed_launched():
        return
    if os.environ.get("CHEMSEEK_ACCELERATE_LAUNCHED") == "1":
        return
    if cfg.accelerate_num_processes <= 1:
        return

    child_env = os.environ.copy()
    child_env["CHEMSEEK_ACCELERATE_LAUNCHED"] = "1"
    if cfg.accelerate_gpu_ids not in (None, ""):
        child_env["CUDA_VISIBLE_DEVICES"] = str(cfg.accelerate_gpu_ids)

    script_path = str(Path(__file__).resolve())
    command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(cfg.accelerate_num_processes),
        script_path,
        "--config",
        str(Path(cli_args.config).resolve()),
    ]
    print(f"Launching with accelerate: {' '.join(command)}")
    subprocess.run(command, env=child_env, check=True)
    sys.exit(0)


def _extract_smiles_candidate(text: str) -> str:
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""
    line = raw.splitlines()[0].strip()
    if line.startswith("```"):
        line = line.strip("`").strip()
    line = line.replace(" ", "")
    return line


def _replace_empty(smiles: str) -> str:
    return smiles if isinstance(smiles, str) and smiles != "" else "<empty>"


def _compute_reward_components(
    gold_smiles: str,
    pred_smiles: str,
    *,
    chiral_no_annotation_reward: float,
) -> Dict[str, float]:
    canon_gold, _ = canonicalize_smiles(gold_smiles, ignore_cistrans=True)
    canon_pred, valid_pred = canonicalize_smiles(pred_smiles, ignore_cistrans=True)
    graph_gold, _ = canonicalize_smiles(gold_smiles, ignore_chiral=True, ignore_cistrans=True)
    graph_pred, _ = canonicalize_smiles(pred_smiles, ignore_chiral=True, ignore_cistrans=True)

    canon_gold = _replace_empty(canon_gold)
    canon_pred = _replace_empty(canon_pred)
    graph_gold = _replace_empty(graph_gold)
    graph_pred = _replace_empty(graph_pred)

    has_chiral = "@" in canon_gold
    chiral_acc = float(canon_gold == canon_pred) if has_chiral else float(chiral_no_annotation_reward)

    components = {
        "validity": float(valid_pred),
        "tanimoto": float(tanimoto_similarity(gold_smiles, pred_smiles)),
        "canon_smiles": float(canon_gold == canon_pred),
        "graph": float(graph_gold == graph_pred),
        "chiral": chiral_acc,
    }
    return components


def _combine_reward(components: Dict[str, float], weights: RewardWeightConfig) -> float:
    total = (
        weights.validity * components["validity"]
        + weights.tanimoto * components["tanimoto"]
        + weights.canon_smiles * components["canon_smiles"]
        + weights.graph * components["graph"]
        + weights.chiral * components["chiral"]
    )
    return float(total)


def _set_expert_trainable_parameters(model: Any, patterns: List[str]) -> None:
    normalized_patterns = [p.lower().strip() for p in patterns if str(p).strip()]
    if not normalized_patterns:
        normalized_patterns = ["mlp.experts.", "mlp.shared_experts."]

    trainable_count = 0
    total_count = 0
    trainable_names: List[str] = []
    for name, param in model.named_parameters():
        total_count += param.numel()
        lowered = name.lower()
        should_train = any(pattern in lowered for pattern in normalized_patterns)
        if ".mlp.gate" in lowered:
            # Routing Replay: always freeze gate/router.
            should_train = False
        param.requires_grad = should_train
        if should_train:
            trainable_count += param.numel()
            if len(trainable_names) < 24:
                trainable_names.append(name)

    ratio = 100.0 * float(trainable_count) / float(max(total_count, 1))
    print(f"Expert trainable params: {trainable_count:,} / {total_count:,} ({ratio:.4f}%)")
    if trainable_names:
        preview = "\n".join([f"  - {n}" for n in trainable_names])
        print(f"Trainable parameter name preview:\n{preview}")

    if trainable_count == 0:
        fallback_patterns = ["mlp.experts."]
        print(
            "No parameters matched configured expert patterns. "
            f"Trying fallback patterns: {fallback_patterns}"
        )
        trainable_count = 0
        total_count = 0
        trainable_names = []
        for name, param in model.named_parameters():
            total_count += param.numel()
            lowered = name.lower()
            should_train = any(pattern in lowered for pattern in fallback_patterns)
            if ".mlp.gate" in lowered:
                should_train = False
            param.requires_grad = should_train
            if should_train:
                trainable_count += param.numel()
                if len(trainable_names) < 24:
                    trainable_names.append(name)

        ratio = 100.0 * float(trainable_count) / float(max(total_count, 1))
        print(f"Fallback expert trainable params: {trainable_count:,} / {total_count:,} ({ratio:.4f}%)")
        if trainable_names:
            preview = "\n".join([f"  - {n}" for n in trainable_names])
            print(f"Fallback trainable parameter name preview:\n{preview}")

    if trainable_count == 0:
        raise RuntimeError(
            "No trainable parameters matched expert_trainable_patterns and fallback patterns. "
            f"Patterns: {normalized_patterns}, fallback: ['mlp.experts.']"
        )


class RoutingReplayController:
    """
    Routing Replay:
      1) rollout/capture: record each MoE gate output (topk_idx/topk_weight)
      2) update/replay: bypass gate computation and force cached routing
    """

    def __init__(self, model: Any):
        self.model = model
        self._gate_modules: List[tuple[str, Any]] = []
        self._orig_gate_forward: Dict[int, Any] = {}
        self._patched = False
        self._discover_and_patch()

    def _discover_and_patch(self) -> None:
        for name, module in self.model.named_modules():
            if name.endswith(".mlp.gate"):
                self._gate_modules.append((name, module))
        if not self._gate_modules:
            raise RuntimeError("No MoE gate modules found (expected names ending with '.mlp.gate').")

        import types

        for _name, gate in self._gate_modules:
            self._orig_gate_forward[id(gate)] = gate.forward

            def _patched_forward(this_gate, hidden_states, _orig=self._orig_gate_forward[id(gate)]):
                mode = getattr(this_gate, "_rr_mode", "off")
                if mode == "replay":
                    payload = getattr(this_gate, "_rr_payload", None)
                    if payload is None:
                        raise RuntimeError("Routing replay mode enabled but no cached payload is attached.")
                    topk_idx, topk_weight = payload
                    # DeepseekV2MoE training path unconditionally applies AddAuxiliaryLoss,
                    # so we must provide a scalar tensor (not None) for aux_loss.
                    aux_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
                    return (
                        topk_idx.to(hidden_states.device),
                        topk_weight.to(hidden_states.device),
                        aux_loss,
                    )

                out = _orig(hidden_states)
                if mode == "capture":
                    topk_idx, topk_weight, _aux_loss = out
                    # Keep on CPU to reduce replay memory pressure.
                    this_gate._rr_payload = (topk_idx.detach().cpu(), topk_weight.detach().cpu())
                return out

            gate.forward = types.MethodType(_patched_forward, gate)
            gate._rr_mode = "off"
            gate._rr_payload = None
        self._patched = True
        print(f"Routing replay controller attached to {len(self._gate_modules)} MoE gates.")

    def _set_mode(self, mode: str) -> None:
        for _, gate in self._gate_modules:
            gate._rr_mode = mode

    def capture(self, model_inputs: Dict[str, Any]) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if not self._patched:
            raise RuntimeError("Routing replay controller is not patched.")

        for _, gate in self._gate_modules:
            gate._rr_payload = None
        self._set_mode("capture")
        with torch.no_grad():
            _ = self.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                images=model_inputs["images"],
                images_seq_mask=model_inputs["images_seq_mask"],
                images_spatial_crop=model_inputs["images_spatial_crop"],
            )
        self._set_mode("off")

        cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, gate in self._gate_modules:
            payload = getattr(gate, "_rr_payload", None)
            if payload is None:
                raise RuntimeError(f"Failed to capture routing payload for gate: {name}")
            cache[name] = payload
        return cache

    def enable_replay(self, cache: Dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        for name, gate in self._gate_modules:
            if name not in cache:
                raise RuntimeError(f"Missing routing replay cache for gate: {name}")
            gate._rr_payload = cache[name]
        self._set_mode("replay")

    def disable_replay(self) -> None:
        self._set_mode("off")


def _iter_dataloader_forever(loader: DataLoader) -> Iterator[List[Dict[str, Any]]]:
    while True:
        for batch in loader:
            yield batch


def _build_optimizer(cfg: RouterRLConfig, model: Any) -> AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for optimizer.")
    use_fused = bool(torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    return AdamW(
        params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=use_fused if use_fused else False,
    )


def _build_scheduler(cfg: RouterRLConfig, optimizer: AdamW):
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    def lr_lambda(current_step: int) -> float:
        if cfg.warmup_steps > 0 and current_step < cfg.warmup_steps:
            return float(current_step + 1) / float(max(cfg.warmup_steps, 1))
        remain = cfg.max_steps - current_step
        total_decay = max(cfg.max_steps - cfg.warmup_steps, 1)
        return max(float(remain) / float(total_decay), 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_completion_logprob(model: Any, completion_inputs: Dict[str, Any]) -> torch.Tensor:
    outputs = model(
        input_ids=completion_inputs["input_ids"],
        attention_mask=completion_inputs["attention_mask"],
        images=completion_inputs["images"],
        images_seq_mask=completion_inputs["images_seq_mask"],
        images_spatial_crop=completion_inputs["images_spatial_crop"],
    )
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = completion_inputs["input_ids"][:, 1:]
    # DeepSeek's lm_head returns float32 logits by default; casting to model dtype
    # and using CE avoids an explicit full log_softmax allocation.
    target_dtype = torch.bfloat16 if shift_logits.device.type == "cuda" else shift_logits.dtype
    shift_logits = shift_logits.to(target_dtype)
    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    token_log_probs = -token_nll

    prompt_token_count = int(completion_inputs["prompt_token_count"])
    start_idx = max(prompt_token_count - 1, 0)
    completion_mask = torch.arange(shift_labels.shape[1], device=shift_labels.device) >= start_idx
    completion_mask = completion_mask.unsqueeze(0)
    selected = token_log_probs[completion_mask]
    if selected.numel() == 0:
        return token_log_probs.mean()
    return selected.mean()


def _compute_kl_like_penalty(
    model: Any,
    ref_model: Any,
    completion_inputs: Dict[str, Any],
    routing_replay: RoutingReplayController | None = None,
    routing_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    if routing_replay is not None and routing_cache is not None:
        routing_replay.enable_replay(routing_cache)
        try:
            current_logp = _compute_completion_logprob(model, completion_inputs)
        finally:
            routing_replay.disable_replay()
    else:
        current_logp = _compute_completion_logprob(model, completion_inputs)
    with torch.no_grad():
        ref_logp = _compute_completion_logprob(ref_model, completion_inputs)
    return (current_logp - ref_logp).pow(2)


def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt_inputs: Dict[str, Any],
    cfg: RouterRLConfig,
) -> str:
    device = prompt_inputs["input_ids"].device
    input_ids = prompt_inputs["input_ids"]
    images_seq_mask = prompt_inputs["images_seq_mask"]
    images = prompt_inputs["images"]
    images_spatial_crop = prompt_inputs["images_spatial_crop"]
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    generated_ids: List[int] = []

    def _sample_from_logits(next_logits: torch.Tensor) -> int:
        if cfg.generation_temperature <= 0:
            return int(torch.argmax(next_logits, dim=-1).item())
        scaled = next_logits / max(cfg.generation_temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if cfg.generation_top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumsum > cfg.generation_top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / torch.clamp(sorted_probs.sum(dim=-1, keepdim=True), min=1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            token_id = sorted_idx.gather(-1, sampled).squeeze(-1)
            return int(token_id.item())
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return int(token_id.item())

    with torch.no_grad():
        for _ in range(cfg.generation_max_new_tokens):
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_id = _sample_from_logits(next_token_logits)
            generated_ids.append(next_token_id)

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            next_mask_tensor = torch.zeros((1, 1), dtype=torch.bool, device=device)
            images_seq_mask = torch.cat([images_seq_mask, next_mask_tensor], dim=1)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    # Trim eos/pad tail for cleaner reward text.
    while generated_ids and generated_ids[-1] in {eos_token_id, pad_token_id}:
        generated_ids.pop()

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return _extract_smiles_candidate(text)


def _build_train_dataset(cfg: RouterRLConfig) -> Dataset:
    base_sets: List[Dataset] = []
    for ts in cfg.train_sets:
        train_df = pd.read_csv(ts.train_csv)
        train_df = _apply_subset_sampling(
            train_df,
            sample_num=ts.sample_num,
            max_samples=ts.max_samples,
            data_mode=ts.data_mode,
            seed=cfg.seed,
        )
        base_dataset = _build_conversation_dataset(
            dataframe=train_df,
            data_mode=ts.data_mode,
            instruction=ts.instruction,
            style=ts.style,
            mol_augment=ts.mol_augment,
            include_condensed=ts.include_condensed,
            pre_rendered_image_dir=ts.pre_rendered_image_dir,
            realistic_image_root=ts.realistic_image_root,
            use_rendered_smiles_as_label=True,
        )
        base_sets.append(base_dataset)
    merged = base_sets[0] if len(base_sets) == 1 else torch.utils.data.ConcatDataset(base_sets)
    return RouterRLDataset(merged)


def _to_device(model: Any) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_config(cfg: RouterRLConfig) -> None:
    if cfg.attn_implementation not in ("eager", "flash_attention_2"):
        raise ValueError("attn_implementation must be 'eager' or 'flash_attention_2'.")
    if cfg.image_size != 768:
        raise ValueError("image_size must be 768 for DeepSeek-OCR-2.")
    if cfg.base_size not in (768, 1024):
        raise ValueError("base_size must be 768 or 1024.")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if cfg.grad_accum <= 0:
        raise ValueError("grad_accum must be > 0.")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if cfg.save_steps <= 0:
        raise ValueError("save_steps must be > 0.")
    if cfg.log_steps <= 0:
        raise ValueError("log_steps must be > 0.")
    if cfg.generation_num <= 1:
        raise ValueError("generation_num must be > 1 for GRPO group normalization.")
    if cfg.generation_max_new_tokens <= 0:
        raise ValueError("generation_max_new_tokens must be > 0.")
    if not cfg.train_sets:
        raise ValueError("train_sets must contain at least one dataset.")
    for i, ts in enumerate(cfg.train_sets):
        if ts.data_mode not in ("dynamic", "pre_rendered", "realistic"):
            raise ValueError(f"train_sets[{i}].data_mode must be dynamic/pre_rendered/realistic.")
        if not os.path.isfile(ts.train_csv):
            raise ValueError(f"train_sets[{i}].train_csv not found: {ts.train_csv}")
        if ts.data_mode == "pre_rendered":
            if not ts.pre_rendered_image_dir or not os.path.isdir(ts.pre_rendered_image_dir):
                raise ValueError(f"train_sets[{i}].pre_rendered_image_dir is invalid.")
        if ts.data_mode == "realistic":
            if not ts.realistic_image_root or not os.path.isdir(ts.realistic_image_root):
                raise ValueError(f"train_sets[{i}].realistic_image_root is invalid.")


def load_config(config_path: str) -> RouterRLConfig:
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

    with open(config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config root must be a mapping/object.")
    if "pretrained_weight_path" not in raw and "model_root" in raw:
        raw["pretrained_weight_path"] = raw["model_root"]
    if "pretrained_weight_path" not in raw:
        raise ValueError("Missing required config key: pretrained_weight_path")

    raw["pretrained_weight_path"] = _resolve_path(raw["pretrained_weight_path"], config_file.parent)
    raw["output_dir"] = _resolve_path(raw.get("output_dir", "./weight_router_rl"), config_file.parent)
    raw["batch_size"] = int(raw.get("batch_size", 2))
    raw["grad_accum"] = int(raw.get("grad_accum", 4))
    raw["max_steps"] = int(raw.get("max_steps", 500))
    raw["warmup_steps"] = int(raw.get("warmup_steps", 50))
    raw["learning_rate"] = float(raw.get("learning_rate", 2e-6))
    raw["weight_decay"] = float(raw.get("weight_decay", 0.0))
    raw["max_grad_norm"] = float(raw.get("max_grad_norm", 1.0))
    raw["allow_tf32"] = bool(raw.get("allow_tf32", True))
    raw["load_in_4bit"] = bool(raw.get("load_in_4bit", False))
    raw["attn_implementation"] = raw.get("attn_implementation", "flash_attention_2")
    raw["image_size"] = int(raw.get("image_size", 768))
    raw["base_size"] = int(raw.get("base_size", 1024))
    raw["crop_mode"] = bool(raw.get("crop_mode", True))
    raw["dataloader_num_workers"] = int(raw.get("dataloader_num_workers", 4))
    raw["dataloader_persistent_workers"] = bool(raw.get("dataloader_persistent_workers", False))
    raw["enable_gradient_checkpointing"] = bool(raw.get("enable_gradient_checkpointing", True))
    raw["train_on_responses_only"] = bool(raw.get("train_on_responses_only", True))
    raw["generation_num"] = int(raw.get("generation_num", 4))
    raw["generation_max_new_tokens"] = int(raw.get("generation_max_new_tokens", 96))
    raw["generation_temperature"] = float(raw.get("generation_temperature", 0.8))
    raw["generation_top_p"] = float(raw.get("generation_top_p", 0.95))
    raw["grpo_beta"] = float(raw.get("grpo_beta", 0.02))
    raw["grpo_adv_epsilon"] = float(raw.get("grpo_adv_epsilon", 1e-6))
    raw["use_reference_model"] = bool(raw.get("use_reference_model", False))
    raw["chiral_no_annotation_reward"] = float(raw.get("chiral_no_annotation_reward", 1.0))
    raw["expert_trainable_patterns"] = list(
        raw.get("expert_trainable_patterns", raw.get("router_trainable_patterns", ["mlp.experts."]))
    )
    raw["use_accelerate"] = bool(raw.get("use_accelerate", False))
    raw["accelerate_num_processes"] = int(raw.get("accelerate_num_processes", 1))
    raw["accelerate_gpu_ids"] = raw.get("accelerate_gpu_ids", None)
    raw["save_steps"] = int(raw.get("save_steps", 50))
    raw["log_steps"] = int(raw.get("log_steps", 5))
    raw["wandb"] = bool(raw.get("wandb", False))
    raw["wandb_project"] = str(raw.get("wandb_project", "ChemSeek-OCR"))
    raw["wandb_run_name"] = raw.get("wandb_run_name", None)
    if raw.get("wandb_api_key") in ("", None):
        raw["wandb_api_key"] = os.getenv("WANDB_API_KEY", None)

    reward_raw = dict(raw.get("reward_weights", {}) or {})
    reward_raw.setdefault("validity", 2.0)
    reward_raw.setdefault("tanimoto", 1.0)
    reward_raw.setdefault("canon_smiles", 2.0)
    reward_raw.setdefault("graph", 1.5)
    reward_raw.setdefault("chiral", 1.5)
    raw["reward_weights"] = RewardWeightConfig(**reward_raw)

    restart_raw = dict(raw.get("restart", {}) or {})
    restart_raw.setdefault("enable", False)
    restart_raw.setdefault("dir", raw["output_dir"])
    restart_raw.setdefault("auto_resume_from_latest_checkpoint", False)
    restart_raw.setdefault("resume_from_checkpoint", None)
    restart_raw.setdefault("wandb_resume_same_run", False)
    restart_raw["dir"] = _resolve_path(restart_raw["dir"], config_file.parent)
    raw["restart"] = RestartConfig(**restart_raw)

    if "train_sets" not in raw:
        raw["train_sets"] = [
            {
                "train_csv": raw.get("train_csv"),
                "data_mode": raw.get("data_mode", "dynamic"),
                "pre_rendered_image_dir": raw.get("pre_rendered_image_dir"),
                "realistic_image_root": raw.get("realistic_image_root"),
                "instruction": raw.get("instruction", DEFAULT_INSTRUCTION),
                "style": raw.get("style", "molscribe_default"),
                "mol_augment": raw.get("mol_augment", True),
                "include_condensed": raw.get("include_condensed", True),
                "max_samples": raw.get("max_samples", None),
                "sample_num": raw.get("sample_num", None),
            }
        ]
    parsed_train_sets: List[TrainSetConfig] = []
    for item in raw["train_sets"]:
        item = dict(item)
        item["train_csv"] = _resolve_path(item["train_csv"], config_file.parent)
        if item.get("pre_rendered_image_dir"):
            item["pre_rendered_image_dir"] = _resolve_path(item["pre_rendered_image_dir"], config_file.parent)
        if item.get("realistic_image_root"):
            item["realistic_image_root"] = _resolve_path(item["realistic_image_root"], config_file.parent)
        if item.get("instruction") in ("", None):
            item["instruction"] = DEFAULT_INSTRUCTION
        item.setdefault("style", None)
        item.setdefault("mol_augment", None)
        item.setdefault("include_condensed", None)
        item.setdefault("max_samples", None)
        item.setdefault("sample_num", None)
        item.setdefault("pre_rendered_image_dir", None)
        item.setdefault("realistic_image_root", None)
        parsed_train_sets.append(TrainSetConfig(**item))
    raw["train_sets"] = parsed_train_sets

    for legacy_key in [
        "train_csv",
        "data_mode",
        "pre_rendered_image_dir",
        "realistic_image_root",
        "instruction",
        "style",
        "mol_augment",
        "include_condensed",
        "max_samples",
        "sample_num",
        "router_trainable_patterns",
    ]:
        raw.pop(legacy_key, None)

    cfg = RouterRLConfig(**raw)
    _validate_config(cfg)
    return cfg


def parse_cli_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="GRPO router RL for DeepSeek-OCR-2.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(workspace / "router_rl_config.yaml"),
        help="Path to router rl YAML config.",
    )
    args, _ = parser.parse_known_args()
    return args


def _save_checkpoint(model: Any, tokenizer: Any, output_dir: str, step: int) -> None:
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))


def _log_wandb(payload: Dict[str, float], step: int) -> None:
    if "wandb" not in sys.modules:
        return
    try:
        import wandb  # type: ignore[import-not-found]

        wandb.log(payload, step=step)
    except Exception:
        pass


def main() -> None:
    cli_args = parse_cli_args()
    cfg = load_config(cli_args.config)
    _maybe_launch_with_accelerate(cfg, cli_args)

    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    set_seed(cfg.seed)
    apply_transformers_compat_shims()
    os.makedirs(cfg.output_dir, exist_ok=True)
    maybe_init_wandb(cfg)

    model_root = Path(cfg.pretrained_weight_path).resolve()
    model_path = resolve_local_model_path(model_root)
    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
        quantization_config=quantization_config,
    )
    apply_deepseek_runtime_patches(model)
    model.config.use_cache = True
    if cfg.enable_gradient_checkpointing:
        # Routing Replay reuses cached gate decisions between rollout and update passes.
        # torch checkpoint recomputation can break this assumption and emits large
        # metadata dumps ("saved metadata / recomputed metadata"). Disable it here.
        print("Routing Replay mode: forcing gradient checkpointing off for stable replay.")
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    text_encode, basic_image_transform_cls, dynamic_preprocess_fn = load_modeling_utils_from_loaded_model(model)
    prompt_builder = DeepSeekPromptBuilder(
        tokenizer=tokenizer,
        model=model,
        text_encode=text_encode,
        basic_image_transform_cls=basic_image_transform_cls,
        dynamic_preprocess_fn=dynamic_preprocess_fn,
        image_size=cfg.image_size,
        base_size=cfg.base_size,
        crop_mode=cfg.crop_mode,
        train_on_responses_only=cfg.train_on_responses_only,
    )

    _set_expert_trainable_parameters(model, cfg.expert_trainable_patterns)
    device = _to_device(model)
    model.to(device)
    routing_replay = RoutingReplayController(model)

    reference_model = None
    if cfg.use_reference_model and cfg.grpo_beta > 0:
        reference_model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=torch.bfloat16 if bf16_supported() else torch.float16,
            quantization_config=quantization_config,
        )
        apply_deepseek_runtime_patches(reference_model)
        reference_model.config.use_cache = True
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False
        reference_model.to(device)
        print("Reference model enabled for KL-like GRPO regularization.")
    elif cfg.grpo_beta > 0 and not cfg.use_reference_model:
        print("Warning: grpo_beta > 0 but use_reference_model=false, KL-like regularization is disabled.")

    train_dataset = _build_train_dataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        persistent_workers=cfg.dataloader_persistent_workers and cfg.dataloader_num_workers > 0,
        collate_fn=PromptOnlyDataCollator(),
        drop_last=True,
    )
    stream = _iter_dataloader_forever(train_loader)

    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    print(
        f"Starting GRPO Routing-Replay RL: max_steps={cfg.max_steps}, batch_size={cfg.batch_size}, "
        f"grad_accum={cfg.grad_accum}, generations={cfg.generation_num}"
    )

    running_total_loss = 0.0
    running_reward = 0.0
    running_validity = 0.0
    running_tanimoto = 0.0
    running_canon = 0.0
    running_graph = 0.0
    running_chiral = 0.0

    model.train()
    progress = tqdm(
        range(1, cfg.max_steps + 1),
        total=cfg.max_steps,
        desc="RoutingReplay-GRPO",
        dynamic_ncols=True,
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        accum_loss_value = 0.0
        accum_reward_value = 0.0
        accum_component_sums = {
            "validity": 0.0,
            "tanimoto": 0.0,
            "canon_smiles": 0.0,
            "graph": 0.0,
            "chiral": 0.0,
        }
        accum_component_count = 0

        for _ in range(cfg.grad_accum):
            batch = next(stream)
            micro_losses: List[torch.Tensor] = []

            for sample in batch:
                user_message = sample["user_message"]
                gold_smiles = str(sample["gold_smiles"])

                prompt_inputs = prompt_builder.build_prompt_inputs(user_message, device)
                completions: List[str] = []
                rewards: List[float] = []
                completion_logprobs: List[torch.Tensor] = []
                completion_routing_caches: List[Dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
                component_records: List[Dict[str, float]] = []

                for _gen_idx in range(cfg.generation_num):
                    completion = _generate_completion(model, tokenizer, prompt_inputs, cfg)
                    completions.append(completion)

                    components = _compute_reward_components(
                        gold_smiles,
                        completion,
                        chiral_no_annotation_reward=cfg.chiral_no_annotation_reward,
                    )
                    reward_value = _combine_reward(components, cfg.reward_weights)
                    rewards.append(reward_value)
                    component_records.append(components)

                    completion_inputs = prompt_builder.build_completion_inputs(user_message, completion, device)
                    routing_cache = routing_replay.capture(completion_inputs)
                    routing_replay.enable_replay(routing_cache)
                    try:
                        logp = _compute_completion_logprob(model, completion_inputs)
                    finally:
                        routing_replay.disable_replay()
                    completion_logprobs.append(logp)
                    completion_routing_caches.append(routing_cache)

                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + cfg.grpo_adv_epsilon)

                sample_losses: List[torch.Tensor] = []
                for j in range(cfg.generation_num):
                    term = -advantages[j].detach() * completion_logprobs[j]
                    if reference_model is not None and cfg.grpo_beta > 0:
                        completion_inputs = prompt_builder.build_completion_inputs(
                            user_message, completions[j], device
                        )
                        if j < len(completion_routing_caches):
                            kl_like = _compute_kl_like_penalty(
                                model,
                                reference_model,
                                completion_inputs,
                                routing_replay=routing_replay,
                                routing_cache=completion_routing_caches[j],
                            )
                        else:
                            kl_like = _compute_kl_like_penalty(model, reference_model, completion_inputs)
                        term = term + cfg.grpo_beta * kl_like
                    sample_losses.append(term)
                sample_loss = torch.stack(sample_losses).mean()
                micro_losses.append(sample_loss)

                accum_reward_value += float(sum(rewards) / max(len(rewards), 1))
                for c in component_records:
                    for key in accum_component_sums:
                        accum_component_sums[key] += float(c[key])
                    accum_component_count += 1

            if not micro_losses:
                continue
            micro_loss = torch.stack(micro_losses).mean()
            loss_for_backward = micro_loss / cfg.grad_accum
            loss_for_backward.backward()
            accum_loss_value += float(loss_for_backward.detach().item())

        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.max_grad_norm,
            )
        optimizer.step()
        scheduler.step()

        running_total_loss += accum_loss_value
        step_reward = accum_reward_value / max(cfg.grad_accum, 1)
        running_reward += step_reward
        if accum_component_count > 0:
            running_validity += accum_component_sums["validity"] / accum_component_count
            running_tanimoto += accum_component_sums["tanimoto"] / accum_component_count
            running_canon += accum_component_sums["canon_smiles"] / accum_component_count
            running_graph += accum_component_sums["graph"] / accum_component_count
            running_chiral += accum_component_sums["chiral"] / accum_component_count

        # Real-time progress bar stats (updated every step).
        progress.set_postfix(
            loss=f"{accum_loss_value:.4f}",
            reward=f"{step_reward:.4f}",
            tanimoto=f"{(accum_component_sums['tanimoto'] / max(accum_component_count, 1)):.4f}",
        )

        if step % cfg.log_steps == 0:
            denom = float(cfg.log_steps)
            metrics = {
                "train/loss": running_total_loss / denom,
                "train/reward": running_reward / denom,
                "train/validity": running_validity / denom,
                "train/tanimoto": running_tanimoto / denom,
                "train/canon_smiles": running_canon / denom,
                "train/graph": running_graph / denom,
                "train/chiral": running_chiral / denom,
                "train/lr": float(scheduler.get_last_lr()[0]),
            }
            print(
                f"[step {step}] "
                f"loss={metrics['train/loss']:.6f} "
                f"reward={metrics['train/reward']:.4f} "
                f"valid={metrics['train/validity']:.4f} "
                f"tanimoto={metrics['train/tanimoto']:.4f} "
                f"canon={metrics['train/canon_smiles']:.4f} "
                f"graph={metrics['train/graph']:.4f} "
                f"chiral={metrics['train/chiral']:.4f}"
            )
            if cfg.wandb and is_main_process():
                _log_wandb(metrics, step=step)
            running_total_loss = 0.0
            running_reward = 0.0
            running_validity = 0.0
            running_tanimoto = 0.0
            running_canon = 0.0
            running_graph = 0.0
            running_chiral = 0.0

        if step % cfg.save_steps == 0 and is_main_process():
            _save_checkpoint(model, tokenizer, cfg.output_dir, step)

    if is_main_process():
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved router RL model to: {cfg.output_dir}")
    progress.close()


if __name__ == "__main__":
    main()
