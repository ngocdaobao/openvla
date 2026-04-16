"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.models.load import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from jepa.app.vjepa.utils import init_video_model

import gc
gc.collect
torch.cuda.empty_cache()

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/ngoc/openvla-7b"  # Local HF dir | local .pt file | native ID on openvla/openvla-dev
                                                                    #   OR absolute path to a local .pt checkpoint file

    # Directory Paths
    data_root_dir: Path = Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/ngoc/modified_libero_rlds")        # Path to Open-X dataset directory
    dataset_name: str = "libero_goal_no_noops"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 8                                             # Fine-tuning batch size
    max_steps: int = 25_000                                         # Max number of fine-tuning steps
    save_steps: int = 25_000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Not supported with native OpenVLA loading

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "ngoc-db-vinuniversity"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    # JEPA config
    use_encoder: bool = False
    ckpt_path: str = "jepa_ckpt/vitl16.pth.tar"
    jepa_patch_size: int = 16
    jepa_num_frames: int = 16
    tubelet_sizeL: int = 2
    jepa_model_name: str = "vit_large"
    frame_crop_size: int = 224
    camera_use: int = 0         # 0 for primary camera, 1 for wrist camera


def load_openvla_from_hf_local(encoder, hf_model_path: str, load_for_training: bool = False):
    """Load a native OpenVLA from a locally downloaded HF-format directory (e.g. openvla/openvla-7b).

    Remaps HF SafeTensors weights (fc1/fc2/fc3 projector, language_model.* LLM,
    vision_backbone.featurizer.* DINO, vision_backbone.fused_featurizer.* SigLIP)
    into native Prismatic key names so the OpenVLA class can be subclassed freely.
    """
    import shutil, tempfile
    from transformers import AutoModelForVision2Seq, AutoConfig, AutoImageProcessor, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
    from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    from prismatic.models.vlas import OpenVLA, OpenVLA_Temporal_Finetune

    # Register HF AutoClasses so from_pretrained resolves correctly
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load HF model from local SafeTensors (no internet needed after initial download)
    hf_model = AutoModelForVision2Seq.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    cfg_hf = hf_model.config  # OpenVLAConfig

    # Build native vision backbone (TIMM weights — likely already cached)
    vision_backbone, _ = get_vision_backbone_and_transform(
        cfg_hf.vision_backbone_id,
        cfg_hf.image_resize_strategy,
    )

    # Build native LLM backbone without accessing the gated meta-llama repo.
    # Strategy: save the LlamaConfig (already embedded in cfg_hf.text_config) and
    # the local tokenizer files to a temp dir, then point Prismatic's LLM backbone
    # at that dir instead of "meta-llama/Llama-2-7b-hf".
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save LlamaConfig extracted from OpenVLAConfig (no network call)
        cfg_hf.text_config.save_pretrained(tmpdir)

        # Copy tokenizer files that are already present in the local HF directory
        for fname in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                      "special_tokens_map.json", "added_tokens.json"]:
            src = Path(hf_model_path) / fname
            if src.exists():
                shutil.copy(str(src), tmpdir)

        # Temporarily redirect Prismatic's LLM registry to the local temp dir
        orig_path = LLAMA2_MODELS[cfg_hf.llm_backbone_id]["hf_hub_path"]
        LLAMA2_MODELS[cfg_hf.llm_backbone_id]["hf_hub_path"] = tmpdir
        try:
            llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
                cfg_hf.llm_backbone_id,
                llm_max_length=cfg_hf.llm_max_length,
                inference_mode=True,   # creates empty LLM from config; weights loaded below
            )
        finally:
            LLAMA2_MODELS[cfg_hf.llm_backbone_id]["hf_hub_path"] = orig_path  # always restore

    action_tokenizer = ActionTokenizer(tokenizer)

    # Instantiate native OpenVLA (weights will be overwritten)
    vla = OpenVLA_Temporal_Finetune(
        cfg_hf.vision_backbone_id,
        vision_backbone,
        llm_backbone,
        encoder=encoder,
        arch_specifier=cfg_hf.arch_specifier,
        norm_stats=cfg_hf.norm_stats,
        action_tokenizer=action_tokenizer,
    )

    # --- Remap HF state dict → native Prismatic format ---
    # Projector: HF uses named attrs (fc1/fc2/fc3), native uses Sequential indices (0/2/4)
    HF_TO_NATIVE_PROJ = {
        "projector.fc1.weight": "projector.0.weight",
        "projector.fc1.bias":   "projector.0.bias",
        "projector.fc2.weight": "projector.2.weight",
        "projector.fc2.bias":   "projector.2.bias",
        "projector.fc3.weight": "projector.4.weight",
        "projector.fc3.bias":   "projector.4.bias",
    }

    proj_sd, llm_sd, vis_sd = {}, {}, {}
    for k, v in hf_model.state_dict().items():
        if k in HF_TO_NATIVE_PROJ:
            proj_sd[HF_TO_NATIVE_PROJ[k]] = v
        elif k.startswith("language_model."):
            llm_sd[k.replace("language_model.", "llm.")] = v
        elif cfg_hf.use_fused_vision_backbone:
            if k.startswith("vision_backbone.featurizer."):
                # DINO half — HF patches LayerScale gamma→scale_factor, reverse it
                nk = k.replace("vision_backbone.featurizer.", "dino_featurizer.")
                nk = nk.replace(".scale_factor", ".gamma")
                vis_sd[nk] = v
            elif k.startswith("vision_backbone.fused_featurizer."):
                # SigLIP half
                vis_sd[k.replace("vision_backbone.fused_featurizer.", "siglip_featurizer.")] = v
        else:
            if k.startswith("vision_backbone.featurizer."):
                vis_sd[k.replace("vision_backbone.featurizer.", "featurizer.")] = v

    vla.projector.load_state_dict(proj_sd)
    vla.llm_backbone.load_state_dict(llm_sd)
    vla.vision_backbone.load_state_dict(vis_sd, strict=False)

    # Set training mode flags that inference_mode=True skipped
    if load_for_training:
        vla.llm_backbone.llm.config.use_cache = False
        vla.llm_backbone.llm.enable_input_require_grads()
        vla.train()
    else:
        vla.requires_grad_(False)
        vla.eval()

    return vla


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    assert not cfg.use_quantization, "Quantization is not supported with native OpenVLA loading!"

    # Init JEPA encoder
    if cfg.use_encoder:
        encoder, _ = init_video_model(
            # device='cuda',
            patch_size=16,
            num_frames=16,        # must match T
            tubelet_size=2,
            model_name='vit_large', # or vit_base, vit_large, etc.
            crop_size=224,         # must match H and W
            pred_depth=12,
            pred_embed_dim=384,
        )    

        ckpt = torch.load(cfg.ckpt_path, map_location='cpu')
        # Checkpoint was saved with DDP (adds 'module.') + MultiMaskWrapper (adds 'backbone.')
        # Strip both prefixes to match encoder.backbone (VisionTransformer) key names
        encoder_state_dict = {
            k.replace('module.backbone.', ''): v
            for k, v in ckpt['encoder'].items()
        }
        encoder.backbone.load_state_dict(encoder_state_dict)
        encoder.to(device_id)
        encoder.eval()
        encoder.requires_grad_(False)
    else:
        encoder = None

    # Load native OpenVLA model
    if os.path.isdir(cfg.vla_path):
        # Local HF-format directory (e.g. downloaded from openvla/openvla-7b)
        vla = load_openvla_from_hf_local(encoder, cfg.vla_path, load_for_training=True)
    else:
        # Local .pt file OR native model ID in openvla/openvla-dev HF repo
        vla = load_vla(cfg.vla_path, load_for_training=True)

    # Extract tokenizer, image_transform, and other attributes BEFORE LoRA/DDP wrapping
    tokenizer = vla.llm_backbone.tokenizer
    image_transform = vla.vision_backbone.image_transform
    prompt_builder_fn = vla.llm_backbone.prompt_builder_fn
    num_patches = vla.vision_backbone.num_patches
    resize_resolution = vla.vision_backbone.default_image_resolution[1:]  # (H, W)

    # Device Placement
    vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            # "all-linear" requires PreTrainedModel; list LLaMA-2 linear layer names explicitly
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        pin_memory=True
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        cache_history = []
        use_encoder_in_vla = False
        for batch_idx, batch in enumerate(dataloader):
            # pixel_values is a dict {"dino": Tensor, "siglip": Tensor} for fused backbones,
            # or a plain Tensor for single-backbone models.
            raw_pv = batch["pixel_values"]
            if isinstance(raw_pv, dict):
                # Concat pixel values for vla
                pv_for_vla  = {k: v.to(torch.bfloat16).to(device_id) for k, v in raw_pv.items()}
                pv_for_jepa = raw_pv["dino"]   # (B, 3, H, W) — single camera, no channel slicing needed
            else:
                pv_for_vla  = raw_pv.to(torch.bfloat16).to(device_id)
                # Legacy 6-channel concat: channels 0-2 = primary, 3-5 = wrist
                pv_for_jepa = raw_pv[:, :3] if cfg.camera_use == 0 else raw_pv[:, 3:]

            if encoder is not None:
                if len(cache_history) < 16:
                    cache_history.append(pv_for_jepa.to(device_id))  # accumulate (B, 3, H, W) on GPU
                    use_encoder_in_vla = False
                else:
                    use_encoder_in_vla = True

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    use_encoder=use_encoder_in_vla,
                    cache_history=cache_history,
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=pv_for_vla,
                    labels=batch["labels"],
                )
                loss = output.loss
            
            if use_encoder_in_vla:
                cache_history = []  # Reset for next temporal window


            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            temporal_offset = vla.module.resize_length if use_encoder_in_vla else 0
            action_logits = output.logits[:, num_patches + temporal_offset : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save LoRA adapter weights (PEFT format) or full weights
                    if cfg.use_lora:
                        vla.module.save_pretrained(save_dir)   # saves adapter weights only
                    else:
                        os.makedirs(save_dir / "checkpoints", exist_ok=True)
                        model_state = {
                            "projector": vla.module.projector.state_dict(),
                            "llm_backbone": vla.module.llm_backbone.state_dict(),
                        }
                        if cfg.use_encoder:
                            model_state["temporal_proj"] = vla.module.proj.state_dict()
                        torch.save(
                            {"model": model_state},
                            save_dir / "checkpoints" / "latest-checkpoint.pt",
                        )

                # Wait for weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = load_vla(cfg.vla_path, load_for_training=True)
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint (native .pt format)
                            os.makedirs(run_dir / "checkpoints", exist_ok=True)
                            model_state = {
                                "projector": merged_vla.projector.state_dict(),
                                "llm_backbone": merged_vla.llm_backbone.state_dict(),
                            }
                            if cfg.use_encoder:
                                model_state["temporal_proj"] = vla.module.proj.state_dict()
                            torch.save(
                                {"model": model_state},
                                run_dir / "checkpoints" / "latest-checkpoint.pt",
                            )
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir / "checkpoints", exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save model weights to new directory (native .pt format)
                            model_state = {
                                "projector": merged_vla.projector.state_dict(),
                                "llm_backbone": merged_vla.llm_backbone.state_dict(),
                            }
                            if cfg.use_encoder:
                                model_state["temporal_proj"] = vla.module.proj.state_dict()
                            torch.save(
                                {"model": model_state},
                                checkpoint_dir / "checkpoints" / "latest-checkpoint.pt",
                            )
                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
