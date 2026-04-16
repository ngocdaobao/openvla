"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# HuggingFace default IGNORE_INDEX for CrossEntropyLoss labels
IGNORE_INDEX = -100

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=self.get_action_dim(unnorm_key),
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]

class OpenVLA_Temporal_Finetune(OpenVLA):
    def __init__(
        self,
        *args,
        encoder,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        resize_length: int = 198,
        layers: list = [-1],
        **kwargs,
    ):
        super().__init__(*args, norm_stats=norm_stats, action_tokenizer=action_tokenizer, **kwargs)

        self.encoder = encoder
        self.resize_length = resize_length
        self.patch_dim = self.encoder.num_tokens * self.encoder.embed_dim // self.resize_length

        self.proj = nn.Linear(self.patch_dim, self.llm_backbone.embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.requires_grad_(True)
    
    def forward(
        self,
        use_encoder: bool = False,
        cache_history: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            output = self.llm_backbone(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")

        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        elif len(multimodal_indices) == 0:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

        # Extract temporal information
        temporal_embed = None
        if use_encoder and cache_history is not None:
            # Stack list of (B, 3, H, W) frames → (B, 3, T, H, W) video tensor
            video = torch.stack(cache_history, dim=2).to(projected_patch_embeddings.device)
            with torch.no_grad():
                temporal_embed = self.encoder(video)
            temporal_embed = temporal_embed.reshape(temporal_embed.shape[0], self.encoder.num_tokens * self.encoder.embed_dim)
            temporal_embed = temporal_embed.reshape(temporal_embed.shape[0], self.resize_length, self.patch_dim)
            temporal_embed = self.proj(temporal_embed)
        elif use_encoder and cache_history is None:
            raise ValueError("use_encoder=True but cache_history is None")


        # Build Multimodal Embeddings (and build resulting attention mask)
        embed_parts = [input_embeddings[multimodal_indices, :1, :]]
        if temporal_embed is not None:
            embed_parts.append(temporal_embed)
        embed_parts += [projected_patch_embeddings, input_embeddings[multimodal_indices, 1:, :]]
        multimodal_embeddings = torch.cat(embed_parts, dim=1)

        multimodal_attention_mask = None
        if attention_mask is not None:
            mask_parts = [attention_mask[multimodal_indices, :1]]
            if temporal_embed is not None:
                temporal_attention_mask = torch.full(
                    (temporal_embed.shape[0], temporal_embed.shape[1]),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                mask_parts.append(temporal_attention_mask)
            mask_parts += [projected_patch_attention_mask, attention_mask[multimodal_indices, 1:]]
            multimodal_attention_mask = torch.cat(mask_parts, dim=1)

        # [Contract] We assume the first token of `labels` (associated with <BOS>) is already marked as "IGNORE"
        #   => We'll ignore the per-token outputs for each of the patch/temporal embeddings as well!
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            label_parts = [labels[multimodal_indices, :1]]
            if temporal_embed is not None:
                temporal_labels = torch.full(
                    (temporal_embed.shape[0], temporal_embed.shape[1]),
                    IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                label_parts.append(temporal_labels)
            label_parts += [projected_patch_labels, labels[multimodal_indices, 1:]]
            multimodal_labels = torch.cat(label_parts, dim=1)

        # === Add Unimodal Handling ===

        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels

        else:
            # Otherwise --> Merge w/ unimodal data

            # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
            #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
            # Pad width must match the full extra tokens inserted into multimodal sequences
            #   (temporal tokens + vision patch tokens)
            num_pad_tokens = projected_patch_embeddings.shape[1] + (temporal_embed.shape[1] if temporal_embed is not None else 0)
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), num_pad_tokens, input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_attention_pad = torch.full(
                (len(unimodal_indices), num_pad_tokens),
                False,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), num_pad_tokens),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_attention_mask = torch.cat([attention_mask[unimodal_indices], unimodal_attention_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_attention_mask = torch.vstack([multimodal_attention_mask, unimodal_attention_mask])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


