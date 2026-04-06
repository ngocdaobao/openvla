import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

class TemporalAlignProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, train: bool = True) -> None:
        super().__init__()
        hidden_dim = max(input_dim, output_dim)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        return self.projector(temporal_features)


def prepare_videoprism_inputs(
    video_frames: torch.Tensor,
    image_processor: PrismaticImageProcessor,
    camera_index: int = 0, # 0 for third-person view, 1 for wrist-camera view
) -> np.ndarray:
    print(f"Original video frames shape: {video_frames.shape}")

    if video_frames.ndim == 5:
        num_channels = video_frames.shape[2]
        if num_channels % 3 != 0:
            raise ValueError(
                f"Expected channel dimension to be RGB or concatenated RGB multiples of 3, got C={num_channels}"
            )

        num_cameras = num_channels // 3
        if not (0 <= camera_index < num_cameras):
            raise ValueError(
                f"camera_index={camera_index} out of bounds for {num_cameras} concatenated camera views"
            )

        if num_cameras > 1:
            start = camera_index * 3
            end = start + 3
            video_frames = video_frames[:, :, start:end, :, :]

    print(f"Using single-camera video shape: {video_frames.shape}")
    return video_frames.permute(0, 1, 3, 4, 2).contiguous().float().cpu().numpy()


def resize_token_sequence(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if features.shape[1] == target_tokens:
        return features
    features = features.transpose(1, 2)
    features = F.adaptive_avg_pool1d(features, target_tokens)
    return features.transpose(1, 2)


def compute_cosine_align_loss(projected_temporal: torch.Tensor, vision_hidden: torch.Tensor) -> torch.Tensor:
    projected_temporal = F.normalize(projected_temporal.float(), dim=-1)
    vision_hidden = F.normalize(vision_hidden.float(), dim=-1)
    return 1.0 - F.cosine_similarity(projected_temporal, vision_hidden, dim=-1).mean()
