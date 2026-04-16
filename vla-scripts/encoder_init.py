from jepa.app.vjepa.utils import init_video_model
import torch

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

ckpt = torch.load("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/ngoc/openvla/jepa_ckpt/vitl16.pth.tar", 
                    map_location='cpu')

encoder_state_dict = {
    k.replace('module.backbone.', ''): v
    for k, v in ckpt['encoder'].items()
}
encoder.backbone.load_state_dict(encoder_state_dict)
print("LOAD SUCCESSFULLY")