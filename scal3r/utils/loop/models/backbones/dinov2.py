import torch

from torch import nn

from scal3r.utils.vggt.layers.vision_transformer import vit_base


dinov2_archs = {
    "dinov2_vitb14": 768,
}


class DINOv2(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        num_trainable_blocks: int = 2,
        norm_layer: bool = False,
        return_token: bool = False,
        local_pretrained_path: str = "",
        img_size: int = 518,
    ):
        super().__init__()
        if model_name not in dinov2_archs:
            raise ValueError(f"Unsupported DINOv2 model: {model_name}")

        self.model = vit_base(
            img_size=img_size,
            patch_size=14,
            init_values=1.0,
            block_chunks=0,
        )
        if local_pretrained_path:
            state_dict = torch.load(local_pretrained_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)

        self.num_channels = dinov2_archs[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x: torch.Tensor):
        batch_size, _, height, width = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        frozen_blocks = self.model.blocks[:-self.num_trainable_blocks]
        with torch.no_grad():
            for block in frozen_blocks:
                x = block(x)
        x = x.detach()

        for block in self.model.blocks[-self.num_trainable_blocks:]:
            x = block(x)

        if self.norm_layer:
            x = self.model.norm(x)

        token = x[:, 0]
        feat = x[:, 1:]
        feat = feat.reshape(batch_size, height // 14, width // 14, self.num_channels)
        feat = feat.permute(0, 3, 1, 2).contiguous()

        if self.return_token:
            return feat, token
        return feat
