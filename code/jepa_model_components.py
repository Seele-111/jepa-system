"""Small project-owned architecture adapter for the pinned I-JEPA backbone.

Upstream Block/PatchEmbed/mask implementations are imported, never vendored.
The checkpoint-compatible layer names and preprocessing match the established
local scorer: patch-only ViT-g, LayerNorm eps=1e-5, 40 blocks, no CLS token.
"""
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from src.models.vision_transformer import Block, PatchEmbed
from src.masks.utils import apply_masks


class PatchOnlyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=1408)
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 196, 1408), requires_grad=False)
        block_config = dict(dim=1408, num_heads=16, mlp_ratio=6144 / 1408, qkv_bias=True)
        self.blocks = torch.nn.ModuleList(
            Block(**block_config, drop_path=.4 * index / 39) for index in range(40))
        self.norm = torch.nn.LayerNorm(1408, eps=1e-5)

    def forward(self, image, masks_x=None):
        tokens = self.patch_embed(image) + self.pos_embed
        if masks_x is not None:
            tokens = apply_masks(tokens, masks_x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


ContextEncoder = PatchOnlyEncoder
TargetEncoder = PatchOnlyEncoder
transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])
