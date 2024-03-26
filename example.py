import timm
import torch.nn as nn

from lora import QkvWithLoRA
from functools import partial


model_name = 'vit_tiny_patch16_224.augreg_in21k'
num_classes = 10

lora_rank = 8
lora_alpha = 1.0


# Create base ViT model
model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
assert isinstance(model, timm.models.VisionTransformer)


# Add LoRA adapters to self-attention blocks (query, value)
assign_lora = partial(QkvWithLoRA, rank=lora_rank, alpha=lora_alpha)
for block in model.blocks:
    block.attn.qkv = assign_lora(block.attn.qkv)


# Freeze all params
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LoRA layers
for block in model.blocks:
    for param in block.attn.qkv.lora_q.parameters():
        param.requires_grad = True
    for param in block.attn.qkv.lora_v.parameters():
        param.requires_grad = True

# Unfreeze classifier layer
for param in model.head.parameters():
    param.requires_grad = True
