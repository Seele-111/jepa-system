"""Load the existing encoder without a transient FP32 CUDA checkpoint copy.

This changes storage residency, not the strict weights or inference dtype. The
caller must validate a trusted local checkpoint's digest before calling.
"""
from __future__ import annotations
import gc
from pathlib import Path


def load_encoder_cpu_checkpoint(factory, checkpoint, device):
    import torch
    encoder, unused_predictor = factory(pretrained=False)
    del unused_predictor
    # Match the legacy BF16 target dtype before copying the same FP32 weights.
    encoder = encoder.to(dtype=torch.bfloat16).eval()
    state = torch.load(Path(checkpoint), map_location='cpu',
                       weights_only=True, mmap=True)
    if not isinstance(state, dict) or set(state) != {'encoder'}:
        raise ValueError('unexpected existing encoder-only checkpoint schema')
    if not isinstance(state['encoder'], dict):
        raise ValueError('invalid encoder state dictionary')
    encoder.load_state_dict(state['encoder'], strict=True)
    del state
    gc.collect()
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval()
    return encoder
