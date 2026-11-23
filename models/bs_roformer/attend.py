"""Attention for RoFormer inference."""
from inspect import signature

from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


INFERENCE_SDPA_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
_SDPA_HAS_PRIORITY = 'set_priority' in signature(sdpa_kernel).parameters


class Attend(nn.Module):
    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        # Keep the existing constructor interface; inference disables dropout.
        self.scale = scale
        self.flash = flash

    def forward(self, q, k, v):
        if self.flash:
            # Preserve the existing custom scaling of the query.
            if self.scale is not None:
                q = q * (self.scale / (q.shape[-1] ** -0.5))
            if q.is_cuda:
                if _SDPA_HAS_PRIORITY:
                    with sdpa_kernel(INFERENCE_SDPA_BACKENDS, set_priority=True):
                        return F.scaled_dot_product_attention(q, k, v)
                with sdpa_kernel(INFERENCE_SDPA_BACKENDS):
                    return F.scaled_dot_product_attention(q, k, v)
            return F.scaled_dot_product_attention(q, k, v)

        # Keep the non-SDPA path for custom inference configurations.
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)
