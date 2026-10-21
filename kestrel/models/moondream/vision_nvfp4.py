"""Opt-in NVFP4 linears for the Moondream 3.1 vision tower on SM120.

Only QKV, attention output and MLP projections are quantized. The original
weights remain owned by the model; this is a throughput option, not a model
compression format. Imports and compilation are deferred until selected.
"""

from importlib import import_module

import torch
import torch.nn.functional as F

from kestrel.device import get_device_capability


def _load_flashinfer():
    try:
        flashinfer = import_module("flashinfer")
        import_module("cutlass")
    except ImportError as exc:
        raise RuntimeError(
            'NVFP4 vision requires the optional dependencies: pip install "kestrel[nvfp4]"'
        ) from exc
    return flashinfer.nvfp4_quantize, flashinfer.mm_fp4


def validate_nvfp4(model: str, device: torch.device, dtype: torch.dtype) -> None:
    """Reject unsupported requests before downloading/loading weights."""
    if model != "moondream3.1-9B-A2B":
        raise ValueError("NVFP4 vision currently supports moondream3.1-9B-A2B")
    if device.type != "cuda" or dtype != torch.bfloat16:
        raise ValueError("NVFP4 vision requires CUDA and bfloat16")
    if tuple(get_device_capability(device)) != (12, 0):
        raise ValueError("NVFP4 vision currently supports SM120 (RTX 5090)")
    _load_flashinfer()


def _activation_scale(x: torch.Tensor) -> torch.Tensor:
    # 448 (FP8 block scale) * 6 (FP4 value). Keep reduction in BF16 as
    # in the measured prototype, then promote its scalar to FP32.
    return 2688 / x.abs().amax().float().clamp_min(1e-12)


class NVFP4Linear:
    def __init__(self, layer, quantize, matmul, activation_scale):
        self.n, self.k = layer.weight.shape
        self.bias = layer.bias
        self.kpad = (-self.k) % 64
        weight = F.pad(layer.weight, (0, self.kpad, 0, (-self.n) % 64)).contiguous()
        self.weight_scale = 2688 / weight.float().abs().amax().clamp_min(1e-12)
        self.weight, self.block_scales = quantize(
            weight, self.weight_scale, backend="cute-dsl", nvfp4_4over6=None,
        )
        self.quantize = quantize
        self.matmul = matmul
        self.activation_scale = activation_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.k)
        a = F.pad(x, (0, self.kpad)).contiguous() if self.kpad else x.contiguous()
        scale = self.activation_scale(a)
        values, blocks = self.quantize(
            a, scale, backend="cute-dsl", nvfp4_4over6=None,
        )
        y = self.matmul(
            values, self.weight.t(), blocks, self.block_scales.t(),
            1 / (scale * self.weight_scale), out_dtype=torch.bfloat16,
            backend="cudnn",
        )[:, :self.n]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.n)


class NVFP4VisionEncoder:
    @torch.no_grad()
    def __init__(self, vision, config, device: torch.device):
        from kestrel_kernels import get_runtime

        quantize, matmul = _load_flashinfer()
        scale = torch.compile(_activation_scale, fullgraph=True, dynamic=True)
        self.vision = vision
        self.config = config
        self.backend = get_runtime(device)
        self.blocks = [
            {name: NVFP4Linear(layer, quantize, matmul, scale)
             for name, layer in (("qkv", block.attn["qkv"]),
                                 ("proj", block.attn["proj"]),
                                 ("fc1", block.mlp["fc1"]),
                                 ("fc2", block.mlp["fc2"]))}
            for block in vision.blocks
        ]

    def __call__(self, crops: torch.Tensor) -> torch.Tensor:
        from .vision import create_patches

        model, config = self.vision, self.config
        x = model.patch_emb(create_patches(crops, config.enc_patch_size)) + model.pos_emb
        norm = torch.empty_like(x)

        def layernorm(value, module):
            self.backend.dense.layernorm_bias_into(
                norm, value, module.weight, module.bias, float(module.eps),
            )
            return norm

        for block, quant in zip(model.blocks, self.blocks):
            q, k, v = quant["qkv"](layernorm(x, block.ln1)).chunk(3, dim=-1)
            shape = (*x.shape[:2], config.enc_n_heads, config.enc_dim // config.enc_n_heads)
            attention, _ = self.backend.attention.flash_attn_fwd(
                q.view(shape), k.view(shape), v.view(shape), causal=False,
            )
            x = x + quant["proj"](attention.reshape_as(x))
            x = x + quant["fc2"](F.gelu(quant["fc1"](layernorm(x, block.ln2)), approximate="tanh"))
        return layernorm(x, model.post_ln)
