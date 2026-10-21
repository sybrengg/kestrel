"""CPU contracts for optional NVFP4 dependencies, shapes and graph ownership.

The arithmetic doubles below do not validate FP4 CUDA arithmetic; that is covered
by the explicit GPU probe. They exercise padding, bias and runtime dispatch.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.models.moondream import runtime as runtime_module
from kestrel.models.moondream import vision_nvfp4 as nv
from .test_vision_request_batch import make_runtime


def test_native_default_does_not_load_optional_dependencies(monkeypatch):
    monkeypatch.setattr(nv, "_load_flashinfer", lambda: pytest.fail("optional import"))
    assert RuntimeConfig(device="cpu", model_path="unused").moondream_vision_precision == "native"


@pytest.mark.parametrize("kwargs, message", [
    ({"moondream_vision_precision": "fp4"}, "must be"),
    ({"model": "moondream2"}, "supports moondream3.1"),
    ({"device": "cpu"}, "CUDA and bfloat16"),
    ({"dtype": torch.float16}, "CUDA and bfloat16"),
])
def test_invalid_option_fails_before_download(monkeypatch, kwargs, message):
    from kestrel import model_download
    monkeypatch.setattr(model_download, "ensure_model_weights", lambda *_: pytest.fail("download"))
    options = dict(model="moondream3.1-9B-A2B", device="cuda", moondream_vision_precision="nvfp4")
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RuntimeConfig(**options)


def test_unsupported_hardware_does_not_import_flashinfer(monkeypatch):
    monkeypatch.setattr(nv, "get_device_capability", lambda *_: (9, 0))
    monkeypatch.setattr(nv, "_load_flashinfer", lambda: pytest.fail("optional import"))
    with pytest.raises(ValueError, match="SM120"):
        nv.validate_nvfp4("moondream3.1-9B-A2B", torch.device("cuda"), torch.bfloat16)


def test_missing_extra_has_actionable_error(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(nv, "import_module", missing)
    with pytest.raises(RuntimeError, match=r'kestrel\[nvfp4\]'):
        nv._load_flashinfer()


@pytest.mark.parametrize("k,n,bias", [(64, 64, False), (65, 67, True), (4304, 16, True)])
def test_linear_pads_reduction_and_output_axes_then_restores_shape(k, n, bias):
    torch.manual_seed(7)
    layer = torch.nn.Linear(k, n, bias=bias, dtype=torch.float32)
    shapes = []
    def quantize(value, scale, **kwargs):
        assert kwargs == dict(backend="cute-dsl", nvfp4_4over6=None)
        shapes.append(tuple(value.shape))
        # Reversible scale-only double: preserve the actual matrix orientation.
        return value * scale, torch.ones(1, 1)
    def matmul(a, b, sa, sb, alpha, **kwargs):
        assert kwargs == dict(out_dtype=torch.bfloat16, backend="cudnn")
        return (a @ b) * alpha
    linear = nv.NVFP4Linear(layer, quantize, matmul, nv._activation_scale)
    x = torch.randn(2, 3, k)
    result = linear(x)
    assert result.shape == (2, 3, n)
    assert shapes == [(((n + 63) // 64) * 64, ((k + 63) // 64) * 64),
                      (6, ((k + 63) // 64) * 64)]
    torch.testing.assert_close(result, layer(x), atol=2e-6, rtol=2e-5)
    assert torch.isfinite(nv._activation_scale(torch.zeros(2, k)))


def test_nvfp4_graph_does_not_pad_or_replay_stale_crops():
    runtime = make_runtime()
    runtime._vision_output = torch.zeros(4, 1)
    calls = []
    def encoder(crops):
        calls.append(crops.clone())
        return crops * 2
    runtime._nvfp4_vision = encoder
    runtime._vision_graphs = {4: SimpleNamespace(replay=lambda: pytest.fail("padded graph"))}
    result = runtime._encode_native_vision_crops(torch.tensor([[1.], [2.], [3.]]))
    assert result.flatten().tolist() == [2., 4., 6.]
    assert calls[0].shape == (3, 1)
    runtime._vision_graphs[3] = SimpleNamespace(
        replay=lambda: runtime._vision_output[:3].copy_(runtime._vision_input[:3] * 3))
    result = runtime._encode_native_vision_crops(torch.tensor([[4.], [5.], [6.]]))
    assert result.flatten().tolist() == [12., 15., 18.]
    assert len(calls) == 1


def test_eager_default_uses_original_tower(monkeypatch):
    runtime = make_runtime()
    runtime._vision_graphs = {}
    runtime._vision_output = torch.empty(4, 1)
    runtime.model = SimpleNamespace(vision=object())
    runtime.config = SimpleNamespace(vision=object())
    monkeypatch.setattr(runtime_module, "vision_encoder", lambda crops, *_: crops * 7)
    assert runtime._encode_native_vision_crops(torch.ones(2, 1)).tolist() == [[7.], [7.]]


@pytest.mark.parametrize("graphs", [False, True])
def test_rebuild_requantizes_changed_weights_before_recapturing(monkeypatch, graphs):
    runtime = make_runtime()
    runtime._nvfp4_vision = object()
    runtime._vision_graphs = {1: object()}
    runtime._use_cuda_graphs = graphs
    runtime.graph_capture_lock = nullcontext()
    runtime.model = SimpleNamespace(vision=object())
    runtime.config = SimpleNamespace(vision=object())
    calls = []
    def rebuild(*_):
        assert not runtime._vision_graphs
        calls.append("requantize")
        return "new weights"
    monkeypatch.setattr(nv, "NVFP4VisionEncoder", rebuild)
    runtime._decode_graphs = SimpleNamespace(clear=lambda: calls.append("clear decode"))
    runtime._ensure_cuda_graphs_ready = lambda: calls.append("decode")
    runtime._capture_vision_graphs = lambda: calls.append(runtime._nvfp4_vision)
    runtime.rebuild_cuda_graphs()
    assert calls == (["requantize", "clear decode", "decode", "new weights"] if graphs else ["requantize"])
