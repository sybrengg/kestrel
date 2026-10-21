"""Request ownership and shared-graph buffer regression tests (CPU)."""
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import runtime as module


def make_runtime():
    runtime = module.MoondreamRuntime.__new__(module.MoondreamRuntime)
    runtime._vision_backend = None
    runtime._batch_vision = True
    runtime.device = torch.device('cpu')
    runtime.dtype = torch.float32
    runtime._vision_input = torch.empty(4, 1)
    return runtime


def test_encoder_batches_across_images_and_copies_replayed_storage(monkeypatch):
    runtime = make_runtime()
    # The second image crosses the graph-launch boundary. The encoder deliberately
    # returns a shared output buffer that the next invocation overwrites.
    overlaps = [dict(values=[10., 11.], tiling=(1, 1)),
                dict(values=[20., 21., 22.], tiling=(1, 2)),
                dict(values=[30., 31.], tiling=(1, 1))]
    shared = torch.empty(4, 1)
    launches = []
    def prepare(item, *_):
        if item is overlaps[-1]:
            # A full group must start before preparing later images: otherwise
            # CPU decoding/cropping serializes ahead of all GPU tower work.
            assert launches, "full crop groups must not wait for every image"
        return torch.tensor(item['values']).view(-1, 1), item['tiling']
    monkeypatch.setattr(module, 'prepare_crops_from_overlap', prepare)
    def encode(crops):
        launches.append(crops.flatten().tolist())
        shared[:len(crops)].copy_(crops)
        return shared[:len(crops)]
    runtime._encode_native_vision_crops = encode
    runtime._project_vision_crops = lambda features, tiling: (features.clone(), tiling)
    outputs = runtime.encode_images([None]*3, overlaps)
    assert launches == [[10., 11., 20., 21.], [22., 30., 31.]]
    for result, source in zip(outputs, overlaps):
        assert result[0].flatten().tolist() == source['values']
        assert result[1] == source['tiling']


def test_prefill_gather_excludes_cache_hits_and_keeps_multi_image_ownership():
    runtime = make_runtime()
    prepared = [SimpleNamespace(cache_result=SimpleNamespace(can_reuse=v))
                for v in [False, True, False, False]]
    a, b, c, cached = object(), object(), object(), object()
    captured = {}
    def encode(images, overlaps):
        captured.update(images=images, overlaps=overlaps)
        return ['a-features', 'b-features', 'c-features']
    runtime.encode_images = encode
    result = runtime._encode_prefill_images(
        prepared, [a, cached, [b, c], None], ['a-crops', None, None, None])
    assert captured == dict(images=[a, b, c], overlaps=['a-crops', None, None])
    assert result == [['a-features'], None, ['b-features', 'c-features'], None]


def test_empty_and_invalid_batches():
    runtime = make_runtime()
    assert runtime.encode_images([], []) == []
    with pytest.raises(ValueError, match='same length'):
        runtime.encode_images([None], [])


def test_default_does_not_gather_or_change_legacy_image_path():
    from kestrel.config import RuntimeConfig
    assert RuntimeConfig(device="cpu", model_path="unused").moondream_vision_batch_crops is None
    runtime = make_runtime()
    runtime._batch_vision = False
    runtime.encode_images = lambda *_: pytest.fail("default must retain legacy path")
    assert runtime._encode_prefill_images([object()], [object()], [None]) == [None]


def test_backend_without_crop_batch_api_retains_owned_image_outputs():
    runtime = make_runtime()
    runtime._vision_backend = object()
    buffer = torch.empty(2)
    def encode(image, *, overlap):
        buffer.fill_(image)
        return buffer
    runtime.encode_image = encode
    result = runtime.encode_images([10, 20], [None, None])
    assert result[0].tolist() == [10, 10]
    assert result[1].tolist() == [20, 20]


def test_graph_bucket_padding_does_not_reuse_stale_crops():
    runtime = make_runtime()
    runtime._vision_input = torch.full((4, 1), 99.)
    runtime._vision_output = torch.empty(4, 1)
    seen = []
    def replay():
        seen.append(runtime._vision_input.clone())
        runtime._vision_output.copy_(runtime._vision_input * 2)
    runtime._vision_graphs = {4: SimpleNamespace(replay=replay)}
    result = runtime._encode_native_vision_crops(torch.tensor([[1.], [2.], [3.]]))
    assert result.flatten().tolist() == [2., 4., 6.]
    assert seen[0].flatten().tolist() == [1., 2., 3., 0.]
    with pytest.raises(ValueError, match='capacity'):
        runtime._encode_native_vision_crops(torch.ones(5, 1))


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_capacity_rejected_before_model_download(value):
    from kestrel.config import RuntimeConfig
    with pytest.raises(ValueError, match='moondream_vision_batch_crops'):
        RuntimeConfig(device='cpu', moondream_vision_batch_crops=value)


def test_common_gathered_crop_count_uses_exact_graph_without_padded_work(monkeypatch):
    from contextlib import contextmanager, nullcontext
    runtime = make_runtime()
    runtime._use_cuda_graphs = True
    runtime._vision_batch_capacity = 32
    runtime.config = SimpleNamespace(vision=SimpleNamespace(max_crops=12))
    runtime.model = SimpleNamespace(vision=object())
    runtime.graph_capture_lock = nullcontext()
    runtime._vision_input = torch.zeros(32, 1)
    runtime._vision_output = torch.zeros(32, 1)
    runtime._vision_graphs = {}
    active = []
    captured = {}
    class Graph:
        def pool(self): return None
        def replay(self): active.append(captured[id(self)])
    @contextmanager
    def capture(graph, **_):
        active.append(graph)
        try:
            yield
        finally:
            active.pop()
    def encoder(crops, *_):
        if active and isinstance(active[-1], Graph):
            captured[id(active[-1])] = len(crops)
        return crops
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda **_: None)
    monkeypatch.setattr(module, 'vision_encoder', encoder)
    runtime._capture_vision_graphs()
    active.clear()
    # Three ordinary nine-crop images should not execute five padded crops.
    runtime._encode_native_vision_crops(torch.ones(27, 1))
    assert active == [27]
    assert max(runtime._vision_graphs) == 32
