"""Explicit opt-out retains the ordinary reduction during graph capture."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA attention regression')
@pytest.mark.parametrize('kv_dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_persistent_split_optout_matches_eager_across_graph_batch_sizes(kv_dtype):
    from kestrel_kernels import get_runtime
    attention = get_runtime(torch.device('cuda')).attention.flash_attn_fwd
    gen = torch.Generator(device='cuda').manual_seed(175)
    heads, dim, length = 32, 64, 739
    base = torch.randn(1, 1, heads, dim, device='cuda', dtype=torch.bfloat16, generator=gen)
    k = torch.randn(length, 1, heads, dim, device='cuda', dtype=torch.bfloat16, generator=gen).to(kv_dtype)
    v = torch.randn(length, 1, heads, dim, device='cuda', dtype=torch.bfloat16, generator=gen).to(kv_dtype)
    scale = .5 if kv_dtype == torch.float8_e4m3fn else 1.
    kwargs = dict(causal=True, paged_kv_non_tma=True, pack_gqa=False,
                  num_splits=1, allow_persistent_split=False)
    if kv_dtype == torch.float8_e4m3fn: kwargs.update(k_scale=scale, v_scale=scale)
    reference = None
    for batch in [1, 2, 4, 8, 16]:
        q = base.expand(batch, -1, -1, -1).contiguous()
        page_table = torch.arange(length, device='cuda', dtype=torch.int32).expand(batch, -1).contiguous()
        lengths = torch.full((batch,), length, device='cuda', dtype=torch.int32)
        def call():
            return attention(q, k, v, page_table=page_table, seqused_k=lengths, **kwargs)[0]
        with torch.inference_mode():
            eager = call().clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            graph.replay()
            torch.cuda.synchronize()
        assert torch.equal(captured, eager), ('graph vs eager', batch)
        if reference is None: reference = eager[0].clone()
        assert torch.equal(captured, reference.expand_as(captured)), ('batch', batch)
    q64 = base[0, 0].double()
    k64 = k[:, 0].double().permute(1, 0, 2) * scale
    v64 = v[:, 0].double().permute(1, 0, 2) * scale
    scores = (q64[:, None, :] * k64).sum(-1) / dim**.5
    oracle = (scores.softmax(-1)[:, :, None] * v64).sum(1)
    torch.testing.assert_close(reference[0].double(), oracle, atol=.002, rtol=.01)
