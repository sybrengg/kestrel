"""One seeded, model-free reproduction of graph-dependent attention rounding.

Run with the installed original wheel, then with the patched Python wrapper:
  python minimal_repro.py --output original
  python minimal_repro.py --kernel-root /path/to/patched-parent --disable --output fixed
The parent must contain kestrel_kernels/. No weights or dataset are required.
"""
import argparse
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--kernel-root')
p.add_argument('--disable', action='store_true')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.kernel_root:
    sys.path.insert(0, a.kernel_root)
import torch
from kestrel_kernels import get_runtime
from kestrel_kernels.flash_attn.cute import interface

attention = get_runtime(torch.device('cuda')).attention.flash_attn_fwd
with torch.inference_mode():
    torch.manual_seed(914)  # 175 + 739; same fixture as the numerical sweep
    q = torch.randn(1, 1, 32, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(739, 1, 32, 64, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(739, 1, 32, 64, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    pages = torch.arange(739, device='cuda', dtype=torch.int32)[None]
    lengths = torch.tensor([739], device='cuda', dtype=torch.int32)
    kwargs = dict(causal=True, paged_kv_non_tma=True, pack_gqa=False,
                  num_splits=1, page_table=pages, seqused_k=lengths,
                  k_scale=.5, v_scale=.5)
    if a.disable:
        kwargs['allow_persistent_split'] = False
    def call():
        return attention(q, k, v, **kwargs)[0]
    eager = call().clone()
    eager_impl = str(interface._flash_attn_fwd._debug_last_impl)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    graph_impl = str(interface._flash_attn_fwd._debug_last_impl)
    graph.replay()
    torch.cuda.synchronize()
    changed = (captured != eager).nonzero().tolist()
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  disabled=a.disable, eager_impl=eager_impl, graph_impl=graph_impl,
                  different_elements=len(changed),
                  differences=[dict(index=i, eager=float(eager[tuple(i)]),
                                    captured=float(captured[tuple(i)])) for i in changed])
    if a.disable:
        assert torch.equal(captured, eager), result
    a.output.mkdir(parents=True)
    (a.output/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
