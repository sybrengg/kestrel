"""Run normal model loading, mixed images, cache pressure and batch counts 1-16.

Use local model/tokenizer files and your own representative images. This script
never modifies installed kernels or model weights. Outputs must not already exist.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--tokenizer-path', required=True)
parser.add_argument('--images', nargs='+', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--vision-crops', type=int, default=32,
                    help='0 retains the legacy path; 32 enables cross-request encoding')
parser.add_argument('--memory-fraction', type=float, default=.61)
args = parser.parse_args()
if len(args.images) < 24:
    parser.error("Provide at least 24 distinct representative images to force cache eviction")
args.output.mkdir(parents=True, exist_ok=False)

import torch
import moondream
from kestrel.config import RuntimeConfig
from probe import run

torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
kwargs = {}
if 'moondream_vision_batch_crops' in RuntimeConfig.__dataclass_fields__:
    kwargs['moondream_vision_batch_crops'] = args.vision_crops or None
model = moondream.photon(
    'moondream3.1-9B-A2B', model_path=args.model_path,
    tokenizer_path=args.tokenizer_path, max_batch_size=16,
    kv_cache_pages=8192, enable_prefix_cache=True, enable_cuda_graphs=True,
    **kwargs,
)
try:
    sources = [dict(id=str(i)) for i in range(24)]
    blobs = {str(i): args.images[i % len(args.images)].read_bytes() for i in range(24)}
    run(model, blobs, sources, args.output)
finally:
    model.close()
