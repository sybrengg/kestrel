"""Check the real optional runtime, graph/eager features and mixed/cache requests.

Install the review branch plus its nvfp4 extra. Use 32 distinct image files;
run native and nvfp4, each with --graphs 0 and 1, into separate output directories.
Optional --prototype points to the frozen quant_encoder_clean.py for a same-input
bitwise comparison. Normal checkpoint loading is used without monkeypatching it.
"""
import argparse
import asyncio
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import torch
from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.moondream import vision
from kestrel.utils.image import decode_to_srgb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-path', required=True)
parser.add_argument('--tokenizer-path', required=True)
parser.add_argument('--images', nargs='+', type=Path, required=True)
parser.add_argument('--precision', choices=['native', 'nvfp4'], required=True)
parser.add_argument('--graphs', type=int, choices=[0, 1], default=1)
parser.add_argument('--prototype', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if len(args.images) < 32:
    parser.error('Provide at least 32 distinct images to exercise cache eviction')
blobs = [p.read_bytes() for p in args.images[:32]]
hashes = [hashlib.sha256(b).hexdigest() for b in blobs]
if len(set(hashes)) != 32:
    parser.error('The first 32 images must be distinct')
args.output.mkdir(parents=True, exist_ok=False)
def save(name, value):
    (args.output / name).write_text(json.dumps(value, indent=2) + '\n')

async def main():
    cfg = RuntimeConfig(
        model='moondream3.1-9B-A2B', model_path=args.model_path,
        tokenizer_path=args.tokenizer_path, max_batch_size=32,
        kv_cache_pages=8192, enable_prefix_cache=True,
        enable_cuda_graphs=bool(args.graphs), moondream_vision_batch_crops=32,
        moondream_vision_precision=args.precision,
    )
    save('inputs.json', dict(hashes=hashes, precision=args.precision, graphs=args.graphs,
         device=torch.cuda.get_device_name(), torch=torch.__version__))
    started = time.perf_counter()
    engine = await InferenceEngine.create(cfg)
    model = engine.model(cfg.model)
    runtime = engine.runtime
    save('load.json', dict(elapsed_s=time.perf_counter()-started))
    try:
        prototype = None
        if args.prototype and args.precision == 'nvfp4':
            spec = importlib.util.spec_from_file_location('frozen_quant_encoder', args.prototype)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with torch.inference_mode(), torch.cuda.stream(runtime._compute_stream):
            if args.prototype and args.precision == 'nvfp4':
                prototype = module.Tower(runtime, 'nvfp4-fast')
            overlaps = [vision.compute_overlap_crops(decode_to_srgb(b), runtime.config.vision)
                        for b in blobs[:8]]
            crops = torch.cat([vision.prepare_crops_from_overlap(o, runtime.device, runtime.dtype)[0]
                               for o in overlaps])
            assert len(crops) >= 32
            feature_checks = []
            for count in range(1, 33):
                value = runtime._encode_native_vision_crops(crops[:count]).clone()
                replay = runtime._encode_native_vision_crops(crops[:count]).clone()
                eager = runtime._run_vision_tower(crops[:count]).clone()
                check = dict(crops=count, replay_exact=torch.equal(value, replay),
                             eager_exact=torch.equal(value, eager), finite=bool(value.isfinite().all()))
                if prototype:
                    ref = prototype.forward(crops[:count]).clone()
                    check['prototype_exact'] = torch.equal(value, ref)
                    check['prototype_max_abs'] = float((value-ref).abs().max())
                feature_checks.append(check)
                save('features.json', feature_checks)
                assert check['finite'] and check['replay_exact'], check
                if args.precision == 'nvfp4':
                    assert check['eager_exact'], check
                if prototype:
                    assert check['prototype_exact'], check
            torch.cuda.synchronize()
            if args.precision == 'nvfp4':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    runtime._encode_native_vision_crops(crops[:9])
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(args.output/'runtime-fp4-trace.json'))
            del prototype, crops
        # Observe encoding/eviction only; the execution method is unchanged.
        original_encoder = runtime._encode_native_vision_crops
        launches = []
        def tracked(crops):
            launches.append(len(crops))
            return original_encoder(crops)
        runtime._encode_native_vision_crops = tracked
        cache = runtime.prefix_cache
        evictions = []
        original_evict = cache.evict
        def evict(count):
            result = original_evict(count)
            evictions.append(dict(requested=count, evicted=result))
            return result
        cache.evict = evict
        cache_rows = []
        try:
            for index in [*range(32), 0, 0]:
                launches.clear()
                result = await model.query(image=blobs[index], question='Briefly describe the main subject.',
                                           reasoning=False, settings=dict(temperature=0, max_tokens=24))
                assert result.output['answer'].strip(), result.output
                cache_rows.append(dict(index=index, output=result.output, cached_tokens=result.metrics.cached_tokens,
                                       encoder_launches=list(launches)))
                save('cache.json', dict(rows=cache_rows, evictions=evictions))
            assert any(x['evicted'] > 0 for x in evictions), evictions
            assert cache_rows[-2]['encoder_launches'], cache_rows[-2]
            assert not cache_rows[-1]['encoder_launches'], cache_rows[-1]
            def image_part(i):
                return dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                            base64.b64encode(blobs[i % 32]).decode()))
            async def one(index):
                content = ('Reply with the word hello.' if index % 4 == 0 else
                           [*[image_part(index+j) for j in range(1+index % 3)],
                            dict(type='text', text='Briefly describe the images.')])
                result = await model.chat(messages=[dict(role='user', content=content)], reasoning=False,
                                          settings=dict(temperature=0, max_tokens=24))
                assert result.output['message']['content'].strip(), result.output
                return dict(index=index, output=result.output, finish_reason=result.finish_reason)
            for concurrency in [1, 16, 32]:
                semaphore = asyncio.Semaphore(concurrency)
                async def bounded(i):
                    async with semaphore:
                        return await one(i)
                rows = await asyncio.gather(*(bounded(i) for i in range(32)))
                save(f'mixed-{concurrency}.json', rows)
        finally:
            runtime._encode_native_vision_crops = original_encoder
            cache.evict = original_evict
        save('complete.json', dict(feature_counts=32, mixed_requests=96, cache_requests=34,
             errors=0, precision=args.precision, graphs=args.graphs,
             scope='Runtime integration and request ownership; answers retained, not semantic accuracy scoring.'))
    finally:
        await engine.shutdown()

asyncio.run(main())
