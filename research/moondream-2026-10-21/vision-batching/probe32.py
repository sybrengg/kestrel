"""Real image ownership, geometry, mixed chat/cache and allocator reuse probe."""
import asyncio,base64,io,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def run(model,blobs,source,out):
 runtime=model._engine.runtime
 geometries=[(1280,720),(320,320),(128,1024),(1024,128),(73,51),(640,480),(480,640),(960,540)]
 images=[];manifest=[]
 for i in range(32):
  row=source[(i*7)%32];im=Image.open(io.BytesIO(blobs[row['id']])).convert('RGB').resize(geometries[i%8]);buffer=io.BytesIO();im.save(buffer,format='PNG');images.append(buffer.getvalue());manifest.append(dict(id=row['id'],size=im.size))
 with torch.inference_mode(),torch.cuda.stream(runtime._compute_stream):
  reference=[runtime.encode_image(im).clone() for im in images]
  results=[];memory=[];timings=[]
  original_encoder=runtime._encode_native_vision_crops;launches=[]
  def tracked_encoder(crops):
   launches.append(len(crops));return original_encoder(crops)
  runtime._encode_native_vision_crops=tracked_encoder
  for repeat in range(2):
   for count in ([x for pair in zip(range(1,17),range(32,16,-1)) for x in pair] if not repeat else list(range(32,0,-1))):
    order=[(j*7+repeat*3)%32 for j in range(count)]
    torch.cuda.synchronize();launches.clear();start=time.perf_counter()
    values=runtime.encode_images([images[i] for i in order],[None]*count) if hasattr(runtime,'encode_images') else [runtime.encode_image(images[i]).clone() for i in order]
    torch.cuda.synchronize();wall=time.perf_counter()-start;batch_launches=list(launches)
    if hasattr(runtime,'encode_images'):
     launches.clear();start=time.perf_counter()
     individual=[runtime.encode_image(images[i]).clone() for i in order]
     torch.cuda.synchronize();serial_wall=time.perf_counter()-start;del individual
    else:serial_wall=wall
    timings.append(dict(repeat=repeat,count=count,wall_s=wall,individual_wall_s=serial_wall,crop_launches=batch_launches))
    for i,value in zip(order,values):
     error=(value.float()-reference[i].float());relative=float(error.norm()/reference[i].float().norm())
     results.append(dict(repeat=repeat,count=count,image=i,different=int((value!=reference[i]).sum()),relative_l2=relative,max_abs=float(error.abs().max())))
     assert relative<.02,results[-1]
    torch.cuda.synchronize();memory.append(torch.cuda.memory_allocated());del values
  runtime._encode_native_vision_crops=original_encoder
  del reference
 (out/'encoder-mixed.json').write_text(json.dumps(dict(manifest=manifest,comparisons=results,allocated_bytes=memory,timings=timings),indent=2)+'\n')
 # Observe real evictions without changing cache behavior; all work is in this
 # fresh engine and no shared service cache is touched.
 evictions=[];cache=runtime.prefix_cache;original_evict=cache.evict
 def evict(needed):
  result=original_evict(needed);evictions.append(dict(needed=needed,result=result));return result
 cache.evict=evict
 def content(index):return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(images[index%32]).decode()))
 async def one(index):
  if index%4==0:parts='Reply with the word hello.'
  else:parts=[*[content(index+j) for j in range(1+index%3)],dict(type='text',text='Briefly describe the images in one sentence.')]
  value=await model._model.chat(messages=[dict(role='user',content=parts)],settings=dict(temperature=0,max_tokens=24),reasoning=False)
  assert value.output['message']['content'].strip(), dict(index=index,output=value.output)
  return dict(index=index,output=value.output,finish_reason=value.finish_reason)
 async def group(order,concurrency):
  sem=asyncio.Semaphore(concurrency)
  async def bounded(i):
   async with sem:return await one(i)
  return await asyncio.gather(*(bounded(i) for i in order))
 histories=[]
 try:
  for name,order,c in [('serial',list(range(32)),1),('mixed',list(reversed(range(32))),32),('cached',list(range(32)),32),('serial-after',list(range(32)),1)]:
   start=time.perf_counter();rows=model._run(group(order,c));histories.append(dict(name=name,rows=rows,wall_s=time.perf_counter()-start))
   (out/'chat-mixed.json').write_text(json.dumps(dict(histories=histories,evictions=evictions),indent=2)+'\n')
  # Chat image markers deliberately bypass prefix caching upstream. Exercise
  # the real single-image query cache separately, with more than 8192 pages.
  cache_rows=[];runtime._encode_native_vision_crops=tracked_encoder
  async def cached_query(index):
   return await model._model.query(image=images[index],question='What is the main visible subject? Reply briefly.',reasoning=False,settings=dict(temperature=0,max_tokens=24))
  for index in [*range(32),0,0]:
   launches.clear();value=model._run(cached_query(index))
   assert value.output['answer'].strip(),value.output
   cache_rows.append(dict(index=index,output=value.output,encoder_launches=list(launches)))
   (out/'cache-pressure.json').write_text(json.dumps(dict(rows=cache_rows,evictions=evictions),indent=2)+'\n')
  assert any(x['result']>0 for x in evictions),'cache pressure did not exercise eviction'
  assert cache_rows[-2]['encoder_launches'],'old image must require re-encoding after eviction'
  assert not cache_rows[-1]['encoder_launches'],'immediate repeat must reuse cached image'
 finally:
  cache.evict=original_evict;runtime._encode_native_vision_crops=original_encoder
 (out/'complete.json').write_text(json.dumps(dict(encoder_comparisons=len(results),max_relative_l2=max(x['relative_l2'] for x in results),chat_requests=sum(len(x['rows']) for x in histories),cache_pressure_requests=len(cache_rows),evictions=evictions,errors=0),indent=2)+'\n')
