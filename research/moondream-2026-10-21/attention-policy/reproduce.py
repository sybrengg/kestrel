"""Attention dispatch, graph/eager equality and independent FP64 error audit."""
import argparse,json,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kernel-root');p.add_argument('--patched',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
if a.kernel_root:sys.path.insert(0,a.kernel_root)
import torch
from kestrel_kernels import get_runtime
from kestrel_kernels.flash_attn.cute import interface
attention=get_runtime(torch.device('cuda')).attention.flash_attn_fwd
torch.cuda.set_per_process_memory_fraction(.3)
rows=[]
with torch.inference_mode():
 for dtype in [torch.bfloat16,torch.float8_e4m3fn]:
  for length in [1,17,65,739,2049,4097]:
   torch.manual_seed(175+length)
   heads,dim=32,64
   q0=torch.randn(1,1,heads,dim,device='cuda',dtype=torch.bfloat16)
   k=torch.randn(length,1,heads,dim,device='cuda',dtype=torch.bfloat16).to(dtype)
   v=torch.randn(length,1,heads,dim,device='cuda',dtype=torch.bfloat16).to(dtype)
   scale=.5 if dtype==torch.float8_e4m3fn else 1.
   oracle=((q0[0,0].double()[:,None,:]*k[:,0].double().permute(1,0,2)*scale).sum(-1)/dim**.5).softmax(-1)
   oracle=(oracle[:,:,None]*v[:,0].double().permute(1,0,2)*scale).sum(1)
   references={}
   for batch in [1,2,16]:
    q=torch.randn(batch,1,heads,dim,device='cuda',dtype=torch.bfloat16);q[0]=q0[0]
    pages=torch.arange(length,device='cuda',dtype=torch.int32).expand(batch,-1).contiguous()
    lengths=torch.full((batch,),length,device='cuda',dtype=torch.int32)
    for policy in (['default','allow','disable'] if a.patched else ['default']):
     kwargs=dict(causal=True,paged_kv_non_tma=True,pack_gqa=False,num_splits=1,page_table=pages,seqused_k=lengths)
     if policy!='default':kwargs['allow_persistent_split']=policy=='allow'
     if dtype==torch.float8_e4m3fn:kwargs.update(k_scale=scale,v_scale=scale)
     def call():return attention(q,k,v,**kwargs)[0]
     eager=call().clone();eager_impl=str(getattr(interface._flash_attn_fwd,'_debug_last_impl',None))
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):call()
     stream.synchronize();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph,stream=stream):captured=call()
     graph_impl=str(getattr(interface._flash_attn_fwd,'_debug_last_impl',None));graph.replay();torch.cuda.synchronize()
     out=captured.clone();error=out[0,0].double()-oracle;reference=references.setdefault(policy,out[0].clone())
     row=dict(dtype=str(dtype),length=length,batch=batch,policy=policy,eager_impl=eager_impl,graph_impl=graph_impl,graph_eager_different=int((out!=eager).sum()),row0_batch_different=int((out[0]!=reference).sum()),oracle_max_abs=float(error.abs().max()),oracle_rmse=float(error.square().mean().sqrt()),row0_sha256=hashlib.sha256(out[0].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),finite=bool(out.isfinite().all()))
     rows.append(row);(a.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
     assert row['finite']
     torch.testing.assert_close(out[0,0].double(),oracle,atol=.008,rtol=.02)
     if policy=='disable':assert not row['graph_eager_different'] and not row['row0_batch_different'],row
(a.output/'complete.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(cases=len(rows),graph_eager_nonexact=sum(x['graph_eager_different']>0 for x in rows),max_oracle_error=max(x['oracle_max_abs'] for x in rows))))
