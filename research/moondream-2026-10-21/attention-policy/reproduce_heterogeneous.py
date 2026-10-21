"""Mixed lengths, distinct row KV, shuffled pages and unused-page poisoning."""
import argparse,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kernel-root',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path.insert(0,a.kernel_root);a.output.mkdir()
import torch
torch.cuda.set_per_process_memory_fraction(.05)
from kestrel_kernels import get_runtime
attention=get_runtime(torch.device('cuda')).attention.flash_attn_fwd
rows=[];references={}
with torch.inference_mode():
 for dtype in [torch.bfloat16,torch.float8_e4m3fn]:
  for batch in [1,2,16,32]:
   torch.manual_seed(914)
   lengths=[739]+[[1,17,65,129,511][i%5] for i in range(batch-1)]
   n=sum(lengths);heads,dim=32,64
   # Generate first-row inputs before any batch-dependent shapes.
   q0=torch.randn(1,1,heads,dim,device='cuda',dtype=torch.bfloat16)
   k0=torch.randn(739,1,heads,dim,device='cuda',dtype=torch.bfloat16)
   v0=torch.randn_like(k0)
   q=torch.cat([q0,torch.randn(batch-1,1,heads,dim,device='cuda',dtype=torch.bfloat16)])
   logical_k=torch.cat([k0,torch.randn(n-739,1,heads,dim,device='cuda',dtype=torch.bfloat16)]).to(dtype)
   logical_v=torch.cat([v0,torch.randn(n-739,1,heads,dim,device='cuda',dtype=torch.bfloat16)]).to(dtype)
   permutation=torch.randperm(n,device='cuda');k=torch.zeros(n+1,1,heads,dim,device='cuda',dtype=dtype);v=torch.zeros_like(k)
   k.view(torch.uint8)[permutation]=logical_k.view(torch.uint8);v.view(torch.uint8)[permutation]=logical_v.view(torch.uint8)
   pages=torch.full((batch,739),n,device='cuda',dtype=torch.int32);start=0
   oracles=[];scale=.5 if dtype==torch.float8_e4m3fn else 1.
   for i,length in enumerate(lengths):
    pages[i,:length]=permutation[start:start+length].int()
    kk=logical_k[start:start+length,0].double().permute(1,0,2)*scale
    vv=logical_v[start:start+length,0].double().permute(1,0,2)*scale
    scores=(q[i,0].double()[:,None,:]*kk).sum(-1)/dim**.5
    oracles.append((scores.softmax(-1)[:,:,None]*vv).sum(1));start+=length
   oracle=torch.stack(oracles)[:,None]
   seq=torch.tensor(lengths,device='cuda',dtype=torch.int32)
   for allow in [True,False]:
    kwargs=dict(page_table=pages,seqused_k=seq,causal=True,paged_kv_non_tma=True,pack_gqa=False,num_splits=1,allow_persistent_split=allow)
    if dtype==torch.float8_e4m3fn:kwargs.update(k_scale=scale,v_scale=scale)
    def call():return attention(q,k,v,**kwargs)[0]
    eager=call().clone();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):call()
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=call()
    graph.replay();torch.cuda.synchronize();out=captured.clone()
    # Unused padded page entries must not enter attention, even when nonzero.
    k[-1].fill_(64);v[-1].fill_(-64);graph.replay();torch.cuda.synchronize()
    poison_diff=int((captured!=out).sum());assert poison_diff==0
    k[-1].zero_();v[-1].zero_()
    torch.testing.assert_close(out.double(),oracle,atol=.008,rtol=.02)
    ref=references.setdefault((str(dtype),allow),out[0].clone())
    row=dict(dtype=str(dtype),batch=batch,allow_persistent_split=allow,lengths=lengths,graph_eager_different=int((out!=eager).sum()),row0_batch_different=int((out[0]!=ref).sum()),unused_page_different=poison_diff,oracle_max_abs=float((out.double()-oracle).abs().max()),oracle_rows=batch)
    rows.append(row);(a.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    if not allow:assert row['graph_eager_different']==0 and row['row0_batch_different']==0,row
(a.output/'complete.json').write_text(json.dumps(dict(cases=rows,total_oracle_rows=sum(r['oracle_rows'] for r in rows),errors=0),indent=2)+'\n')
