"""Fixed-budget paired execution; truth is scored only after method traces."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from .methods import Frame,METHODS,WALD,run_method
from .storage import utc,canonical,file_sha,load,new_json,write_chunk,read_chunk,owned,WorkerLock,execution_permission

def seed(text):return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8],'big')
def rng(text):return np.random.Generator(np.random.PCG64(seed(text)))

def streams(config,mechanism,budget,rep,N):
    prefix=f"{config['audit_seed_prefix']}|{mechanism}|{budget}|{rep}|"
    index=int(rng(prefix+'population_index').integers(config['populations_per_mechanism']))
    order=rng(prefix+'srs_order').permutation(N)
    folds=[rng(prefix+f'fold_{k}_order').permutation(N//2) for k in (0,1)]
    uniforms=rng(prefix+'active_uniforms').random(budget)
    return index,order,folds,uniforms

def paired(frame,y,budget,order,folds,uniforms,key):
    outputs=[]
    for method in METHODS:
        out=run_method(frame,budget,method,lambda i:int(y[i]),order,folds,uniforms)
        outputs.append(out)
    # Scorer is outside every estimator invocation.
    B=int(sum(y));capacity=frame.N*frame.K;points=[];intervals=[];traces={}
    for o in outputs:
        point={k:o[k] for k in ('method','budget','estimate','labels','elapsed_seconds')}
        point.update(truth=B,squared_error=float((o['estimate']-B)**2),signed_error=float(o['estimate']-B))
        if point['squared_error']/(capacity**2)>(529 if o['method'].startswith(('ppat_','active_')) else 4)+1e-10:raise ArithmeticError('score support')
        points.append(point);traces[o['method']]=o['trace']
        for name,ci in o['intervals'].items():
            intervals.append(dict(ci,interval_id=name,method=o['method'],covers=bool(ci['status']=='OK' and ci['lower']<=B<=ci['upper'])))
    return dict(key,frame_id=frame.frame_id,capacity=capacity,points=points,intervals=intervals,traces=traces)

def construct_population(mechanism,index,config,directory,context):
    """Called ONLY by the approval-gated production path; no fixture uses it."""
    from paper2_generators_v01 import generate
    directory=Path(directory);path=directory/f'{mechanism}_{index:03}.npz';side=path.with_suffix('.json')
    if path.exists():
        return load_population(mechanism,index,config,directory,context)
    if side.exists() or path.with_suffix('.partial').exists():raise RuntimeError('population partial needs recovery')
    s=seed(f"{config['population_seed_prefix']}|{mechanism}|{index}")
    pub,truth=generate(mechanism,s,N=config['N'],K=config['K'])
    if not np.array_equal(truth.omitted,truth.lower) or not np.array_equal(truth.lower,truth.upper):raise ValueError('not exact measurements')
    partial=path.with_suffix('.partial')
    with partial.open('xb') as f:
        np.savez_compressed(f,**pub.__dict__,omitted=truth.omitted)
        f.flush();os.fsync(f.fileno())
    os.rename(partial,path)
    new_json(side,{'sha256':file_sha(path),'context_sha256':context,'seed':s,'mechanism':mechanism,'index':index})
    return Frame.from_public(pub,f'{mechanism}:{index}'),truth.omitted

def load_population(mechanism,index,config,directory,context):
    """Read and verify an existing population; never construct or regenerate it."""
    from paper2_generators_v01 import PublicFrame
    path=Path(directory)/f'{mechanism}_{index:03}.npz';info=load(path.with_suffix('.json'))
    expected=seed(f"{config['population_seed_prefix']}|{mechanism}|{index}")
    if info['context_sha256']!=context or info['sha256']!=file_sha(path):raise ValueError('population drift')
    if info['seed']!=expected or info['mechanism']!=mechanism or info['index']!=index:raise ValueError('population seed identity')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(PublicFrame.__dataclass_fields__)|{'omitted'}:raise ValueError('population schema')
        pub=PublicFrame(**{name:z[name].copy() for name in PublicFrame.__dataclass_fields__});y=z['omitted'].copy()
    if y.shape!=(config['N'],) or not np.issubdtype(y.dtype,np.integer) or np.any(y<0) or np.any(y>config['K']):raise ValueError('population count support')
    frame=Frame.from_public(pub,f'{mechanism}:{index}')
    if frame.N!=config['N'] or frame.K!=config['K']:raise ValueError('population size')
    return frame,y

def ensure_clean_partials(out):
    if list(Path(out).rglob('*.partial')):raise RuntimeError('preserved partial requires explicit recovery')
    for p in Path(out).rglob('*.json.gz'):
        if not p.with_name(p.name+'.sha256.json').is_file():raise RuntimeError('chunk without sidecar')
    for p in Path(out).rglob('*.npz'):
        if not p.with_suffix('.json').is_file():raise RuntimeError('population without sidecar')
    for p in Path(out).rglob('*.json.gz.sha256.json'):
        if not p.with_name(p.name[:-12]).is_file():raise RuntimeError('sidecar without chunk')

def run(manifest,approval,output,root):
    out,release,config=execution_permission(manifest,approval,output,root)
    context=file_sha(manifest)
    with WorkerLock(owned(Path(root)/'80_LOGS'/'COMPARATOR_WORKER.lock',root)):
        if out.exists():
            if load(out/'RUN_CONTEXT.json')['manifest_sha256']!=context:raise ValueError('wrong output context')
            if (out/'COMPLETE.json').exists():raise RuntimeError('completed study must not restart')
            if list(out.glob('FAILURE_*.json')):raise RuntimeError('unresolved failure; preserve and review before restart')
            ensure_clean_partials(out)
        else:
            out.mkdir(parents=True)
            new_json(out/'RUN_CONTEXT.json',{'manifest_sha256':context,'approval_sha256':file_sha(approval),'created_utc':utc(),'config':config})
        if (out/'FINAL_SUMMARY.json').exists():raise RuntimeError('summary without completion marker needs review')
        (out/'populations').mkdir(exist_ok=True);(out/'chunks').mkdir(exist_ok=True)
        receipt=out/f'LAUNCH_{time.time_ns()}.json'
        new_json(receipt,{'utc':utc(),'pid':os.getpid(),'parent_pid':os.getppid(),'manifest_sha256':context})
        try:
            # No outcome display during construction. Materialize and hash all
            # prescribed populations before processing any performance chunks.
            for mechanism in config['mechanisms']:
                for index in range(config['populations_per_mechanism']):construct_population(mechanism,index,config,out/'populations',context)
            bank={}
            for mechanism in config['mechanisms']:
                bank={i:construct_population(mechanism,i,config,out/'populations',context) for i in range(config['populations_per_mechanism'])}
                for budget in config['budgets']:
                    for start in range(0,config['replicates_per_cell'],config['chunk_size']):
                        path=out/'chunks'/f'{mechanism}_b{budget}_r{start:05}.json.gz'
                        if path.exists():
                            existing=read_chunk(path,context)
                            from .review import verify_row
                            if [(r['mechanism'],r['budget'],r['replicate']) for r in existing['rows']]!=[(mechanism,budget,j) for j in range(start,start+config['chunk_size'])]:raise ValueError('checkpoint IDs')
                            for row in existing['rows']:verify_row(row,config)
                            continue
                        rows=[]
                        for rep in range(start,start+config['chunk_size']):
                            idx,order,folds,u=streams(config,mechanism,budget,rep,config['N']);frame,y=bank[idx]
                            rows.append(paired(frame,y,budget,order,folds,u,{'mechanism':mechanism,'budget':budget,'replicate':rep,'population_index':idx}))
                        write_chunk(path,{'context_sha256':context,'rows':rows})
                        print(json.dumps({'utc':utc(),'completed_chunk':path.name}),flush=True)
            # Independently recompute and verify all rows before completion.
            from .review import summarize
            summary=summarize(out,context,config,strict=True,replay=True)
            new_json(out/'FINAL_SUMMARY.json',summary)
            new_json(out/'COMPLETE.json',{'utc':utc(),'manifest_sha256':context,'summary_sha256':file_sha(out/'FINAL_SUMMARY.json'),'counts':summary['counts']})
        except BaseException as e:
            new_json(out/f'FAILURE_{time.time_ns()}.json',{'utc':utc(),'type':type(e).__name__,'message':str(e),'preserved':True})
            raise
