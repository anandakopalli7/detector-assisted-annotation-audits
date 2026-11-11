"""Finite paired study runner with atomic per-path checkpoints and bounded workers."""
from pathlib import Path
import argparse, gzip, hashlib, json, multiprocessing as mp, os, queue, signal, sys, time, traceback
from datetime import datetime,timezone
# Prevent each process from launching its own BLAS thread pool.
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'04_code'))

def utc():return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x,new=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if new and p.exists():raise FileExistsError(p)
    tmp=p.with_suffix(p.suffix+'.partial')
    with tmp.open('xb' if new else 'wb') as f:f.write(canonical(x));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def verify_sources(manifest):
    for name,h in manifest['sources'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT.resolve()) or sha(p)!=h:raise ValueError('Source drift: '+name)

def ids(phase):
    cells=[f'M{i:02}' for i in range(1,11)] if phase=='synthetic' else ['COCO_'+s for s in ('uniform','height_dependent','rare')]
    return [(c,r) for c in cells for r in range(1000)]

def checkpoint_path(out,job):return Path(out)/'paths'/f'{job[0]}_{job[1]:04}.json.gz'

def store_checkpoint(path,context,job,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix+'.sha256').exists():raise FileExistsError(path)
    raw=canonical({'context_sha256':context,'job':list(job),'rows':rows})
    compressed=gzip.compress(raw,compresslevel=3,mtime=0);part=path.with_suffix(path.suffix+'.partial')
    with part.open('xb') as f:f.write(compressed);f.flush();os.fsync(f.fileno())
    os.replace(part,path)
    save(path.with_suffix(path.suffix+'.sha256'),{'sha256':sha(path),'context_sha256':context},new=True)

def load_checkpoint(path,context,job):
    path=Path(path);side=json.loads(path.with_suffix(path.suffix+'.sha256').read_text())
    if side['context_sha256']!=context or side['sha256']!=sha(path):raise ValueError('Checkpoint hash/context mismatch')
    obj=json.loads(gzip.decompress(path.read_bytes()))
    if obj['context_sha256']!=context or obj['job']!=list(job):raise ValueError('Checkpoint job mismatch')
    return obj['rows']

def validate(rows,job,budgets=(50,100,200)):
    from ojcs_engine import STRONG
    from paper2_compare.methods import METHODS
    from paper2_compare.review import verify_row
    if [r['budget'] for r in rows]!=list(budgets):raise ValueError('Budget coverage')
    for r in rows:
        if (r['mechanism'],r['replicate'])!=tuple(job):raise ValueError('Row identity')
        conf={'N':int(r['capacity']//20),'K':20,'mechanisms':[job[0]],'budgets':list(budgets),'replicates_per_cell':job[1]+1,'populations_per_mechanism':job[1]+1}
        verify_row(r,conf)
        if job[0].startswith('COCO_') and (r['population_index']!=0 or r['lineage']['generator_seed'] is not None or r['lineage']['fixed_frame_id']!=r['frame_id']):raise ValueError('Fixed external population mislabeled')
        if {p['method'] for p in r['points']}!=set(METHODS):raise ValueError('Point coverage')
        if [i['interval_id'] for i in r['strong_intervals']]!=list(STRONG):raise ValueError('Interval coverage')
        B=r['points'][0]['truth']
        for i in r['strong_intervals']:
            if i['status']=='OK':
                if not 0<=i['lower']<=i['upper']<=r['capacity']:raise ValueError('Interval support')
                if i['scored_width']!=i['upper']-i['lower']:raise ValueError('Interval width')
                if i['covers']!=(i['lower']<=B<=i['upper']):raise ValueError('Coverage arithmetic')
                if not 1<=i['accepted_size']<=i['upper']-i['lower']+1:raise ValueError('Set size')
            elif i['status']=='EMPTY':
                if i['accepted_size']!=0 or i['covers'] or i['scored_width']!=r['capacity']:raise ValueError('Empty penalty')
            else:raise ValueError('Interval status')
            if i['accepted_set_covers'] and not i['covers']:raise ValueError('Set-hull containment')
            if i['all_prefix_certificate']!=i['accepted_set_covers']:raise ValueError('Running coverage')

def external_frames():
    import numpy as np
    from paper2_generators_v01 import PublicFrame
    d=ROOT/'04_code'/'data'/'coco';receipt=json.loads((d/'FRAMES_COMPLETE.json').read_text());frames={}
    for s,rec in receipt['frames'].items():
        p=d/rec['path']
        if sha(p)!=rec['sha256']:raise ValueError('External frame hash')
        with np.load(p,allow_pickle=False) as z:
            pub=PublicFrame(**{k:z[k].copy() for k in PublicFrame.__dataclass_fields__});y=z['omitted'].copy()
        frames['COCO_'+s]=(pub,y,rec['frame_id'])
    return frames

def worker(task_queue,result_queue,out,context,phase,fixture):
    try:
        from ojcs_engine import evaluate_pair,synthetic
        frames=external_frames() if phase=='external' and not fixture else {}
        while True:
            job=task_queue.get()
            if job is None:return
            started=time.monotonic();result_queue.put(('START',os.getpid(),job,started))
            if fixture:
                from paper2_compare.fixtures import make_fixture
                pub,truth,_=make_fixture(N=40);y=truth.omitted;fid='FABRICATED_FIXED_FRAME';budgets=(20,40)
            elif phase=='synthetic':pub,y=synthetic(*job);fid=None;budgets=(50,100,200)
            else:pub,y,fid=frames[job[0]];budgets=(50,100,200)
            rows=evaluate_pair(pub,y,*job,budgets=budgets,frame_id=fid)
            validate(rows,job,budgets)
            store_checkpoint(checkpoint_path(out,job),context,job,rows)
            result_queue.put(('DONE',os.getpid(),job,time.monotonic()-started))
    except BaseException:
        result_queue.put(('ERROR',os.getpid(),None,traceback.format_exc()));raise

def run(manifest_path,phase,output,workers,max_seconds,fixture=False):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());context=sha(manifest_path)
    if manifest['schema']!='OJCS_EXECUTION_V1' or manifest['status']!='FROZEN':raise ValueError('Unfrozen execution context')
    if fixture!=bool(manifest['fixture']):raise ValueError('Fixture/performance boundary')
    if not 1<=workers<=32 or not 1<=max_seconds<=12*3600:raise ValueError('Execution bound')
    verify_sources(manifest)
    if not fixture:
        approval=json.loads((ROOT/'03_protocol'/'BUDGET_APPROVAL.json').read_text())
        if approval['status']!='APPROVED' or approval['ceiling_usd']!=20:raise ValueError('Budget approval')
        if phase=='external' and sha(ROOT/'04_code'/'data'/'coco'/'FRAMES_COMPLETE.json')!=manifest['external_frames_sha256']:raise ValueError('External manifest mismatch')
    out=Path(output).resolve();allowed=(ROOT/'05_results').resolve()
    if not out.is_relative_to(allowed):raise ValueError('Output outside task results')
    out.mkdir(parents=True,exist_ok=True);lock=out/'RUN.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    procs=[];task_queue=None;result_queue=None;began=time.monotonic()
    try:
        if (out/'COMPLETE.json').exists():raise RuntimeError('Completed run must not restart')
        if list(out.glob('FAILURE_*.json')):raise RuntimeError('Failure requires explicit review')
        if list(out.rglob('*.partial')):raise RuntimeError('Partial files preserved for review')
        jobs=[('FABRICATED_RUNNER',i) for i in range(3)] if fixture else ids(phase)
        expected={checkpoint_path(out,j).name for j in jobs}
        if any(p.name not in expected for p in (out/'paths').glob('*.json.gz')):raise ValueError('Unexpected path')
        context_path=out/'RUN_CONTEXT.json'
        if context_path.exists():
            if json.loads(context_path.read_text())['context_sha256']!=context:raise ValueError('Resume context changed')
        else:save(context_path,{'context_sha256':context,'phase':phase,'jobs':len(jobs),'fixture':fixture,'created_utc':utc()},new=True)
        pending=[];completed=0
        for j in jobs:
            p=checkpoint_path(out,j)
            if p.exists():validate(load_checkpoint(p,context,j),j,(20,40) if fixture else (50,100,200));completed+=1
            else:
                if p.with_suffix(p.suffix+'.sha256').exists():raise ValueError('Orphan sidecar')
                pending.append(j)
        save(out/f'LAUNCH_{time.time_ns()}.json',{'utc':utc(),'workers':workers,'resume_verified':completed,'remaining':len(pending),'max_seconds':max_seconds},new=True)
        ctx=mp.get_context('spawn');task_queue=ctx.Queue();result_queue=ctx.Queue()
        for _ in range(min(workers,len(pending))):
            p=ctx.Process(target=worker,args=(task_queue,result_queue,str(out),context,phase,fixture));p.start();procs.append(p)
        submitted=0;running={};finished=0;last=time.monotonic()
        for j in pending[:len(procs)]:task_queue.put(j);submitted+=1
        while finished<len(pending):
            elapsed=time.monotonic()-began
            if elapsed>max_seconds:raise TimeoutError('Fixed run deadline')
            if any(time.monotonic()-v[1]>600 for v in running.values()):raise TimeoutError('Individual path exceeded 600 seconds')
            if any(p.exitcode not in (None,0) for p in procs):raise RuntimeError('Worker exited unexpectedly')
            try:kind,pid,job,value=result_queue.get(timeout=2)
            except queue.Empty:
                if time.monotonic()-last>120 and not running:raise TimeoutError('Worker initialization timeout')
                continue
            last=time.monotonic()
            if kind=='ERROR':raise RuntimeError(value)
            if kind=='START':running[pid]=(job,value)
            if kind=='DONE':
                running.pop(pid,None);finished+=1;completed+=1
                state={'utc':utc(),'state':'RUNNING','phase':phase,'completed':completed,'total':len(jobs),'elapsed_seconds':elapsed,'active_workers':len(running),'last_path_seconds':value,'scientific_outcomes_displayed':False}
                save(out/'PROGRESS.json',state);print(json.dumps(state),flush=True)
                if submitted<len(pending):task_queue.put(pending[submitted]);submitted+=1
        for _ in procs:task_queue.put(None)
        for p in procs:p.join(timeout=15)
        if any(p.is_alive() for p in procs):raise RuntimeError('Worker shutdown timeout')
        for j in jobs:validate(load_checkpoint(checkpoint_path(out,j),context,j),j,(20,40) if fixture else (50,100,200))
        save(out/'COMPLETE.json',{'utc':utc(),'context_sha256':context,'paths':len(jobs),'phase':phase,'seconds':time.monotonic()-began,'row_validation':'PASS','fixture':fixture},new=True)
    except BaseException as e:
        save(out/f'FAILURE_{time.time_ns()}.json',{'utc':utc(),'type':type(e).__name__,'error':str(e),'preserved':True},new=True);raise
    finally:
        for p in procs:
            if p.is_alive():p.terminate();p.join(timeout=5)
            if p.is_alive():p.kill();p.join(timeout=5)
        if task_queue is not None:task_queue.close()
        if result_queue is not None:result_queue.close()
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--phase',choices=['synthetic','external'],required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--fixture',action='store_true');a=p.parse_args()
    def stop(signum,frame):raise InterruptedError('Shutdown signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    run(a.manifest,a.phase,a.output,a.workers,a.max_seconds,a.fixture)
