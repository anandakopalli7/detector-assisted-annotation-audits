"""Paired OJCS comparisons. This module does not launch a performance study."""
from pathlib import Path
import sys,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code'
BASE=ROOT/'01_baseline'/'codex_paper2'/'code_package'
sys.path[:0]=[str(CODE/'deps'),str(CODE),str(BASE/'comparator'),str(BASE/'integer')]
import numpy as np
from paper2_compare.methods import Frame,METHODS,digest_array,interval_row
from paper2_compare.engine import paired
from paper2_compare.review import verify_row
from paper2_sequential_v01 import build_trace,predictable_risk
from paper2_candidate_v01 import make_forecaster,UPDATES
from paper2_generators_v01 import generate
from ojcs_fast_running import RunningRLFA,MonotoneConBo

NAMESPACE='PROGRAMMEB_OJCS_20261110_EVALUATION_V1'
STRONG=('rlfa_fixed_certified','rlfa_past_certified','conbo_wor_certified')

def seed(*parts):
    # Prior retained generators use <=64-bit seed integers. This disjoint
    # integer namespace avoids reusing them; SeedSequence consumes all bits.
    text=NAMESPACE+'|'+'|'.join(map(str,parts))
    return (1<<128)+int.from_bytes(hashlib.sha256(text.encode()).digest()[:16],'big')

def streams(cell,rep,N,budget):
    def rng(tag):return np.random.Generator(np.random.PCG64(seed(cell,rep,tag)))
    return (rng('uniform_order').permutation(N),
            [rng('fold_'+str(k)).permutation(N//2) for k in (0,1)],
            rng('active_uniforms').random(budget))

def make_trace(frame,y,order,budget):
    return build_trace(np.full(frame.N,frame.K),order,int(budget),
        lambda i:(int(y[i]),int(y[i])),make_forecaster(frame.X,frame.q,frame.K),UPDATES)

def strong_intervals(frame,y,order,budgets):
    budgets=tuple(sorted(budgets));trace=make_trace(frame,y,order,budgets[-1])
    models={STRONG[0]:RunningRLFA(frame.N,frame.K),
            STRONG[1]:RunningRLFA(frame.N,frame.K),
            STRONG[2]:MonotoneConBo(frame.N,frame.K)}
    owners={STRONG[0]:'raw_difference',STRONG[1]:'seq_detector_100',STRONG[2]:'direct_HT'}
    elapsed={name:0. for name in models};snapshots={};B=int(sum(y));C=frame.N*frame.K
    for t in range(trace.budget):
        f=trace.forecasts[int(trace.forecast_index[t])]
        if f.before_draw>t or f.fit_ids!=tuple(int(i) for i in order[:f.before_draw]):raise ValueError('Future-label lineage')
        i=int(order[t]);observed=int(trace.lower[t])
        for name,m in models.items():
            start=time.perf_counter()
            if name==STRONG[2]:m.prepare()
            else:
                predictions=frame.q if name==STRONG[0] else f.predictions
                m.prepare(tuple(predictions[j] for j in m.remaining))
            m.observe(i,observed)
            elapsed[name]+=time.perf_counter()-start
        b=t+1
        if b in budgets:
            rows=[]
            for name,m in models.items():
                start=time.perf_counter();accepted=tuple(m.accepted(True));elapsed[name]+=time.perf_counter()-start
                ok=bool(accepted);lo=accepted[0] if ok else None;hi=accepted[-1] if ok else None
                row=interval_row(lo,hi,'OK' if ok else 'EMPTY','FINITE_SAMPLE_BOUNDED',C)
                row.update(interval_id=name,method=owners[name],covers=bool(ok and lo<=B<=hi),
                    accepted_set_covers=B in accepted,all_prefix_certificate=B in accepted,
                    accepted_size=len(accepted),accepted_sha256=hashlib.sha256(np.asarray(accepted,dtype='<i8').tobytes()).hexdigest(),
                    full_logical_support=bool(ok and len(accepted)==len(m.support())),
                    elapsed_seconds=elapsed[name],sampling='UNIFORM_WITHOUT_REPLACEMENT',running=True)
                rows.append(row)
            snapshots[b]=rows
    return snapshots,trace

def evaluate_pair(public,y,cell,rep,budgets=(50,100,200),frame_id=None):
    y=np.asarray(y,dtype=int);budgets=tuple(sorted(map(int,budgets)))
    frame=Frame.from_public(public,frame_id or cell+':'+str(rep))
    if y.shape!=(frame.N,) or np.any(y<0) or np.any(y>frame.K):raise ValueError('Truth support')
    order,folds,uniforms=streams(cell,rep,frame.N,max(budgets))
    strong,fulltrace=strong_intervals(frame,y,order,budgets)
    result=[]
    config={'N':frame.N,'K':frame.K,'mechanisms':[cell],'budgets':list(budgets),
            'replicates_per_cell':rep+1,'populations_per_mechanism':1 if frame_id is not None else rep+1}
    for b in budgets:
        row=paired(frame,y,b,order,folds,uniforms,{'mechanism':cell,'budget':b,'replicate':rep,'population_index':0 if frame_id is not None else rep})
        verify_row(row,config)
        prefix=make_trace(frame,y,order,b)
        points={p['method']:p for p in row['points']}
        if prefix.estimate.hex()!=points['seq_detector_100']['estimate'].hex():raise ArithmeticError('Sequential point changed')
        if not np.array_equal(prefix.lower,fulltrace.lower[:b]):raise ArithmeticError('Prefix observations changed')
        for a,f in zip(prefix.forecasts,fulltrace.forecasts):
            if a.fit_ids!=f.fit_ids or not np.array_equal(a.predictions,f.predictions):raise ArithmeticError('Prefix forecasts changed')
        fits=[{'before_draw':f.before_draw,'fit_ids':list(f.fit_ids),'prediction_sha256':digest_array(f.predictions)} for f in prefix.forecasts]
        if fits!=row['traces']['seq_detector_100']['fits']:raise ArithmeticError('Anchor forecast identity changed')
        row['strong_intervals']=strong[b]
        row['sequential_risk_score']=predictable_risk(prefix,y)
        row['lineage']={'shared_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            'truth_sha256':hashlib.sha256(y.astype('<i8').tobytes()).hexdigest(),
            'features_sha256':digest_array(frame.X),'proxy_sha256':digest_array(frame.q),
            'forecasts':fits,'generator_seed':None if frame_id is not None else str(seed('population',cell,rep)),
            'fixed_frame_id':frame_id}
        result.append(row)
    return result

def synthetic(cell,rep):
    if cell not in {f'M{i:02}' for i in range(1,11)}:raise ValueError('Unplanned mechanism')
    public,truth=generate(cell,seed('population',cell,rep),N=1000,K=20)
    return public,truth.omitted

def scientific_view(value):
    """Strip timing only when comparing deterministic replay outputs."""
    if isinstance(value,dict):return {k:scientific_view(v) for k,v in value.items() if k!='elapsed_seconds'}
    if isinstance(value,list):return [scientific_view(v) for v in value]
    return value
