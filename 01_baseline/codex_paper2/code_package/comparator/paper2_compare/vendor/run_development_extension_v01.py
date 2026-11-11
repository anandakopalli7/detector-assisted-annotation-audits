"""Diagnose development failures; never writes outside the Paper 2 root."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path(__file__).parent))
import hashlib,json,time,math
import numpy as np
from paper2_core_v01 import ResidualCS,difference_total,exact_design_variance,ridge_predictions,tree_predictions,allocate_strata
from paper2_generators_v01 import generate

def drive_features(public):
    X=public.features
    counts=np.column_stack((public.original,public.detector,public.unmatched)).astype(float)
    extra=np.empty_like(counts)
    for d in np.unique(public.drive):
        ids=np.flatnonzero(public.drive==d)
        extra[ids]=(counts[ids].sum(axis=0)-counts[ids])/(len(ids)-1)
    return np.column_stack((X,extra))

def offset_ridge(X,q,P,y,penalty,K=20):
    center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1.
    Z=np.column_stack((np.ones(len(X)),(X-center)/scale))
    reg=np.eye(Z.shape[1])*penalty;reg[0,0]=0.
    coef=np.linalg.solve(Z[P].T@Z[P]+reg,Z[P].T@(y-q[P]))
    return np.clip(q+Z@coef,0,K)

def main():
    out=ROOT/'50_RESULTS/DEVELOPMENT_EXTENSION_v01';out.mkdir(exist_ok=False)
    sources=[Path(__file__),ROOT/'30_CODE/paper2_core_v01.py',ROOT/'30_CODE/paper2_generators_v01.py',ROOT/'20_DESIGN/DEVELOPMENT_EXTENSION_SPEC_v01.md']
    (out/'RECEIPT_v01.json').write_text(json.dumps({'development_only':True,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
    rows=[];began=time.perf_counter()
    raw=(out/'REPLICATES_v01.jsonl').open('x')
    for j in range(1,7):
        mechanism=f'M{j:02d}'
        for rep in range(100):
            seed=int.from_bytes(hashlib.sha256(f'P2_DEVELOPMENT_V01|{mechanism}|{rep}'.encode()).digest()[:8],'big')
            public,truth=generate(mechanism,seed);y=truth.omitted;N=len(y);K=20;b=200;p=40
            order=np.random.default_rng(seed^0x8349227).permutation(N);P=order[:p];R=order[p:];n=b-p;S=np.arange(n)
            X=drive_features(public)
            predictions={'direct_HT':np.zeros(N),'raw_difference':np.clip(public.unmatched,0,K),
                         'ridge_10':ridge_predictions(public.features,P,y[P],K,10),
                         'tree_depth2_leaf5':tree_predictions(public.features,P,y[P],K)}
            for penalty in (1,10,100):predictions[f'offset_ridge_drive_{penalty}']=offset_ridge(X,np.clip(public.unmatched,0,K),P,y[P],penalty,K)
            for name,m in predictions.items():
                learned=name not in ('direct_HT','raw_difference')
                frame=R if learned else np.arange(N);ids=S if learned else order[:b]
                point=(sum(y[P]) if learned else 0)+difference_total(m[frame],ids,y[frame[ids]])
                risk=exact_design_variance(y[frame],m[frame],len(ids))
                for strategy in ('predmix','grid'):
                    row={'mechanism':mechanism,'replicate':rep,'method':name,'strategy':strategy,'conditional_risk':risk,'squared_error':float((point-sum(y))**2)}
                    try:
                        ci=ResidualCS(public.caps[frame],m[frame],ids,y[frame[ids]],strategy=strategy).bounds()
                        truth_total=int(sum(y[frame]));row.update(status=ci.status,covers=ci.covers(truth_total),width=ci.scored_width(N*K))
                    except Exception as ex:row.update(status='IMPLEMENTATION_FAILURE',covers=False,width=float(N*K),error=repr(ex))
                    rows.append(row);raw.write(json.dumps(row)+'\n')
            # The fixed grid is also applied to stratification with the same allocation.
            strata=np.minimum(public.unmatched,2);groups=[np.flatnonzero(strata==h) for h in range(3)]
            sizes=np.array([len(g) for g in groups]);allocation=allocate_strata(sizes,b);H=sum(sizes>0)
            row={'mechanism':mechanism,'replicate':rep,'method':'stratified_HT','strategy':'grid'}
            try:
                low=up=0;status='OK';risk=0.;point=0.
                for h,g in enumerate(groups):
                    if not len(g):continue
                    chosen=order[strata[order]==h][:allocation[h]];lookup={int(v):k for k,v in enumerate(g)}
                    ci=ResidualCS(public.caps[g],np.zeros(len(g)),[lookup[int(v)] for v in chosen],y[chosen],alpha=.05/H,strategy='grid').bounds()
                    risk+=exact_design_variance(y[g],np.zeros(len(g)),len(chosen));point+=len(g)*y[chosen].mean()
                    if ci.status!='OK':status=ci.status
                    else:low+=ci.lower;up+=ci.upper
                row.update(status=status,covers=bool(status=='OK' and low<=sum(y)<=up),width=float(up-low) if status=='OK' else float(N*K),conditional_risk=risk,squared_error=float((point-sum(y))**2))
            except Exception as ex:row.update(status='IMPLEMENTATION_FAILURE',covers=False,width=float(N*K),error=repr(ex))
            rows.append(row);raw.write(json.dumps(row)+'\n')
        raw.flush();print(json.dumps({'completed':mechanism,'rows':len(rows)}),flush=True)
    raw.close()
    summary=[]
    for key in sorted(set((r['mechanism'],r['method'],r['strategy']) for r in rows)):
        rr=[r for r in rows if (r['mechanism'],r['method'],r['strategy'])==key]
        summary.append({'mechanism':key[0],'method':key[1],'strategy':key[2],'repetitions':len(rr),
                        'mean_risk':float(np.mean([r['conditional_risk'] for r in rr if 'conditional_risk' in r])),
                        'mean_width':float(np.mean([r['width'] for r in rr])),'coverage':float(np.mean([r['covers'] for r in rr])),
                        'failures':sum(r['status']=='IMPLEMENTATION_FAILURE' for r in rr)})
    def vv(r,e):return 20*r*e*(1-r*e/(1-r+r*e))
    variances=[vv(r,e) for r,e in ((.12,.04),(.30,.04),(.12,.16),(.30,.16),(.12,.171875))]
    variances.append(.3*vv(.12,.56875)+.7*vv(.12,.055))
    oracle={'scope':'generator-averaged lower bound granted known generator and M06 latent difficulty; NOT an evaluated method or fixed-benchmark impossibility',
            'conditional_residual_variance':variances,'learned_budget_risk':[4800*v for v in variances],
            'mean_learned_budget_risk':float(np.mean(variances)*4800),'fixed_no_pilot_mean_risk':float(np.mean(variances)*4000)}
    (out/'SUMMARY_v01.json').write_text(json.dumps({'development_only':True,'rows':summary,'optimistic_diagnostic':oracle,'elapsed_seconds':time.perf_counter()-began},indent=2))
    print(json.dumps({'completed':True,'rows':len(rows),'failures':sum(r['status']=='IMPLEMENTATION_FAILURE' for r in rows),'elapsed_seconds':time.perf_counter()-began},indent=2))

if __name__=='__main__':main()
