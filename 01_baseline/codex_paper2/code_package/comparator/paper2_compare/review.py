"""Independent raw-record arithmetic and summary validation (no estimator calls)."""
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta,t
from .methods import METHODS,WALD
from .storage import read_chunk,load,file_sha

INTERVAL_IDS=tuple(METHODS)+tuple(m+'_wald' for m in WALD)
class Totals:
    """Compensated streaming sums; do not retain millions of Python tuples."""
    def __init__(self):self.n=0;self.s=None;self.c=None
    def add(self,values):
        if self.s is None:self.s=[0.]*len(values);self.c=[0.]*len(values)
        self.n+=1
        for j,v in enumerate(values):
            y=float(v)-self.c[j];t=self.s[j]+y;self.c[j]=(t-self.s[j])-y;self.s[j]=t

def cp(successes,n,alpha):
    if not 0<=successes<=n or n<1 or not 0<alpha<1:raise ValueError('binomial inputs')
    return [0. if successes==0 else float(beta.ppf(alpha/2,successes,n-successes+1)),
            1. if successes==n else float(beta.ppf(1-alpha/2,successes+1,n-successes))]

def mc_interval(values,rngwidth,J=24):
    a=np.asarray(values,dtype=float);n=len(a)
    if n<2 or not np.all(np.isfinite(a)):raise ValueError('MC inputs')
    mean=float(a.mean());var=float(np.var(a,ddof=1));se=math.sqrt(var/n)
    approx=float(t.ppf(1-.05/(2*J),n-1))*se
    ll=math.log(4*J/.05);bound=math.sqrt(2*var*ll/n)+7*rngwidth*ll/(3*(n-1))
    centered=a-mean;den=float(np.sqrt(centered@centered))
    return {'mean':mean,'mc_standard_error':se,'approximate_bonferroni_interval':[mean-approx,mean+approx],
            'bounded_empirical_bernstein_interval':[mean-bound,mean+bound],
            'max_centered_contribution_fraction':None if den==0 else float(np.max(np.abs(centered))/den),
            'sample_size':n,'known_range':rngwidth,'approximate_interval_not_finite_sample_certificate':True}

def verify_row(row,config):
    N=config['N'];K=config['K'];capacity=N*K;b=row['budget']
    if row['mechanism'] not in config['mechanisms'] or b not in config['budgets']:raise ValueError('cell')
    if not isinstance(row['replicate'],int) or not 0<=row['replicate']<config['replicates_per_cell']:raise ValueError('replicate')
    if not isinstance(row['population_index'],int) or not 0<=row['population_index']<config['populations_per_mechanism']:raise ValueError('population')
    if row['capacity']!=capacity:raise ValueError('capacity')
    if [p['method'] for p in row['points']]!=list(METHODS):raise ValueError('method completeness/order')
    if len(row['intervals'])!=len(INTERVAL_IDS) or set(i['interval_id'] for i in row['intervals'])!=set(INTERVAL_IDS):raise ValueError('interval completeness')
    points={p['method']:p for p in row['points']};truths=set()
    for method,p in points.items():
        if p['budget']!=b or p['labels']!=b:raise ValueError('point budget')
        if not all(math.isfinite(p[k]) for k in ('estimate','truth','squared_error','signed_error','elapsed_seconds')):raise ValueError('nonfinite point')
        B=p['truth'];truths.add(B)
        if not 0<=B<=capacity or B!=int(B):raise ValueError('truth')
        e=p['estimate']-B
        if p['elapsed_seconds']<0 or e*e/capacity**2>(529 if method.startswith(('ppat_','active_')) else 4)+1e-10:raise ValueError('point support/timing')
        if not math.isclose(p['squared_error'],e*e,rel_tol=1e-12,abs_tol=1e-12) or p['signed_error']!=e:raise ValueError('squared error arithmetic')
        trace=row['traces'][method];ids=trace['measurement_ids'];vals=trace['measurement_values']
        if len(ids)!=b or len(set(ids))!=b or any(not isinstance(i,int) or not 0<=i<N for i in ids):raise ValueError('label access IDs')
        if len(vals)!=b or any(not isinstance(v,int) or not 0<=v<=K for v in vals):raise ValueError('label values')
        if method.startswith(('active_','ppat_')):
            if len(trace['draws'])!=b or [d['id'] for d in trace['draws']]!=ids:raise ValueError('active ledger')
            for j,d in enumerate(trace['draws']):
                if not math.isfinite(d['p']) or d['p']<.1/(N-j)-1e-14 or d['p']>1+1e-14:raise ValueError('propensity bounds')
                if len(d['probabilities_sha256'])!=64 or abs(d['normalization_error'])>1e-12:raise ValueError('propensity evidence')
            for f in trace['fits']:
                if f['fit_ids']!=ids[:f['before_draw']]:raise ValueError('future fit ID')
        if method.startswith('frame_cf2_'):
            for f in trace['folds']:
                if set(f['training_ids'])&set(f['fold_ids']):raise ValueError('cross-fit leakage')
                if not set(f['sample_ids']).issubset(f['fold_ids']):raise ValueError('fold inference IDs')
    if len(truths)!=1:raise ValueError('unpaired truth')
    B=next(iter(truths))
    for r in row['intervals']:
        ident=r['interval_id'];method=r['method']
        if ident not in (method,method+'_wald') or method not in points:raise ValueError('interval owner')
        expected='APPROXIMATE_WALD' if ident.endswith('_wald') or method in ('ff_power','ff_strat_power') else 'FINITE_SAMPLE_BOUNDED'
        if r['guarantee']!=expected:raise ValueError('guarantee class')
        if r['status']=='OK':
            lo=r['lower'];hi=r['upper']
            if not math.isfinite(lo+hi) or not 0<=lo<=hi<=capacity+1e-10:raise ValueError('interval bounds')
            if not math.isclose(r['scored_width'],hi-lo,abs_tol=1e-10) or r['raw_width']!=hi-lo:raise ValueError('width arithmetic')
            covers=lo<=B<=hi
        elif r['status']=='EMPTY':
            if r['lower'] is not None or r['upper'] is not None or r['raw_width'] is not None or r['scored_width']!=capacity:raise ValueError('empty penalty')
            covers=False
        else:raise ValueError('unexpected interval failure')
        if r['covers']!=covers:raise ValueError('coverage arithmetic')
    return points,{r['interval_id']:r for r in row['intervals']}

def summarize(out,context,config,strict=True,replay=False):
    out=Path(out);point_stats=defaultdict(Totals);interval_stats=defaultdict(Totals);chunks=0
    mi={m:j for j,m in enumerate(config['mechanisms'])};bi={b:j for j,b in enumerate(config['budgets'])}
    pi={m:j for j,m in enumerate(config['primary_mechanisms'])};methods={m:j for j,m in enumerate(METHODS)}
    seen=np.zeros((len(mi),len(bi),config['replicates_per_cell']),dtype=bool)
    primary=np.full((len(pi),config['replicates_per_cell'],2,len(METHODS)),np.nan)
    bank={};bank_mechanism=None
    if replay:
        from .engine import load_population,streams,ensure_clean_partials
        from .trace_review import replay_row
        ensure_clean_partials(out)
        if len(list((out/'populations').glob('*.npz')))!=config['expected']['populations']:raise ValueError('population count')
        # Verify all 1000 identities even if one was not sampled by chance.
        for mechanism in config['mechanisms']:
            for index in range(config['populations_per_mechanism']):load_population(mechanism,index,config,out/'populations',context)
    for path in sorted((out/'chunks').glob('*.json.gz')):
        chunk=read_chunk(path,context);chunks+=1
        if len(chunk['rows'])!=config['chunk_size']:raise ValueError('chunk length')
        keys=[(r['mechanism'],r['budget'],r['replicate']) for r in chunk['rows']]
        mech,b,start=keys[0]
        if start%config['chunk_size'] or keys!=[(mech,b,j) for j in range(start,start+config['chunk_size'])]:raise ValueError('chunk boundaries')
        if path.name!=f'{mech}_b{b}_r{start:05}.json.gz':raise ValueError('chunk filename')
        for row,key in zip(chunk['rows'],keys):
            p,ii=verify_row(row,config)
            position=(mi[mech],bi[b],row['replicate'])
            if seen[position]:raise ValueError('duplicate paired replicate')
            seen[position]=True
            if replay:
                if bank_mechanism!=mech:bank={};bank_mechanism=mech
                idx,order,folds,u=streams(config,mech,b,row['replicate'],config['N'])
                if idx!=row['population_index']:raise ValueError('population selection stream')
                if idx not in bank:bank[idx]=load_population(mech,idx,config,out/'populations',context)
                frame,y=bank[idx];replay_row(row,frame,y,order,folds,u)
            for method,v in p.items():point_stats[(mech,b,method)].add((v['signed_error'],v['squared_error'],v['elapsed_seconds']))
            for name,v in ii.items():interval_stats[(mech,b,name)].add((v['covers'],v['scored_width'],v['status']!='OK',v['raw_width'] if v['raw_width'] is not None else 0.))
            if b==config['primary_budget'] and mech in config['primary_mechanisms']:
                primary[pi[mech],row['replicate'],0]=[p[k]['squared_error']/row['capacity']**2 for k in METHODS]
                primary[pi[mech],row['replicate'],1]=[ii[k]['scored_width']/row['capacity'] for k in METHODS]
    n_seen=int(seen.sum())
    counts={'paired_chunks':chunks,'paired_replicates':n_seen,'point_method_records':n_seen*len(METHODS),
            'interval_records':n_seen*len(INTERVAL_IDS),'point_cells':len(point_stats),'interval_cells':len(interval_stats)}
    if strict:
        for k,v in counts.items():
            if config['expected'][k]!=v:raise ValueError('completion count '+k)
        for rr in list(point_stats.values())+list(interval_stats.values()):
            if rr.n!=config['replicates_per_cell']:raise ValueError('cell denominator')
        if not np.all(seen) or not np.all(np.isfinite(primary)):raise ValueError('incomplete paired grid')
    point_rows=[];interval_rows=[]
    for (mech,b,method),rr in sorted(point_stats.items()):
        point_rows.append({'mechanism':mech,'budget':b,'method':method,'n':rr.n,'bias':rr.s[0]/rr.n,'mse':rr.s[1]/rr.n,'mean_seconds':rr.s[2]/rr.n})
    family=len(config['mechanisms'])*len(config['budgets'])*len(INTERVAL_IDS)
    for (mech,b,name),rr in sorted(interval_stats.items()):
        success=int(rr.s[0]);empty=int(rr.s[2])
        interval_rows.append({'mechanism':mech,'budget':b,'interval_id':name,'n':rr.n,'coverage':success/rr.n,
                              'mc_cp_marginal_95':cp(success,rr.n,.05),'mc_cp_family_95':cp(success,rr.n,.05/family),
                              'mean_scored_width':rr.s[1]/rr.n,'empty_intervals':empty,
                              'mean_raw_width_among_nonempty':None if empty==rr.n else rr.s[3]/(rr.n-empty)})
    effects=[]
    if strict:
        for left,right in config['primary_pairs']:
            for metric,j in [('normalized_squared_error',0),('normalized_scored_width',1)]:
                l=primary[:,:,j,methods[left]];r=primary[:,:,j,methods[right]];a=np.mean(l-r,axis=0)
                bound=lambda name:529 if name.startswith(('active_','ppat_')) else 4
                width=bound(left)+bound(right) if j==0 else 2
                lmean=float(l.mean());rmean=float(r.mean())
                unit_scale=(config['N']*config['K'])**(2 if j==0 else 1)
                eff=dict(mc_interval(a,width),left=left,right=right,metric=metric)
                eff.update(left_normalized_mean=lmean,right_normalized_mean=rmean,descriptive_ratio=None if rmean==0 else lmean/rmean,
                    ratio_status='UNDEFINED_ZERO_DENOMINATOR' if rmean==0 else 'DESCRIPTIVE_NOT_A_CONFIDENCE_BOUND',raw_unit_scale=unit_scale,
                    raw_unit_mean_difference=eff['mean']*unit_scale,raw_unit_bounded_interval=[v*unit_scale for v in eff['bounded_empirical_bernstein_interval']])
                effects.append(eff)
    return {'status':'COMPLETE_RECOMPUTED' if strict else 'PARTIAL_FIXTURE_ONLY','counts':counts,'points':point_rows,'intervals':interval_rows,'primary_effects':effects,
            'original_medium':'NOT_ATTAINED','new_success_gate':None,'scope':'Conditional on fixed new synthetic population banks; not physical/external validation','mc_families_separate':True,'full_trace_replay':replay}
