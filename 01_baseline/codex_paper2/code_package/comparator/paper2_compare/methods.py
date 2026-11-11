"""Real comparator implementation. Only the measurement callback reveals Y."""
from dataclasses import dataclass
import hashlib, math, time
import numpy as np
from scipy.stats import norm
from paper2_core_v01 import predictable_bets, ridge_predictions
from paper2_sequential_v01 import build_trace, SequentialCS
from paper2_betting_refinements_v01 import RefinedCS, refined_endpoint_interval
from paper2_candidate_v01 import make_forecaster, UPDATES
from paper2_stratified_tie_correction_v01 import allocate_exact
from run_development_extension_v01 import drive_features, offset_ridge

LEGACY=('direct_HT','stratified_HT','raw_difference','split_ridge10','split_offset100','seq_detector_100')
NEW=('ff_power','ff_strat_power','frame_cf2_offset100','frame_cf2_direct','active_count_lure','ppat_count_half','ppat_count_one')
METHODS=LEGACY+NEW
WALD=('direct_HT','raw_difference','split_offset100')

def digest_array(a):
    a=np.asarray(a, dtype='<f8')
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()

@dataclass(frozen=True)
class Frame:
    frame_id:str
    features:np.ndarray
    X:np.ndarray
    q:np.ndarray
    strata:np.ndarray
    K:int
    @property
    def N(self):return len(self.q)
    @classmethod
    def from_public(cls,public,frame_id):
        K=int(public.caps[0]);N=len(public.caps)
        if K<=0 or not np.all(public.caps==K):raise ValueError('uniform positive cap')
        features=np.array(public.features,dtype=float,copy=True)
        X=np.array(drive_features(public),dtype=float,copy=True)
        q=np.clip(public.unmatched,0,K).astype(float)
        strata=np.minimum(public.unmatched,2).astype(int)
        if features.shape!=(N,6) or X.shape!=(N,9):raise ValueError('feature shape')
        if not np.all(np.isfinite(X)) or np.any(q<0) or not np.all(np.isfinite(q)):raise ValueError('features')
        if np.any(strata<0) or np.any(strata>2):raise ValueError('strata')
        for a in (features,X,q,strata):a.flags.writeable=False
        return cls(frame_id,features,X,q,strata,K)

class Measurements:
    def __init__(self,get_measurement,N,K):
        self.callback=get_measurement;self.N=N;self.K=K;self.values={};self.ids=[]
    def get(self,i):
        i=int(i)
        if not 0<=i<self.N:raise ValueError('measurement ID')
        if i not in self.values:
            y=float(self.callback(i))
            if not math.isfinite(y) or not 0<=y<=self.K or y!=math.floor(y):raise ValueError('exact count outside cap')
            self.values[i]=int(y);self.ids.append(i)
        return self.values[i]
    def array(self,ids):return np.array([self.get(i) for i in ids],dtype=float)

def validate_order(order,N):
    a=np.asarray(order)
    if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer) or not np.array_equal(np.sort(a),np.arange(N)):
        raise ValueError('complete permutation required')
    return a.astype(int,copy=False)

def fold_ids(frame):
    if frame.N%2:raise ValueError('equal fixed frame folds need even N')
    ranked=sorted(range(frame.N),key=lambda i:(hashlib.sha256(f'P2_COMPARE_FRAME_FOLDS_V01|{frame.frame_id}|{i}'.encode()).digest(),i))
    return (np.array(sorted(ranked[:frame.N//2])),np.array(sorted(ranked[frame.N//2:])) )

def trace_summary(tr,global_ids):
    return {'sample_ids':[int(global_ids[i]) for i in tr.order[:tr.budget]],
      'fits':[{'before_draw':int(f.before_draw),'fit_ids':[int(global_ids[i]) for i in f.fit_ids],
               'prediction_sha256':digest_array(f.predictions)} for f in tr.forecasts]}

def interval_row(lo,hi,status,guarantee,capacity):
    if status!='OK':return {'lower':None,'upper':None,'status':status,'raw_width':None,'scored_width':float(capacity),'guarantee':guarantee}
    lo=float(lo);hi=float(hi)
    if not math.isfinite(lo+hi) or lo>hi:raise ArithmeticError('invalid interval')
    return {'lower':lo,'upper':hi,'status':'OK','raw_width':hi-lo,'scored_width':hi-lo,'guarantee':guarantee}

def betting(tr,capacity,alpha=.05,offset=0):
    ci=refined_endpoint_interval(tr,alpha=alpha,strategy='predmix_099',guard=1)
    return interval_row(None if ci.lower is None else ci.lower+offset,None if ci.upper is None else ci.upper+offset,ci.status,'FINITE_SAMPLE_BOUNDED',capacity)

def combine_intervals(parts,capacity):
    if any(p['status']!='OK' for p in parts):return interval_row(None,None,'EMPTY','FINITE_SAMPLE_BOUNDED',capacity)
    return interval_row(sum(p['lower'] for p in parts),sum(p['upper'] for p in parts),'OK','FINITE_SAMPLE_BOUNDED',capacity)

def wald(estimate,variance,capacity):
    if not math.isfinite(variance) or variance<0:raise ArithmeticError('variance')
    radius=norm.ppf(.975)*math.sqrt(variance)
    lo=max(0.,estimate-radius);hi=min(float(capacity),estimate+radius)
    row=interval_row(lo if lo<=hi else None,hi if lo<=hi else None,'OK' if lo<=hi else 'EMPTY','APPROXIMATE_WALD',capacity)
    row['estimated_variance']=float(variance);row['zero_estimated_variance']=bool(variance==0)
    return row

def residual_variance(residual,M):
    n=len(residual)
    if n==M:return 0.
    if n<2:raise ValueError('noncensus Wald interval needs two observations')
    return float(M*M*(1-n/M)*np.var(residual,ddof=1)/n)

def power_part(q,y,M):
    n=len(y)
    if n==M:return float(sum(y)),0.,{'lambda':0.,'census':True}
    if n<2:raise ValueError('power interval requires two observations')
    # q is (full-frame proxy, sampled proxy), not unknown outcomes.
    full,qs=q
    centered=qs-qs.mean();den=float(centered@centered)
    lam=0. if den<1e-12 else float(np.clip(centered@(y-y.mean())/den,0.,1.))
    est=M*float(y.mean())+lam*(float(full.sum())-M*float(qs.mean()))
    var=residual_variance(y-lam*qs,M)
    return est,var,{'lambda':lam,'degenerate_proxy':den<1e-12,'denominator':den}

class ActiveCS(RefinedCS):
    """Reuse tested wealth/inversion, with the active conditional-mean mapping."""
    def __init__(self,x,center,scale,observed,N,K,alpha=.05):
        self.n=len(x);self.x=np.asarray(x);self.center=np.asarray(center);self.scale=np.asarray(scale)
        if np.any(self.scale<=0) or not np.all(np.isfinite(self.scale)):raise ArithmeticError('active support')
        if not np.all(np.isfinite(self.x)) or np.any(self.x<-1e-10) or np.any(self.x>1+1e-10):raise ArithmeticError('active normalized observation')
        self.x=np.clip(self.x,0,1);self.active=np.ones(self.n,dtype=bool)
        self.alpha=alpha;self.guard=1;self.evaluations=0;self.refinement='predmix_099'
        self.lo=int(sum(observed));self.hi=self.lo+(N-self.n)*K
        self.bets=predictable_bets(self.x,alpha,self.n)

def acquisition(mu,c,remaining,K,lam):
    a=np.sqrt(mu[remaining]*(1-mu[remaining]/K)+(mu[remaining]-lam*c[remaining])**2)
    if not np.all(np.isfinite(a)) or np.any(a<0):raise ArithmeticError('active scores')
    M=len(remaining);total=float(a.sum())
    p=np.full(M,1/M) if total==0 else .9*a/total+.1/M
    err=float(p.sum()-1)
    if not np.all(np.isfinite(p)) or np.any(p<=0) or abs(err)>1e-12:raise ArithmeticError('propensities')
    p=p/p.sum()
    return p,err

def active(frame,obs,budget,uniforms,lam,*,forced_order=None):
    N=frame.N;K=frame.K;remaining=np.arange(N);chosen=[];values=[];records=[];fits=[]
    c=frame.q-frame.q.mean();forecast=make_forecaster(frame.X,frame.q,K)
    z=[];lure=[];xx=[];centers=[];scales=[];past_r=0.;uniforms=np.asarray(uniforms)
    if len(uniforms)<budget or np.any(uniforms[:budget]<0) or np.any(uniforms[:budget]>=1):raise ValueError('uniform stream')
    for t in range(budget):
        if t==0 or t in UPDATES:
            mu=forecast(np.array(chosen,dtype=int),np.array(values))
            fits.append({'before_draw':t,'fit_ids':chosen.copy(),'prediction_sha256':digest_array(mu)})
        p,err=acquisition(mu,c,remaining,K,lam)
        if forced_order is None:
            k=int(np.searchsorted(np.cumsum(p),uniforms[t],side='right'));k=min(k,len(remaining)-1)
        else:
            hits=np.flatnonzero(remaining==forced_order[t])
            if len(hits)!=1:raise ValueError('forced path')
            k=int(hits[0])
        i=int(remaining[k]);prob=float(p[k]);y=obs.get(i);r=y-lam*c[i]
        a=float(np.min(-lam*c[remaining]/p));d=float(np.max((K-lam*c[remaining])/p)-a)
        xx.append((r/prob-a)/d);centers.append(past_r+a);scales.append(d)
        z.append(past_r+r/prob)
        if budget<N:lure.append((N/budget)*(1+(N-budget)/(N-t-1)*(1/(len(remaining)*prob)-1))*r)
        records.append({'id':i,'p':prob,'probabilities_sha256':digest_array(p),'normalization_error':err})
        chosen.append(i);values.append(y);past_r+=r;remaining=np.delete(remaining,k)
    if budget==N:
        point=float(sum(values));ci=interval_row(point,point,'OK','FINITE_SAMPLE_BOUNDED',N*K)
    else:
        M=N-np.arange(budget);w=N*(N-budget)/(budget*M*(M-1))
        point=float(w@z)
        if abs(point-math.fsum(lure))>1e-8*max(1,abs(point)):raise ArithmeticError('active LURE identity')
        cs=ActiveCS(xx,centers,scales,values,N,K);bounds=cs.bounds()
        ci=interval_row(bounds.lower,bounds.upper,bounds.status,'FINITE_SAMPLE_BOUNDED',N*K)
    return point,ci,{'draws':records,'fits':fits,'lambda':lam,'conditional_mean_centers':centers,'conditional_mean_scales':scales,'normalized_observations':xx}

def run_method(frame,budget,method,get_measurement,order,fold_orders,uniforms):
    if method not in METHODS:raise ValueError('method')
    N=frame.N;K=frame.K;capacity=N*K
    if not isinstance(budget,int) or not 0<budget<=N:raise ValueError('budget')
    order=validate_order(order,N);obs=Measurements(get_measurement,N,K);trace={};intervals={};started=time.perf_counter()
    def build(ids,local_order,n,predict,updates=()):
        return build_trace(np.full(len(ids),K),local_order,n,lambda j:(obs.get(ids[j]),obs.get(ids[j])),predict,updates)
    if method in ('direct_HT','raw_difference','seq_detector_100'):
        predict=make_forecaster(frame.X,frame.q,K) if method=='seq_detector_100' else lambda ids,y:np.zeros(N) if method=='direct_HT' else frame.q.copy()
        tr=build(np.arange(N),order,budget,predict,UPDATES if method=='seq_detector_100' else ())
        point=tr.estimate;intervals[method]=betting(tr,capacity);trace=trace_summary(tr,np.arange(N))
        if method in WALD:
            pred=np.zeros(N) if method=='direct_HT' else frame.q
            residual=obs.array(order[:budget])-pred[order[:budget]]
            intervals[method+'_wald']=wald(point,residual_variance(residual,N),capacity)
    elif method.startswith('split_'):
        p=budget//5
        if p<1:raise ValueError('split pilot cannot be empty')
        P=order[:p];R=order[p:];n=budget-p;y=obs.array(P)
        m=ridge_predictions(frame.features,P,y,K,10) if method=='split_ridge10' else offset_ridge(frame.X,frame.q,P,y,100,K)
        tr=build(R,np.arange(len(R)),n,lambda ids,values:m[R])
        offset=float(y.sum());point=offset+tr.estimate;intervals[method]=betting(tr,capacity,offset=offset)
        trace=trace_summary(tr,R);trace['pilot_ids']=P.tolist();trace['pilot_prediction_sha256']=digest_array(m)
        if method in WALD:intervals[method+'_wald']=wald(point,residual_variance(obs.array(R[:n])-m[R[:n]],len(R)),capacity)
    elif method in ('stratified_HT','ff_strat_power'):
        groups=[np.flatnonzero(frame.strata==h) for h in range(3)];alloc=allocate_exact([len(g) for g in groups],budget)
        H=sum(bool(len(g)) for g in groups);point=variance=0.;parts=[];trace={'strata':[],'allocation':alloc}
        for h,g in enumerate(groups):
            if not len(g):continue
            chosen=order[frame.strata[order]==h];n=alloc[h]
            if method=='stratified_HT':
                mapping={int(v):j for j,v in enumerate(g)};local=np.array([mapping[int(i)] for i in chosen])
                tr=build(g,local,n,lambda ids,y,nn=len(g):np.zeros(nn));point+=tr.estimate
                parts.append(betting(tr,capacity,alpha=.05/H));trace['strata'].append(trace_summary(tr,g))
            else:
                selected=chosen[:n];y=obs.array(selected);est,var,detail=power_part((frame.q[g],frame.q[selected]),y,len(g))
                point+=est;variance+=var;detail.update(stratum=h,sample_ids=selected.tolist());trace['strata'].append(detail)
        intervals[method]=combine_intervals(parts,capacity) if method=='stratified_HT' else wald(point,variance,capacity)
    elif method=='ff_power':
        S=order[:budget];y=obs.array(S);point,variance,trace=power_part((frame.q,frame.q[S]),y,N)
        intervals[method]=wald(point,variance,capacity)
    elif method.startswith('frame_cf2_'):
        if budget%2:raise ValueError('fold budget must be even')
        folds=fold_ids(frame);n=budget//2;selected=[];local_orders=[]
        for k,g in enumerate(folds):
            local=validate_order(fold_orders[k],len(g));local_orders.append(local);selected.append(g[local[:n]])
            obs.array(selected[-1])
        point=0.;parts=[];trace={'folds':[]}
        for k,g in enumerate(folds):
            train=selected[1-k]
            m=np.zeros(N) if method=='frame_cf2_direct' else offset_ridge(frame.X,frame.q,train,obs.array(train),100,K)
            tr=build(g,local_orders[k],n,lambda ids,y,mm=m,gg=g:mm[gg]);point+=tr.estimate
            parts.append(betting(tr,capacity,alpha=.025));record=trace_summary(tr,g)
            record.update(fold_ids=g.tolist(),training_ids=[] if method=='frame_cf2_direct' else train.tolist(),prediction_sha256=digest_array(m[g]))
            trace['folds'].append(record)
        intervals[method]=combine_intervals(parts,capacity)
    else:
        lam={'active_count_lure':0.,'ppat_count_half':.5,'ppat_count_one':1.}[method]
        point,ci,trace=active(frame,obs,budget,uniforms,lam);intervals[method]=ci
    if len(obs.ids)!=budget:raise AssertionError('distinct label budget')
    if not math.isfinite(point):raise ArithmeticError('nonfinite point')
    trace['measurement_ids']=obs.ids;trace['measurement_values']=[obs.values[i] for i in obs.ids]
    return {'method':method,'budget':budget,'estimate':float(point),'labels':budget,'trace':trace,'intervals':intervals,'elapsed_seconds':time.perf_counter()-started}
