"""Common, reviewable evaluation of the six retained methods.

Evaluator truth and public predictors are separate arguments/interfaces. Only
sampled measurement pairs are delivered to build_trace. Full truth is used
afterwards for scoring. This is not an operating-system secrecy mechanism.
"""
import math
import numpy as np
from paper2_core_v01 import Bounds,allocate_strata,exact_design_variance,ridge_predictions
from paper2_sequential_v01 import build_trace,predictable_risk,normalized_risk_support
from paper2_betting_refinements_v01 import RefinedCS,refined_endpoint_interval
from paper2_candidate_v01 import make_forecaster,UPDATES
from run_development_extension_v01 import drive_features,offset_ridge

METHODS=('direct_HT','stratified_HT','raw_difference','split_ridge10','split_offset100','seq_detector_100')

def evaluate(public,truth,order,budget,method,intervals=True):
    if method not in METHODS:raise ValueError('method')
    N=len(public.caps);K=int(public.caps[0])
    if not np.all(public.caps==K):raise ValueError('evaluation expects the frozen global cap')
    order=np.asarray(order)
    if order.ndim!=1 or len(order)!=N or not np.issubdtype(order.dtype,np.integer) or not np.array_equal(np.sort(order),np.arange(N)):
        raise ValueError('complete distinct integer sampling permutation required before any partition')
    if not isinstance(budget,int) or not 0<budget<=N:raise ValueError('budget')
    X=drive_features(public);q=np.clip(public.unmatched,0,K)
    exact=np.array_equal(truth.lower,truth.upper) and np.array_equal(truth.lower,truth.omitted)
    # Exactness is a property of a constructed population, not a human judgement.
    B=int(sum(truth.omitted));mid=(truth.lower+truth.upper)/2;parts=[]
    offsetlow=offsetup=0;alpha=.05;point=risk=0.
    if method=='stratified_HT':
        strata=np.minimum(public.unmatched,2);groups=[np.flatnonzero(strata==h) for h in range(3)]
        sizes=[len(g) for g in groups];alloc=allocate_strata(sizes,budget);alpha=.05/sum(s>0 for s in sizes)
        for h,g in enumerate(groups):
            if not len(g):continue
            lookup={int(v):j for j,v in enumerate(g)};local=np.array([lookup[int(v)] for v in order if strata[v]==h]);bb=int(alloc[h])
            tr=build_trace(public.caps[g],local,bb,lambda i,gg=g:(truth.lower[gg[i]],truth.upper[gg[i]]),lambda ids,values,nn=len(g):np.zeros(nn),updates=())
            parts.append((tr,int(sum(truth.lower[g])),int(sum(truth.upper[g]))));point+=tr.estimate
            if exact:risk+=exact_design_variance(truth.omitted[g],np.zeros(len(g)),bb)
    elif method.startswith('split_'):
        p=budget//5;n=budget-p
        if p<1:raise ValueError('split pilot cannot be empty')
        P=order[:p];R=order[p:]
        m=ridge_predictions(public.features,P,mid[P],K,10) if method=='split_ridge10' else offset_ridge(X,q,P,mid[P],100,K)
        tr=build_trace(public.caps[R],np.arange(len(R)),n,lambda i:(truth.lower[R[i]],truth.upper[R[i]]),lambda ids,y:m[R],updates=())
        parts=[(tr,int(sum(truth.lower[R])),int(sum(truth.upper[R])))];offsetlow=int(sum(truth.lower[P]));offsetup=int(sum(truth.upper[P]))
        point=float(sum(mid[P])+tr.estimate)
        if exact:risk=exact_design_variance(truth.omitted[R],m[R],n)
    else:
        if method=='seq_detector_100':predict=make_forecaster(X,q,K);updates=UPDATES
        else:predict=lambda ids,y:np.zeros(N) if method=='direct_HT' else q.copy();updates=()
        tr=build_trace(public.caps,order,budget,lambda i:(truth.lower[i],truth.upper[i]),predict,updates=updates)
        parts=[(tr,int(sum(truth.lower)),int(sum(truth.upper)))];point=tr.estimate
        if exact:
            if method=='seq_detector_100':risk=predictable_risk(tr,truth.omitted)
            else:risk=exact_design_variance(truth.omitted,np.zeros(N) if method=='direct_HT' else q,budget)
    if not exact:risk=None
    if method=='seq_detector_100' and exact and risk/(N*K)**2>normalized_risk_support(N,budget)+1e-12:
        raise ArithmeticError('sequential risk support exceeded')
    row={'method':method,'budget':budget,'truth':B,'estimate':float(point),'squared_error':float((point-B)**2),
         'risk_score':risk,'exact_counts':bool(exact),'status':'OK'}
    if not intervals:return row
    low=offsetlow;up=offsetup;status='OK';certificate=True;threshold=math.log(2/alpha)+1e-10
    for tr,L,U in parts:
        ci=refined_endpoint_interval(tr,alpha=alpha,strategy='predmix_099',guard=1)
        if ci.status!='OK':status=ci.status
        else:low+=ci.lower;up+=ci.upper
        a=RefinedCS(tr,'lower',alpha,'predmix_099');b=RefinedCS(tr,'upper',alpha,'predmix_099')
        certificate=certificate and a.max_log_wealth(L)[0]<=threshold and b.max_log_wealth(U)[1]<=threshold
    ci=Bounds(low,up,status,0) if status=='OK' else Bounds(None,None,status,0)
    row.update(status=ci.status,lower=ci.lower,upper=ci.upper,final_covers=ci.covers(B),all_prefix_certificate=bool(certificate),width=ci.scored_width(N*K))
    return row

def exact_baseline_risks(public,truth,budget):
    N=len(public.caps);y=truth.omitted
    direct=exact_design_variance(y,np.zeros(N),budget)
    strata=np.minimum(public.unmatched,2);groups=[np.flatnonzero(strata==h) for h in range(3)]
    alloc=allocate_strata([len(g) for g in groups],budget)
    stratified=sum(exact_design_variance(y[g],np.zeros(len(g)),int(alloc[h])) for h,g in enumerate(groups) if len(g))
    return {'direct_HT':direct,'stratified_HT':stratified}
