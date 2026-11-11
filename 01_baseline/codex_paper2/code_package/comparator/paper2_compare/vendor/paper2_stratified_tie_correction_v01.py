"""Post-run correction of stable largest-remainder ties; frozen source untouched.

The allocation policy is unchanged. Integer remainders implement its specified
stable tie order exactly, avoiding rounding differences between equal fractions.
"""
import math
import numpy as np
from paper2_sequential_v01 import build_trace
from paper2_betting_refinements_v01 import RefinedCS, refined_endpoint_interval
from paper2_core_v01 import Bounds

def allocate_exact(sizes,budget):
    sizes=list(map(int,sizes))
    if any(s<0 for s in sizes) or not 0<budget<=sum(sizes):raise ValueError('sizes/budget')
    out=[min(s,2) for s in sizes];left=budget-sum(out)
    if left<0:raise ValueError('minimum allocation')
    cap=[s-a for s,a in zip(sizes,out)]
    if left:
        denominator=sum(cap);num=[left*c for c in cap]
        base=[n//denominator for n in num];extra=left-sum(base)
        out=[a+b for a,b in zip(out,base)]
        for h in sorted(range(len(sizes)),key=lambda j:(-(num[j]%denominator),j))[:extra]:out[h]+=1
    return out

def evaluate_stratified(public,truth,order,budget):
    N=len(public.caps);K=int(public.caps[0]);order=np.asarray(order)
    if not np.all(public.caps==K) or order.ndim!=1 or not np.issubdtype(order.dtype,np.integer) or not np.array_equal(np.sort(order),np.arange(N)):raise ValueError('frozen support/permutation required')
    strata=np.minimum(public.unmatched,2);groups=[np.flatnonzero(strata==h) for h in range(3)]
    sizes=[len(g) for g in groups];alloc=allocate_exact(sizes,budget)
    alpha=.05/sum(s>0 for s in sizes);threshold=math.log(2/alpha)+1e-10
    exact=np.array_equal(truth.lower,truth.upper) and np.array_equal(truth.lower,truth.omitted)
    point=0.;risk=0.;low=0;up=0;status='OK';certificate=True
    for h,g in enumerate(groups):
        if not len(g):continue
        lookup={int(v):j for j,v in enumerate(g)};local=np.array([lookup[int(v)] for v in order if strata[v]==h]);n=alloc[h]
        tr=build_trace(public.caps[g],local,n,lambda i,gg=g:(truth.lower[gg[i]],truth.upper[gg[i]]),lambda ids,values,nn=len(g):np.zeros(nn),updates=())
        point+=tr.estimate
        if exact and n<len(g):
            y=truth.omitted[g];mean=math.fsum(map(float,y))/len(y)
            risk+=len(g)*(len(g)-n)*math.fsum((float(z)-mean)**2 for z in y)/(n*(len(g)-1))
        ci=refined_endpoint_interval(tr,alpha=alpha,strategy='predmix_099',guard=1)
        if ci.status!='OK':status=ci.status
        else:low+=ci.lower;up+=ci.upper
        a=RefinedCS(tr,'lower',alpha,'predmix_099');b=RefinedCS(tr,'upper',alpha,'predmix_099')
        certificate=certificate and a.max_log_wealth(int(sum(truth.lower[g])))[0]<=threshold and b.max_log_wealth(int(sum(truth.upper[g])))[1]<=threshold
    B=int(sum(truth.omitted));ci=Bounds(low,up,status,0) if status=='OK' else Bounds(None,None,status,0)
    return {'method':'stratified_HT','budget':budget,'truth':B,'estimate':float(point),'squared_error':float((point-B)**2),'risk_score':risk if exact else None,'exact_counts':bool(exact),'status':ci.status,'lower':ci.lower,'upper':ci.upper,'final_covers':ci.covers(B),'all_prefix_certificate':bool(certificate),'width':ci.scored_width(N*K)}
