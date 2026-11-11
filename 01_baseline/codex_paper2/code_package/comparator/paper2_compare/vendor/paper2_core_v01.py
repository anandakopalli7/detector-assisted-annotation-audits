"""Development inference kernels. No file access and no physical-bound inference."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp

GRID=np.array([1/16,1/8,1/4,1/2,3/4],dtype=float)

def _array(v,name):
    a=np.asarray(v,dtype=float)
    if a.ndim!=1 or not np.all(np.isfinite(a)):raise ValueError(name+' must be a finite vector')
    return a

def predictable_bets(x,alpha,horizon):
    """Predictable EB strategy: past-only regularized variance, fixed horizon."""
    x=_array(x,'normalized observations')
    t=np.arange(1,len(x)+1)
    mu=(.5+np.cumsum(x))/(t+1)
    var=(.25+np.cumsum((x-mu)**2))/(t+1)
    previous=np.r_[.25,var[:-1]]
    return np.sqrt(2*np.log(2/alpha)/(horizon*previous))

@dataclass(frozen=True)
class Bounds:
    lower: int|None
    upper: int|None
    status: str
    evaluations: int
    def covers(self,b):return self.status=='OK' and self.lower<=b<=self.upper
    def scored_width(self,capacity):
        if self.status!='OK':return float(capacity)
        return float(self.upper-self.lower)

class ResidualCS:
    """Running-intersection integer confidence bounds by monotone tail inversion.

    caps/predictions describe the whole fixed frame. ids/observations describe
    only the ordered random audit. Numeric boundary guard expands endpoints.
    A justified support and correct randomization are caller responsibilities.
    """
    def __init__(self,caps,predictions,ids,observations,alpha=.05,strategy='predmix',guard=1,planned_horizon=None):
        if caps is None:raise ValueError('Known support required')
        self.k=_array(caps,'caps');self.m=_array(predictions,'predictions')
        if not len(self.k) or len(self.k)!=len(self.m) or np.any(self.k<0) or np.any(self.k!=np.floor(self.k)):raise ValueError('caps')
        ids0=np.asarray(ids)
        if ids0.ndim!=1 or np.any(ids0!=ids0.astype(int)):raise ValueError('IDs must be integers')
        self.ids=ids0.astype(int);self.y=_array(observations,'observations')
        if len(self.y)!=len(self.ids) or len(set(self.ids))!=len(self.ids) or np.any(self.ids<0) or np.any(self.ids>=len(self.k)):raise ValueError('invalid sample IDs')
        if np.any(self.y<0) or np.any(self.y>self.k[self.ids]) or np.any(self.y!=np.floor(self.y)):raise ValueError('observed counts violate cap/integer rule')
        if not 0<alpha<1 or strategy not in ('grid','predmix') or not isinstance(guard,int) or guard<0:raise ValueError('inference setting')
        self.N=len(self.k);self.n=len(self.ids);self.alpha=alpha;self.strategy=strategy;self.guard=guard;self.evaluations=0
        self.horizon=max(1,self.n) if planned_horizon is None else planned_horizon
        if not isinstance(self.horizon,int) or self.horizon<self.n or self.horizon>self.N:raise ValueError('planned horizon')
        self.a=float(np.min(-self.m));self.d=float(np.max(self.k-self.m)-self.a)
        self.pred_sum=math.fsum(self.m)
        self.lo=int(np.sum(self.y));self.hi=self.lo+int(np.sum(self.k)-np.sum(self.k[self.ids]))
        if self.d>0:
            self.x=(self.y-self.m[self.ids]-self.a)/self.d
            if np.any(self.x < -1e-12) or np.any(self.x>1+1e-12):raise ArithmeticError('normalized observation out of support')
            self.x=np.clip(self.x,0,1)
            self.past=np.r_[0.,np.cumsum(self.x)[:-1]] if self.n else np.array([])
            self.remaining=self.N-np.arange(self.n)
            self.bets=predictable_bets(self.x,alpha,self.horizon) if strategy=='predmix' else None

    def max_log_wealth(self,total):
        self.evaluations+=1
        if not self.n or self.d==0:return (0.,0.)
        q=((total-self.pred_sum-self.N*self.a)/self.d-self.past)/self.remaining
        # Domains supplied by inversion are within the current logical count support.
        if np.any(q < -1e-9) or np.any(q>1+1e-9):raise ArithmeticError('candidate outside transformed support')
        q=np.clip(q,0,1)
        tiny=1e-14
        impossible=((q<=tiny)&(self.x>1e-10))|((q>=1-tiny)&(self.x<1-1e-10))
        if np.any(impossible):
            # This should occur only outside valid current support; fail conservatively.
            raise ArithmeticError('numerically impossible candidate needs exact fallback')
        interior=(q>tiny)&(q<1-tiny)
        if self.strategy=='grid':
            fp=np.ones((self.n,len(GRID)));fn=fp.copy()
            fp[interior]=(1-GRID)+self.x[interior,None]/q[interior,None]*GRID
            fn[interior]=(1-GRID)+(1-self.x[interior,None])/(1-q[interior,None])*GRID
            lp=logsumexp(np.cumsum(np.log(fp),axis=0),axis=1)-np.log(len(GRID))
            ln=logsumexp(np.cumsum(np.log(fn),axis=0),axis=1)-np.log(len(GRID))
        else:
            fp=np.ones(self.n);fn=fp.copy()
            positive=np.minimum(self.bets[interior],.5/q[interior])
            negative=np.minimum(self.bets[interior],.5/(1-q[interior]))
            fp[interior]=1+positive*(self.x[interior]-q[interior])
            fn[interior]=1-negative*(self.x[interior]-q[interior])
            lp=np.cumsum(np.log(fp));ln=np.cumsum(np.log(fn))
        return float(np.max(lp,initial=0)),float(np.max(ln,initial=0))

    def bounds(self,tail='both'):
        if tail not in ('both','lower','upper'):raise ValueError('tail')
        if self.d==0:return Bounds(0,0,'OK',0)
        if not self.n:return Bounds(self.lo,self.hi,'OK',0)
        threshold=math.log(2/self.alpha)+1e-10
        def rej(b,side):return self.max_log_wealth(b)[side]>threshold
        lo,hi=self.lo,self.hi
        if tail in ('both','lower'):
            if rej(hi,0):return Bounds(None,None,'EMPTY',self.evaluations)
            left,right=lo,hi
            while left<right:
                mid=(left+right)//2
                if rej(mid,0):left=mid+1
                else:right=mid
            lo=max(self.lo,left-self.guard)
        if tail in ('both','upper'):
            if rej(self.lo,1):return Bounds(None,None,'EMPTY',self.evaluations)
            left,right=self.lo,self.hi
            while left<right:
                mid=(left+right+1)//2
                if rej(mid,1):right=mid-1
                else:left=mid
            hi=min(self.hi,left+self.guard)
        if lo>hi:return Bounds(None,None,'EMPTY',self.evaluations)
        return Bounds(lo,hi,'OK',self.evaluations)

def difference_total(predictions,ids,observations):
    m=_array(predictions,'predictions');y=_array(observations,'observations');ids=np.asarray(ids,dtype=int)
    if not len(ids) or len(ids)!=len(y) or len(set(ids))!=len(ids):raise ValueError('sample')
    return float(np.sum(m)+len(m)/len(ids)*np.sum(y-m[ids]))

def exact_design_variance(counts,predictions,n):
    y=_array(counts,'counts');m=_array(predictions,'predictions');N=len(y)
    if len(m)!=N or not 0<n<=N:raise ValueError('variance dimensions')
    if n==N:return 0.
    return float(N*N*(1-n/N)*np.var(y-m,ddof=1)/n)

def allocate_strata(sizes,budget):
    sizes=np.asarray(sizes,dtype=int)
    if np.any(sizes<0) or not 0<budget<=sum(sizes):raise ValueError('stratum budget')
    allocation=np.minimum(sizes,2)
    if sum(allocation)>budget:raise ValueError('budget cannot cover each stratum')
    remaining=budget-int(sum(allocation));capacity=sizes-allocation
    if remaining:
        quota=remaining*capacity/sum(capacity);base=np.floor(quota).astype(int)
        allocation+=base;remaining-=int(sum(base))
        order=sorted(range(len(sizes)),key=lambda j:(-(quota[j]-base[j]),j))
        for j in order[:remaining]:allocation[j]+=1
    assert sum(allocation)==budget and np.all(allocation<=sizes)
    return allocation

def ridge_predictions(features,pilot,counts,cap,penalty=1.):
    X=np.asarray(features,dtype=float);P=np.asarray(pilot,dtype=int);y=_array(counts,'pilot counts')
    if len(P)!=len(y) or not len(P) or not np.all(np.isfinite(X)):raise ValueError('fit inputs')
    # Scaling uses observable feature values only, not hidden outcomes.
    center=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1
    Z=np.column_stack((np.ones(len(X)),(X-center)/scale))
    reg=np.eye(Z.shape[1])*penalty;reg[0,0]=0
    coef=np.linalg.solve(Z[P].T@Z[P]+reg,Z[P].T@y)
    return np.clip(Z@coef,0,cap)

def tree_predictions(features,pilot,counts,cap,depth=2,min_leaf=5):
    X=np.asarray(features,dtype=float);P=np.asarray(pilot,dtype=int);y=_array(counts,'pilot counts')
    if len(P)!=len(y) or not len(P):raise ValueError('fit inputs')
    result=np.empty(len(X))
    def fit(idx,targets,left_depth):
        value=float(y[idx].mean())
        if left_depth==0 or len(idx)<2*min_leaf:
            result[targets]=value;return
        best=None;parent=float(np.sum((y[idx]-value)**2))
        for feature in range(X.shape[1]):
            values=X[P[idx],feature]
            cuts=np.unique(np.quantile(values,[.2,.4,.6,.8]))
            for cut in cuts:
                left=values<=cut
                if min(int(left.sum()),int((~left).sum()))<min_leaf:continue
                loss=float(np.sum((y[idx[left]]-y[idx[left]].mean())**2)+np.sum((y[idx[~left]]-y[idx[~left]].mean())**2))
                if best is None or loss<best[0]-1e-12:best=(loss,feature,float(cut),left)
        if best is None or best[0]>=parent-1e-12:result[targets]=value;return
        _,feature,cut,left=best;route=X[targets,feature]<=cut
        fit(idx[left],targets[route],left_depth-1);fit(idx[~left],targets[~route],left_depth-1)
    fit(np.arange(len(P)),np.arange(len(X)),depth)
    return np.clip(result,0,cap)
