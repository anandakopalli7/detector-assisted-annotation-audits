"""AM-P2-01 sequential audit: forecasts use past measurements only.

No file/network access. Construction truth is accepted only by a separate
evaluation function after the measurement/forecast trace has been constructed.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp
from paper2_core_v01 import Bounds,GRID,predictable_bets

@dataclass(frozen=True)
class Forecast:
    before_draw: int
    fit_ids: tuple
    predictions: np.ndarray

@dataclass(frozen=True)
class Trace:
    caps: np.ndarray
    order: np.ndarray
    budget: int
    lower: np.ndarray
    upper: np.ndarray
    forecasts: tuple
    forecast_index: np.ndarray
    remaining: np.ndarray
    prediction_sum: np.ndarray
    prediction_selected: np.ndarray
    a: np.ndarray
    d: np.ndarray
    weights: np.ndarray
    estimate: float

def _vector(x,name):
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or not np.all(np.isfinite(a)):raise ValueError(name+' must be a finite vector')
    return a

def build_trace(caps,order,budget,get_measurement,predict,updates=(20,40,80,120,160)):
    k=_vector(caps,'caps');N=len(k)
    if not N or np.any(k<0) or np.any(k!=np.floor(k)):raise ValueError('integer nonnegative support required')
    order0=np.asarray(order)
    if order0.ndim!=1 or len(order0)!=N or np.any(order0!=order0.astype(int)):raise ValueError('complete integer sampling order required')
    order=order0.astype(int)
    if sorted(order.tolist())!=list(range(N)):raise ValueError('order must be a permutation')
    if not isinstance(budget,int) or not 0<budget<=N:raise ValueError('budget')
    updates=tuple(updates)
    if any(not isinstance(t,int) or t<=0 for t in updates) or len(set(updates))!=len(updates):raise ValueError('update schedule')
    lo=[];up=[];forecasts=[];which=[];ps=[];ms=[];aa=[];dd=[]
    schedule={0,*updates};selected=[]
    for t in range(budget):
        # The predictor is not given order[t], future IDs, or future outcomes.
        if t in schedule:
            m=_vector(predict(np.array(selected,dtype=int), (np.array(lo)+np.array(up))/2),'predictions')
            if len(m)!=N or np.any(m<0) or np.any(m>k):raise ValueError('predictions outside declared clipping rule')
            m=m.copy();m.flags.writeable=False
            forecasts.append(Forecast(t,tuple(selected),m))
            ordered=m[order]
            sums=np.r_[np.cumsum(ordered[::-1])[::-1],0.]
            mins=np.minimum.accumulate((-ordered)[::-1])[::-1]
            maxs=np.maximum.accumulate((k[order]-ordered)[::-1])[::-1]
        which.append(len(forecasts)-1);ps.append(float(sums[t]));aa.append(float(mins[t]));dd.append(float(maxs[t]-mins[t]))
        i=int(order[t]);ms.append(float(m[i]))
        pair=get_measurement(i)
        if len(pair)!=2:raise ValueError('measurement must supply lower and upper')
        l,u=map(float,pair)
        if not math.isfinite(l+u) or l!=math.floor(l) or u!=math.floor(u) or not 0<=l<=u<=k[i]:raise ValueError('invalid measurement endpoints')
        lo.append(l);up.append(u);selected.append(i)
    lower=np.array(lo);upper=np.array(up);mid=(lower+upper)/2;M=N-np.arange(budget)
    if budget==N:weights=np.zeros(budget);estimate=float(sum(mid))
    else:
        weights=N*(N-budget)/(budget*M*(M-1));past=np.r_[0.,np.cumsum(mid)[:-1]]
        estimates=past+np.array(ps)+M*(mid-np.array(ms));estimate=float(weights@estimates)
        if abs(sum(weights)-1)>1e-12:raise ArithmeticError('weights do not sum to one')
    return Trace(k.copy(),order.copy(),budget,lower,upper,tuple(forecasts),np.array(which),M,np.array(ps),np.array(ms),np.array(aa),np.array(dd),weights,estimate)

def predictable_risk(trace,truth):
    """Evaluator-only risk score; expectation over paths equals MSE for exact counts."""
    y=_vector(truth,'evaluation truth');N=len(trace.caps)
    if len(y)!=N or np.any(y<0) or np.any(y>trace.caps):raise ValueError('truth support')
    if not np.array_equal(trace.lower,trace.upper) or not np.array_equal(trace.lower,y[trace.order[:trace.budget]]):raise ValueError('risk identity requires exact measurements')
    if trace.budget==N:return 0.
    score=0.
    for t in range(trace.budget):
        R=trace.order[t:];m=trace.forecasts[trace.forecast_index[t]].predictions
        score+=trace.weights[t]**2*len(R)**2*np.var(y[R]-m[R],ddof=0)
    return float(score)

def normalized_risk_support(N,b):
    """Residual range <=2K when both truth and forecasts lie in [0,K]."""
    if not isinstance(N,int) or not isinstance(b,int) or not 0<b<=N:raise ValueError('sizes')
    if b==N:return 0.
    return float((N-b)**2/b**2*sum(1/(N-t-1)**2 for t in range(b)))

class SequentialCS:
    """Confidence inversion using each draw's predictable residual normalization."""
    def __init__(self,trace,endpoint='lower',alpha=.05,strategy='predmix',prefix=None,guard=1):
        if endpoint not in ('lower','upper') or strategy not in ('predmix','grid'):raise ValueError('inference mode')
        if not 0<alpha<1 or not isinstance(guard,int) or guard<0:raise ValueError('inference settings')
        n=trace.budget if prefix is None else prefix
        if not isinstance(n,int) or not 0<=n<=trace.budget:raise ValueError('prefix')
        self.trace=trace;self.n=n;self.alpha=alpha;self.strategy=strategy;self.guard=guard;self.evaluations=0
        self.y=getattr(trace,endpoint)[:n];self.M=trace.remaining[:n];self.d=trace.d[:n]
        self.active=self.d>0;past=np.r_[0.,np.cumsum(self.y)[:-1]] if n else np.array([])
        self.center=past+trace.prediction_sum[:n]+self.M*trace.a[:n]
        self.scale=self.M*self.d
        self.x=np.zeros(n)
        self.x[self.active]=(self.y[self.active]-trace.prediction_selected[:n][self.active]-trace.a[:n][self.active])/self.d[self.active]
        if np.any(self.x<-1e-10) or np.any(self.x>1+1e-10):raise ArithmeticError('normalized endpoint outside support')
        self.x=np.clip(self.x,0,1)
        self.lo=int(sum(self.y));self.hi=self.lo+int(sum(trace.caps[trace.order[n:]]))
        self.bets=predictable_bets(self.x,alpha,trace.budget) if strategy=='predmix' else None

    def max_log_wealth(self,total):
        self.evaluations+=1
        if not self.n:return (0.,0.)
        q=np.zeros(self.n);q[self.active]=(total-self.center[self.active])/self.scale[self.active]
        if np.any(q < -1e-9) or np.any(q>1+1e-9):raise ArithmeticError('candidate incompatible with residual support')
        q=np.clip(q,0,1);tiny=1e-14
        impossible=self.active&(((q<=tiny)&(self.x>1e-10))|((q>=1-tiny)&(self.x<1-1e-10)))
        if np.any(impossible):raise ArithmeticError('near-boundary ambiguity needs conservative fallback')
        interior=self.active&(q>tiny)&(q<1-tiny)
        if self.strategy=='grid':
            fp=np.ones((self.n,len(GRID)));fn=fp.copy()
            fp[interior]=1-GRID+self.x[interior,None]/q[interior,None]*GRID
            fn[interior]=1-GRID+(1-self.x[interior,None])/(1-q[interior,None])*GRID
            lp=logsumexp(np.cumsum(np.log(fp),axis=0),axis=1)-math.log(len(GRID))
            ln=logsumexp(np.cumsum(np.log(fn),axis=0),axis=1)-math.log(len(GRID))
        else:
            fp=np.ones(self.n);fn=fp.copy()
            fp[interior]=1+np.minimum(self.bets[interior],.5/q[interior])*(self.x[interior]-q[interior])
            fn[interior]=1-np.minimum(self.bets[interior],.5/(1-q[interior]))*(self.x[interior]-q[interior])
            lp=np.cumsum(np.log(fp));ln=np.cumsum(np.log(fn))
        return float(np.max(lp,initial=0)),float(np.max(ln,initial=0))

    def bounds(self,tail='both'):
        if tail not in ('lower','upper','both'):raise ValueError('tail')
        if not self.n:return Bounds(self.lo,self.hi,'OK',0)
        threshold=math.log(2/self.alpha)+1e-10
        def reject(total,side):return self.max_log_wealth(total)[side]>threshold
        low,high=self.lo,self.hi
        if tail in ('lower','both'):
            if reject(high,0):return Bounds(None,None,'EMPTY',self.evaluations)
            left,right=self.lo,self.hi
            while left<right:
                mid=(left+right)//2
                if reject(mid,0):left=mid+1
                else:right=mid
            low=max(self.lo,left-self.guard)
        if tail in ('upper','both'):
            if reject(self.lo,1):return Bounds(None,None,'EMPTY',self.evaluations)
            left,right=self.lo,self.hi
            while left<right:
                mid=(left+right+1)//2
                if reject(mid,1):right=mid-1
                else:left=mid
            high=min(self.hi,left+self.guard)
        if low>high:return Bounds(None,None,'EMPTY',self.evaluations)
        return Bounds(low,high,'OK',self.evaluations)

def endpoint_interval(trace,alpha=.05,strategy='predmix',guard=1):
    if np.array_equal(trace.lower,trace.upper):return SequentialCS(trace,alpha=alpha,strategy=strategy,guard=guard).bounds()
    lo=SequentialCS(trace,'lower',alpha,strategy,guard=guard).bounds('lower')
    up=SequentialCS(trace,'upper',alpha,strategy,guard=guard).bounds('upper')
    if lo.status!=up.status or lo.status!='OK' or lo.lower>up.upper:return Bounds(None,None,'EMPTY',lo.evaluations+up.evaluations)
    return Bounds(lo.lower,up.upper,'OK',lo.evaluations+up.evaluations)
