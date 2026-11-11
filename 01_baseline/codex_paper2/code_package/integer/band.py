"""D-P2-15 integer-band inference. No file, network or outcome acquisition.

Every rejection is exact: integer fixed-point enclosures plus Fraction fallback.
The existing predictor's represented float is the numerical prediction contract.
"""
from fractions import Fraction as F
from bisect import bisect_left
import math

def centers(predictions,caps):
    if len(predictions)!=len(caps):raise ValueError('prediction length')
    result=[]
    for m,k in zip(predictions,caps):
        if isinstance(m,bool):raise ValueError('boolean forecast')
        if isinstance(m,float):
            if not math.isfinite(m):raise ValueError('nonfinite forecast')
            q=F.from_float(m)
        elif isinstance(m,(int,F)):q=F(m)
        else:
            # numpy.float64 supports as_integer_ratio; never decimal-round.
            try:q=F(*m.as_integer_ratio())
            except Exception as e:raise ValueError('unsupported forecast') from e
        if not 0<=q<=k:raise ValueError('forecast outside cap')
        z=q+F(1,2);result.append(z.numerator//z.denominator)
    return tuple(result)

def prefix(capacities):
    cs=sorted((int(c) for c in capacities if c>0),reverse=True)
    sums=[0]
    for c in cs:sums.append(sums[-1]+c)
    return sums

def factor_pair(M,P,down,up,S,hit,relaxed=False):
    """Integer numerator/denominator, eta=3/4, equal unit draw probabilities."""
    if S==P:return (4 if hit else 1),4
    delta=abs(S-P);sums=down if S<P else up
    j=bisect_left(sums,delta)
    if not 1<=j<len(sums):raise ValueError('candidate outside remaining support')
    if relaxed:
        cap=sums[j]-sums[j-1]
        # Fractional-knapsack violation count j-1 + remaining/cap.
        vnum=(j-1)*cap+delta-sums[j-1]
        return (4 if hit else 1)*M*cap,4*M*cap-3*vnum
    return (4 if hit else 1)*M,4*M-3*j

class BandAudit:
    def __init__(self,caps,relaxed=False,bits=128):
        caps=tuple(caps)
        if not caps or any(isinstance(k,bool) or not isinstance(k,int) or k<0 for k in caps):
            raise ValueError('nonnegative integer caps required')
        if not isinstance(bits,int) or not 1<=bits<=1024:raise ValueError('precision')
        self.caps=caps;self.remaining=tuple(range(len(caps)));self.observed_total=0
        self.capacity=sum(caps);self.relaxed=bool(relaxed);self.scale=1<<bits
        self.lower=[self.scale]*(self.capacity+1);self.upper=self.lower.copy()
        self.max_lower=self.lower.copy();self.max_upper=self.upper.copy()
        self.pending=None;self.history=[];self.fallbacks=0;self.exact_cache={}

    def prepare(self,predictions):
        if self.pending is not None or not self.remaining:raise ValueError('preparation state')
        ks=tuple(self.caps[i] for i in self.remaining)
        c=centers(predictions,ks)
        self.pending=(c,prefix(c),prefix(k-m for k,m in zip(ks,c)))

    def observe(self,identity,value):
        if self.pending is None or identity not in self.remaining:raise ValueError('distinct precommitted draw required')
        if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=self.caps[identity]:
            raise ValueError('observation outside integer cap')
        c,down,up=self.pending;hit=value==c[self.remaining.index(identity)]
        M=len(self.remaining);P=sum(c);A=self.observed_total
        # Keep records needed for exact fallback, not an external outcome source.
        self.history.append((M,P,down,up,A,hit))
        for B in range(A,A+sum(self.caps[i] for i in self.remaining)+1):
            n,d=factor_pair(M,P,down,up,B-A,hit,self.relaxed)
            if n<=0 or d<=0:raise ArithmeticError('nonpositive factor')
            self.lower[B]=self.lower[B]*n//d
            self.upper[B]=(self.upper[B]*n+d-1)//d
            self.max_lower[B]=max(self.max_lower[B],self.lower[B])
            self.max_upper[B]=max(self.max_upper[B],self.upper[B])
        self.observed_total+=value
        self.remaining=tuple(i for i in self.remaining if i!=identity)
        self.pending=None;self.exact_cache.clear()

    def exact_wealth(self,B):
        key=(B,len(self.history))
        if key not in self.exact_cache:
            w=F(1);mx=w
            for M,P,down,up,A,hit in self.history:
                n,d=factor_pair(M,P,down,up,B-A,hit,self.relaxed)
                w*=F(n,d);mx=max(mx,w)
            self.exact_cache[key]=(w,mx)
        return self.exact_cache[key]

    def accepted(self,running=True,alpha=F(1,20)):
        if self.pending is not None:raise ValueError('query during pending draw')
        if not isinstance(alpha,F) or not 0<alpha<1:raise ValueError('exact Fraction alpha required')
        accepted=[];a=alpha.numerator;d=alpha.denominator
        for B in range(self.observed_total,self.observed_total+sum(self.caps[i] for i in self.remaining)+1):
            lo=self.max_lower[B] if running else self.lower[B]
            hi=self.max_upper[B] if running else self.upper[B]
            if lo*a>=self.scale*d:continue
            if hi*a<self.scale*d:accepted.append(B);continue
            self.fallbacks+=1
            if self.exact_wealth(B)[1 if running else 0]*alpha<1:accepted.append(B)
        return tuple(accepted)

    def result(self,running=True,alpha=F(1,20)):
        a=self.accepted(running,alpha)
        return {'status':'OK' if a else 'EMPTY','lower':a[0] if a else None,
                'upper':a[-1] if a else None,'hull_width':a[-1]-a[0] if a else None,
                'accepted_size':len(a),'fallbacks':self.fallbacks,'running':running}

def intervals_from_trace(trace,bits=128):
    """Consume only the legacy trace; acquire no measurements, change no points."""
    if any(float(l)!=float(u) for l,u in zip(trace.lower,trace.upper)):
        raise ValueError('this candidate requires exact operational measurements')
    caps=tuple(int(k) for k in trace.caps)
    if any(int(k)!=k for k in trace.caps):raise ValueError('trace caps')
    exact=BandAudit(caps,False,bits);relaxed=BandAudit(caps,True,bits)
    for t in range(trace.budget):
        f=trace.forecasts[trace.forecast_index[t]]
        prior=tuple(int(i) for i in trace.order[:f.before_draw])
        if f.before_draw>t or f.fit_ids!=prior:raise ValueError('unqualified forecast lineage')
        values=tuple(f.predictions[i] for i in exact.remaining)
        exact.prepare(values);relaxed.prepare(values)
        i=int(trace.order[t]);y=int(trace.lower[t])
        exact.observe(i,y);relaxed.observe(i,y)
    return {'point_estimate':trace.estimate,'budget':trace.budget,
            'integer_running':exact.result(True),'integer_final':exact.result(False),
            'relaxation_running':relaxed.result(True),'relaxation_final':relaxed.result(False)}
