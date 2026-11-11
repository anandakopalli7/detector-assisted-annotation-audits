"""Source-qualified mathematical comparator specializations.

Exact probability, predictable sufficient statistics, factor and wealth arithmetic.
Not a claim of bit-identical reproduction of either complete author pipeline.
"""
from gmpy2 import mpq as F
import math

EPS=F(1,10**10)
def rational(x):
    if isinstance(x,(int,F)):return F(x)
    if not math.isfinite(float(x)):raise ArithmeticError('nonfinite predictable state')
    return F(*float(x).as_integer_ratio())
def clipped(x,lo,hi):return min(hi,max(lo,x))

class Wealth:
    def __init__(self,capacity,bits=128):
        self.scale=1<<bits;self.lo=[self.scale]*(capacity+1);self.hi=self.lo.copy()
        self.ml=self.lo.copy();self.mh=self.lo.copy();self.fallbacks=0
    def update(self,B,factor):
        n,d=factor.numerator,factor.denominator
        if n<=0 or d<=0:raise ArithmeticError('positive factor required')
        self.lo[B]=self.lo[B]*n//d
        self.hi[B]=(self.hi[B]*n+d-1)//d
        self.ml[B]=max(self.ml[B],self.lo[B]);self.mh[B]=max(self.mh[B],self.hi[B])
    def reject(self,B,threshold,running,replay):
        lo=self.ml[B] if running else self.lo[B];hi=self.mh[B] if running else self.hi[B]
        if lo>=threshold*self.scale:return True
        if hi<threshold*self.scale:return False
        self.fallbacks+=1;w=F(1);mx=w
        for f in replay(B):w*=f;mx=max(mx,w)
        return (mx if running else w)>=threshold

class Frame:
    def __init__(self,N,K):
        if type(N)is not int or type(K)is not int or N<1 or K<1:raise ValueError('positive equal-cap frame required')
        self.N=N;self.K=K;self.C=N*K;self.A=0;self.remaining=tuple(range(N));self.pending=None
    def support(self):return range(self.A,self.A+len(self.remaining)*self.K+1)
    def check_observe(self,i,y):
        if self.pending is None or i not in self.remaining:raise ValueError('prepare before distinct draw')
        if type(y)is not int or not 0<=y<=self.K:raise ValueError('integer observed support')
    def finish(self,i,y):
        self.A+=y;self.remaining=tuple(j for j in self.remaining if j!=i);self.pending=None
    def result(self,running=True):
        a=self.accepted(running)
        return {'status':'OK' if a else 'EMPTY','lower':a[0] if a else None,
                'upper':a[-1] if a else None,'hull_width':a[-1]-a[0] if a else None,
                'accepted_size':len(a),'running':running}

class ConBo(Frame):
    """ConBo-WoR with source aKelly stake, c=1/2 and integer-grid boundaries.

Regularization follows confseq lambda_aKelly: one pseudo-observation,
mean 1/2 and variance 1/4. This is an explicit integer-grid specialization.
"""
    def __init__(self,N,K,bits=128):
        super().__init__(N,K);self.plus=Wealth(self.C,bits);self.minus=Wealth(self.C,bits)
        self.sumx=F(0);self.vsum=F(1,4);self.t=0;self.boundaries=(0,self.C)
        self.history=[];self.empty=False
    def base_stake(self,B):
        M=len(self.remaining);mean=(F(1,2)+self.sumx)/(self.t+1)
        muhat=(self.N*mean-self.sumx)/M;q=F(B-self.A,M*self.K)
        variance=self.vsum/(self.t+1);gap=muhat-q
        raw=gap/(variance+gap*gap)
        if q<1:raw=max(raw,-F(1,2)/(1-q))
        if q>0:raw=min(raw,F(1,2)/q)
        return raw
    def prepare(self):
        if self.pending is not None or not self.remaining:raise ValueError('preparation state')
        if self.empty:self.pending=(F(0),F(0));return
        lower=max(self.boundaries[0],self.A);upper=min(self.boundaries[1],self.A+len(self.remaining)*self.K)
        self.pending=(max(F(0),self.base_stake(lower)),max(F(0),-self.base_stake(upper)))
    def factor(self,B,stage,positive):
        M,A,y,lp,lm=stage;q=F(B-A,M*self.K);x=F(y,self.K)
        if positive:
            lam=lp if q==0 else min(lp,F(1,2)/q)
            return 1+lam*(x-q)
        lam=lm if q==1 else min(lm,F(1,2)/(1-q))
        return 1-lam*(x-q)
    def observe(self,i,y):
        self.check_observe(i,y);lp,lm=self.pending
        stage=(len(self.remaining),self.A,y,lp,lm);self.history.append(stage)
        if not self.empty:
            for B in self.support():
                self.plus.update(B,self.factor(B,stage,True))
                self.minus.update(B,self.factor(B,stage,False))
        x=F(y,self.K);self.sumx+=x;self.t+=1
        mu=(F(1,2)+self.sumx)/(self.t+1);self.vsum+=(x-mu)**2
        self.finish(i,y)
        a=self.accepted(False)
        if not a:self.empty=True
        else:self.boundaries=(a[0],a[-1])
    def accepted(self,running=True):
        if self.pending is not None:raise ValueError('pending observation')
        if self.empty:return ()
        return tuple(B for B in self.support()
          if not self.plus.reject(B,40,running,lambda b:(self.factor(b,s,True) for s in self.history))
          and not self.minus.reject(B,40,running,lambda b:(self.factor(b,s,False) for s in self.history)))

def source_beta(sum_upayoff,sum_u2):
    return clipped((sum_upayoff+1)/(sum_u2+2+EPS),F(-1,2),F(1,2))
def source_kelly(values):
    values=tuple(map(rational,values))
    return sum(values,F(0))/(sum((v*v for v in values),F(0))+EPS) if values else F(0)
def safe_stake(raw,lo,hi):
    low=max(F(-5,2),-1/(hi+EPS)) if hi>0 else F(-5,2)
    high=min(F(5,2),1/(-lo+EPS)) if lo<0 else F(5,2)
    return clipped(raw,low,high)

class RLFA(Frame):
    """Uniform count specialization of source CV/ApproxKelly conventions.

Keeps source shared-last-grid-beta history (not silently corrected).
Uses exact q=1/M, pi=1/N instead of source numerical q+1e-15.
Side forecasts can be fixed (source setting) or predictable (label adaptation).
"""
    def __init__(self,N,K,bits=128):
        super().__init__(N,K);self.wealth=Wealth(self.C,bits);self.history=[]
        self.sum_u=F(0);self.sum_uv=F(0);self.sum_u2=F(0);self.sum_c=F(0);self.sum_c2=F(0);self.t=0
    def beta(self,B,state):
        su,suv,su2,sc,sc2,t=state
        return source_beta(rational(suv)-F(B,self.C)*rational(su),rational(su2))
    def prepare(self,predictions):
        if self.pending is not None or not self.remaining:raise ValueError('preparation state')
        if len(predictions)!=len(self.remaining):raise ValueError('forecast length')
        S=tuple(rational(m)/self.K for m in predictions)
        if any(not 0<=s<=1 for s in S):raise ValueError('side information support')
        mean=sum(S,F(0))/len(S);u=tuple(s-mean for s in S)
        state=(self.sum_u,self.sum_uv,self.sum_u2,self.sum_c,self.sum_c2,self.t)
        self.pending=(u,min(u),max(u),state)
    def factor(self,B,stage):
        M,A,y,u,umin,umax,state=stage
        beta=self.beta(B,state);su,suv,su2,sc,sc2,t=state
        b=F(B,self.C)
        numer=rational(sc)-t*b
        denom=rational(sc2)-2*b*rational(sc)+t*b*b+EPS
        # Exact moments make this a sum of squares plus the specified EPS.
        # Any nonpositive result is an implementation error, never repaired.
        if denom<=0:raise ArithmeticError('nonpositive ApproxKelly denominator')
        raw=numer/denom
        mu=F(B-A,self.C)
        low=-mu-max(beta*umin,beta*umax)
        high=F(M,self.N)-mu-min(beta*umin,beta*umax)
        lam=safe_stake(raw,low,high)
        payoff=F(M*y,self.C)-mu-beta*u
        return 1+lam*payoff
    def observe(self,i,y):
        self.check_observe(i,y);us,umin,umax,state=self.pending
        u=us[self.remaining.index(i)];M=len(self.remaining);A=self.A
        stage=(M,A,y,u,umin,umax,state);self.history.append(stage)
        for B in self.support():self.wealth.update(B,self.factor(B,stage))
        # Literal shared-history convention in the author code: last grid m=1.
        beta_last=self.beta(self.C,state)
        v=F(M*y+A,self.C);c=v-beta_last*u
        self.sum_u+=u;self.sum_uv+=u*v;self.sum_u2+=u*u
        self.sum_c+=c;self.sum_c2+=c*c;self.t+=1
        self.finish(i,y)
    def accepted(self,running=True):
        if self.pending is not None:raise ValueError('pending observation')
        return tuple(B for B in self.support()
          if not self.wealth.reject(B,20,running,lambda b:(self.factor(b,s) for s in self.history)))

def intervals_from_trace(trace,fixed_predictions,bits=128):
    if len(set(map(int,trace.caps)))!=1 or any(int(k)!=k for k in trace.caps):
        raise ValueError('comparator specialization requires equal integer caps')
    if any(l!=u for l,u in zip(trace.lower,trace.upper)):raise ValueError('exact measurements required')
    N=len(trace.caps);K=int(trace.caps[0])
    if len(fixed_predictions)!=N:raise ValueError('fixed side-information length')
    methods={'conbo_wor_integer':ConBo(N,K,bits),
             'rlfa_fixed_exact_probability':RLFA(N,K,bits),
             'rlfa_past_learned_adaptation':RLFA(N,K,bits)}
    for t in range(trace.budget):
        f=trace.forecasts[trace.forecast_index[t]]
        if f.before_draw>t or f.fit_ids!=tuple(int(i) for i in trace.order[:f.before_draw]):
            raise ValueError('forecast lineage')
        for name,m in methods.items():
            if name=='conbo_wor_integer':m.prepare()
            else:
                predictions=fixed_predictions if name=='rlfa_fixed_exact_probability' else f.predictions
                m.prepare(tuple(predictions[i] for i in m.remaining))
            m.observe(int(trace.order[t]),int(trace.lower[t]))
    return {'point_estimate':trace.estimate,'budget':trace.budget,
            'methods':{name:{'running':m.result(True),'final':m.result(False)} for name,m in methods.items()}}
