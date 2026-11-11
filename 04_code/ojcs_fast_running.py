"""Exact running-set computation, not a new statistical method.

RLFA: a candidate already rejected by the running maximum never re-enters.
Its tuning history is still updated identically for the entire grid.
ConBo: for each fixed stage the positive factor decreases in candidate total
and the negative factor increases. Positive products and their running maxima
preserve this order, so exhaustive-grid endpoints can be found by bisection.
"""
import numpy as np
from gmpy2 import mpq as F
from exact_backend.rlfa_certified import RLFACertified
from exact_backend.comparators import ConBo

class RunningRLFA(RLFACertified):
    def observe(self,i,y):
        self.check_observe(i,y)
        us,umin,umax,state,native=self.pending;pos=self.remaining.index(i)
        M=len(self.remaining);stage=(M,self.A,y,us[pos],umin,umax,state)
        self.history.append(stage)
        cutoff=20*self.wealth.scale
        for B in self.support():
            if self.wealth.ml[B]<cutoff:
                self.wealth.update(B,self.factor(B,stage))
        q=np.ones(M)/M;cv=float(native[pos]-np.sum(q*native))
        mu0=float(np.sum((np.ones(len(self.past_y))/self.N)*(np.asarray(self.past_y)/self.K)))
        raw=(float(1/self.N)*float(y/self.K))/(float(q[pos])+1e-15)-(self.grid-mu0)
        self.raw=np.concatenate((self.raw,raw[:,None]),axis=1)
        self.cv=np.append(self.cv,cv);self.shared_beta=np.append(self.shared_beta,state[0][-1])
        self.past_y.append(y);self.t+=1;self.finish(i,y)
    def accepted(self,running=True):
        if not running:raise ValueError('This optimization implements running sets only')
        return super().accepted(True)

class MonotoneConBo(ConBo):
    def observe(self,i,y):
        self.check_observe(i,y);lp,lm=self.pending
        self.history.append((len(self.remaining),self.A,y,lp,lm))
        x=F(y,self.K);self.sumx+=x;self.t+=1
        mu=(F(1,2)+self.sumx)/(self.t+1);self.vsum+=(x-mu)**2
        self.finish(i,y)
        a=self.accepted(False)
        if not a:self.empty=True
        else:self.boundaries=(a[0],a[-1])
    def rejected(self,B,positive,running):
        w=F(1)
        for stage in self.history:
            w*=self.factor(B,stage,positive)
            if running and w>=40:return True
        return w>=40
    def accepted(self,running=True):
        if self.pending is not None:raise ValueError('pending observation')
        if self.empty:return ()
        low=self.A;high=self.A+len(self.remaining)*self.K
        if self.rejected(high,True,running) or self.rejected(low,False,running):return ()
        left,right=low,high
        while left<right:
            mid=(left+right)//2
            if self.rejected(mid,True,running):left=mid+1
            else:right=mid
        lower=left;left,right=low,high
        while left<right:
            mid=(left+right+1)//2
            if self.rejected(mid,False,running):right=mid-1
            else:left=mid
        return range(lower,left+1) if lower<=left else ()
