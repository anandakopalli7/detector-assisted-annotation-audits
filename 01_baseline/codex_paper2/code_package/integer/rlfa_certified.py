"""Native predictable tuning, exact conditional factors and certified rejection.

Source q+1e-15 is retained ONLY in past tuning records, never in the inferential
factor. Both that choice and shared-last-grid-beta are explicit adaptations.
No compressed floating sufficient statistics; sums of squares use literal
source histories. No outcome-dependent lambda fallback.
"""
from fractions import Fraction as F
import numpy as np
from comparators import Frame,Wealth,rational,safe_stake

class RLFACertified(Frame):
    def __init__(self,N,K,bits=128):
        super().__init__(N,K)
        self.wealth=Wealth(self.C,bits);self.history=[];self.t=0
        self.grid=np.arange(self.C+1,dtype=float)/self.C
        self.raw=np.zeros((self.C+1,1));self.cv=np.zeros(1);self.shared_beta=np.zeros(1)
        self.past_y=[]
    def prepare(self,predictions):
        if self.pending is not None or not self.remaining:raise ValueError('preparation state')
        if len(predictions)!=len(self.remaining):raise ValueError('forecast length')
        native=np.asarray(predictions,dtype=float)/self.K
        if not np.all(np.isfinite(native)) or np.any(native<0) or np.any(native>1):raise ValueError('side information')
        exact=tuple(rational(s) for s in native);mean=sum(exact,F(0))/len(exact)
        us=tuple(s-mean for s in exact)
        # The literal author history convention, including its zero pseudo-row.
        betas=np.clip((np.sum(self.raw*self.cv,axis=1)+1)/(np.sum(self.cv*self.cv)+1e-10+2),-.5,.5)
        values=self.raw-self.shared_beta*self.cv
        raw_lambdas=np.sum(values,axis=1)/(np.sum(values*values,axis=1)+1e-10)
        if not np.all(np.isfinite(betas)) or not np.all(np.isfinite(raw_lambdas)):raise ArithmeticError('predictable tuning')
        betas.flags.writeable=False;raw_lambdas.flags.writeable=False
        self.pending=(us,min(us),max(us),(betas,raw_lambdas),native)
    def beta(self,B,state):return rational(state[0][B])
    def factor(self,B,stage):
        M,A,y,u,umin,umax,state=stage;beta=self.beta(B,state)
        mu=F(B-A,self.C)
        lo=-mu-max(beta*umin,beta*umax);hi=F(M,self.N)-mu-min(beta*umin,beta*umax)
        lam=safe_stake(rational(state[1][B]),lo,hi)
        return 1+lam*(F(M*y,self.C)-mu-beta*u)
    def observe(self,i,y):
        self.check_observe(i,y);us,umin,umax,state,native=self.pending;pos=self.remaining.index(i)
        M=len(self.remaining);stage=(M,self.A,y,us[pos],umin,umax,state);self.history.append(stage)
        for B in self.support():self.wealth.update(B,self.factor(B,stage))
        # Update numerical tuning state only AFTER observing this draw.
        q=np.ones(M)/M;cv=float(native[pos]-np.sum(q*native))
        mu0=float(np.sum((np.ones(len(self.past_y))/self.N)*(np.asarray(self.past_y)/self.K)))
        raw=(float(1/self.N)*float(y/self.K))/(float(q[pos])+1e-15)-(self.grid-mu0)
        self.raw=np.concatenate((self.raw,raw[:,None]),axis=1)
        self.cv=np.append(self.cv,cv);self.shared_beta=np.append(self.shared_beta,state[0][-1])
        self.past_y.append(y);self.t+=1;self.finish(i,y)
    def accepted(self,running=True):
        if self.pending is not None:raise ValueError('pending observation')
        return tuple(B for B in self.support() if not self.wealth.reject(B,20,running,lambda b:(self.factor(b,s) for s in self.history)))
