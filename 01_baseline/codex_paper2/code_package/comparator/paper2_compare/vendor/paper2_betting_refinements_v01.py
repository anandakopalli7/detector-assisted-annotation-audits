"""Established stake clipping and fixed-mixture variations; development only."""
import math
import numpy as np
from scipy.special import logsumexp
from paper2_core_v01 import Bounds
from paper2_sequential_v01 import SequentialCS

FRACTIONS=np.array([1/16,1/8,1/4,1/2,3/4,.9,.99])

class RefinedCS(SequentialCS):
    def __init__(self,trace,endpoint='lower',alpha=.05,strategy='predmix_09',prefix=None,guard=1):
        if strategy not in ('predmix_09','predmix_099','grid_extended'):raise ValueError('refinement strategy')
        self.refinement=strategy
        super().__init__(trace,endpoint,alpha,'predmix',prefix,guard)

    def max_log_wealth(self,total):
        self.evaluations+=1
        if not self.n:return (0.,0.)
        q=np.zeros(self.n);q[self.active]=(total-self.center[self.active])/self.scale[self.active]
        if np.any(q < -1e-9) or np.any(q>1+1e-9):raise ArithmeticError('candidate incompatible with support')
        q=np.clip(q,0,1);tiny=1e-14
        bad=self.active&(((q<=tiny)&(self.x>1e-10))|((q>=1-tiny)&(self.x<1-1e-10)))
        if np.any(bad):raise ArithmeticError('near-boundary ambiguity')
        interior=self.active&(q>tiny)&(q<1-tiny)
        if self.refinement=='grid_extended':
            fp=np.ones((self.n,len(FRACTIONS)));fn=fp.copy()
            fp[interior]=1-FRACTIONS+self.x[interior,None]/q[interior,None]*FRACTIONS
            fn[interior]=1-FRACTIONS+(1-self.x[interior,None])/(1-q[interior,None])*FRACTIONS
            lp=logsumexp(np.cumsum(np.log(fp),axis=0),axis=1)-math.log(len(FRACTIONS))
            ln=logsumexp(np.cumsum(np.log(fn),axis=0),axis=1)-math.log(len(FRACTIONS))
        else:
            c=.9 if self.refinement=='predmix_09' else .99
            fp=np.ones(self.n);fn=fp.copy()
            fp[interior]=1+np.minimum(self.bets[interior],c/q[interior])*(self.x[interior]-q[interior])
            fn[interior]=1-np.minimum(self.bets[interior],c/(1-q[interior]))*(self.x[interior]-q[interior])
            if np.any(fp<=0) or np.any(fn<=0):raise ArithmeticError('nonpositive factor')
            lp=np.cumsum(np.log(fp));ln=np.cumsum(np.log(fn))
        return float(np.max(lp,initial=0)),float(np.max(ln,initial=0))

def refined_endpoint_interval(trace,alpha=.05,strategy='predmix_09',guard=1):
    if np.array_equal(trace.lower,trace.upper):return RefinedCS(trace,alpha=alpha,strategy=strategy,guard=guard).bounds()
    lo=RefinedCS(trace,'lower',alpha,strategy,guard=guard).bounds('lower');up=RefinedCS(trace,'upper',alpha,strategy,guard=guard).bounds('upper')
    if lo.status!=up.status or lo.status!='OK' or lo.lower>up.upper:return Bounds(None,None,'EMPTY',lo.evaluations+up.evaluations)
    return Bounds(lo.lower,up.upper,'OK',lo.evaluations+up.evaluations)
