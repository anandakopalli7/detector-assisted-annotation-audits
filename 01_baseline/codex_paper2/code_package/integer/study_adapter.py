"""Full accepted-set records for a prospective study; no outcome acquisition."""
from band import BandAudit
from comparators import ConBo
from rlfa_certified import RLFACertified
from qualified import validate_trace

def evaluate(trace,fixed_predictions):
    N,K=validate_trace(trace)
    if len(fixed_predictions)!=N:raise ValueError('fixed side information')
    methods={'integer':BandAudit((K,)*N),'relaxation':BandAudit((K,)*N,True),
             'conbo':ConBo(N,K),'rlfa_fixed':RLFACertified(N,K),'rlfa_past':RLFACertified(N,K)}
    for t in range(trace.budget):
        f=trace.forecasts[int(trace.forecast_index[t])]
        for name,m in methods.items():
            if name=='conbo':m.prepare()
            else:
                pred=fixed_predictions if name=='rlfa_fixed' else f.predictions
                m.prepare([pred[j] for j in m.remaining])
            m.observe(int(trace.order[t]),int(trace.lower[t]))
    result={}
    for name,m in methods.items():
        for mode in ('running','final'):
            a=m.accepted(mode=='running');r=m.result(mode=='running')
            r['accepted_values']=list(a);result[name+'_'+mode]=r
    return result
