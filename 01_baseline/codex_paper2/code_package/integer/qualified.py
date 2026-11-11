"""Checked public adapter for the frozen integer-band and comparator variants."""
import math
from comparators import ConBo
from rlfa_certified import RLFACertified
from band import intervals_from_trace as band_intervals

def validate_trace(trace):
    N=len(trace.caps)
    if not N or len(trace.order)!=N or sorted(map(int,trace.order))!=list(range(N)) or any(int(i)!=i for i in trace.order):raise ValueError('trace sampling permutation')
    if type(trace.budget)is not int or not 0<trace.budget<=N:raise ValueError('trace budget')
    if any(not math.isfinite(float(k)) or int(k)!=k or k<=0 for k in trace.caps) or len(set(map(int,trace.caps)))!=1:raise ValueError('equal positive integer caps')
    if len(trace.lower)!=trace.budget or len(trace.upper)!=trace.budget or len(trace.forecast_index)!=trace.budget:raise ValueError('trace measurement lengths')
    K=int(trace.caps[0])
    for l,u in zip(trace.lower,trace.upper):
        if not math.isfinite(float(l)) or l!=u or l!=int(l) or not 0<=l<=K:raise ValueError('exact integer operational observations')
    if not math.isfinite(trace.estimate):raise ValueError('point estimate')
    for t in range(trace.budget):
        index=trace.forecast_index[t]
        if int(index)!=index or not 0<=index<len(trace.forecasts):raise ValueError('forecast index')
        f=trace.forecasts[int(index)]
        if type(f.before_draw)is not int or not 0<=f.before_draw<=t or f.fit_ids!=tuple(int(i) for i in trace.order[:f.before_draw]):raise ValueError('past-only forecast lineage')
        if len(f.predictions)!=N or any(not math.isfinite(float(m)) or not 0<=m<=K for m in f.predictions):raise ValueError('forecast support')
    return N,K

def intervals_from_trace(trace,fixed_predictions,bits=128):
    N,K=validate_trace(trace)
    if len(fixed_predictions)!=N:raise ValueError('fixed side-information length')
    methods={'conbo_wor_integer':ConBo(N,K,bits),'rlfa_fixed_certified':RLFACertified(N,K,bits),'rlfa_past_certified':RLFACertified(N,K,bits)}
    for t in range(trace.budget):
        f=trace.forecasts[int(trace.forecast_index[t])]
        for name,m in methods.items():
            if name=='conbo_wor_integer':m.prepare()
            else:
                predictions=fixed_predictions if name=='rlfa_fixed_certified' else f.predictions
                m.prepare(tuple(predictions[i] for i in m.remaining))
            m.observe(int(trace.order[t]),int(trace.lower[t]))
    return {'point_estimate':trace.estimate,'budget':trace.budget,'methods':{name:{'running':m.result(True),'final':m.result(False)} for name,m in methods.items()}}

def all_intervals(trace,fixed_predictions,bits=128):
    validate_trace(trace)
    band=band_intervals(trace,bits);others=intervals_from_trace(trace,fixed_predictions,bits)
    assert band['point_estimate'].hex()==others['point_estimate'].hex()==trace.estimate.hex()
    rows={key:band[key] for key in ('integer_running','integer_final','relaxation_running','relaxation_final')}
    for name,modes in others['methods'].items():
        for mode,row in modes.items():rows[name+'_'+mode]=row
    return {'point_estimate':trace.estimate,'budget':trace.budget,'intervals':rows}
