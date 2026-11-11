"""Validated public-feature interface for the provisional sequential candidate.

This module performs no file/network access. The measurement callback supplied
to run_candidate is invoked only for sampled IDs by the sequential trace engine.
This interface is a programming boundary, not an operating-system data enclave.
"""
import numpy as np
from paper2_sequential_v01 import build_trace
from paper2_betting_refinements_v01 import refined_endpoint_interval
from run_development_extension_v01 import offset_ridge

UPDATES=(20,40,80,120,160)

def make_forecaster(features,raw_proxy,cap,penalty=100):
    X=np.asarray(features,dtype=float);q=np.asarray(raw_proxy,dtype=float)
    if X.ndim!=2 or not len(X) or X.shape[1]!=9 or not np.all(np.isfinite(X)):
        raise ValueError('candidate requires nine finite public features per image')
    if q.ndim!=1 or len(q)!=len(X) or not np.all(np.isfinite(q)):
        raise ValueError('finite public proxy vector required')
    if isinstance(cap,bool) or not isinstance(cap,(int,np.integer)) or cap<=0:
        raise ValueError('positive global integer count cap required')
    if np.any(q<0) or np.any(q>cap):raise ValueError('proxy must be clipped to declared support')
    if penalty!=100:raise ValueError('provisional candidate uses penalty 100')
    X=X.copy();q=q.copy();X.flags.writeable=False;q.flags.writeable=False
    def forecast(ids,measurements):
        a=np.asarray(ids);y=np.asarray(measurements,dtype=float)
        if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer):raise ValueError('integer past IDs required')
        if len(set(map(int,a)))!=len(a) or np.any(a<0) or np.any(a>=len(X)):
            raise ValueError('past IDs must be distinct and inside frame')
        if y.ndim!=1 or len(y)!=len(a) or not np.all(np.isfinite(y)) or np.any(y<0) or np.any(y>cap):
            raise ValueError('invalid past measurements')
        if not len(a):return q.copy()
        return offset_ridge(X,q,a,y,100,int(cap))
    return forecast

def run_candidate(features,raw_proxy,cap,order,budget,get_measurement):
    predict=make_forecaster(features,raw_proxy,cap)
    trace=build_trace(np.full(len(features),cap),order,budget,get_measurement,predict,UPDATES)
    interval=refined_endpoint_interval(trace,alpha=.05,strategy='predmix_099',guard=1)
    return trace,interval
