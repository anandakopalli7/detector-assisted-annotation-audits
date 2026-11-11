"""Replay saved access/sampling evidence; do not invoke run_method or acquisition.

The frozen ridge kernel is reused to check prediction hashes. Sampling arithmetic,
label provenance and new point-estimator formulas are reconstructed separately.
This is an independent computation check, not independent scientific replication.
"""
import hashlib, math
import numpy as np
from paper2_core_v01 import ridge_predictions
from paper2_stratified_tie_correction_v01 import allocate_exact
from run_development_extension_v01 import offset_ridge
from .methods import METHODS,UPDATES,fold_ids,digest_array

def close(a,b,label):
    if not math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-8):raise ValueError(label)

def power_oracle(q,s,y):
    M=len(q);n=len(s)
    if M==n:return float(np.sum(y)),0.
    a=q[s];den=float(np.sum((a-a.mean())**2))
    lam=0. if den<1e-12 else min(1.,max(0.,float(np.sum((a-a.mean())*(y-y.mean())))/den))
    return M*float(y.mean())+lam*(float(np.sum(q))-M*float(a.mean())),lam

def replay_row(row,frame,y,order,fold_orders,uniforms):
    N=frame.N;K=frame.K;b=row['budget'];points={p['method']:p for p in row['points']}
    if row['frame_id']!=frame.frame_id:raise ValueError('frame identity')
    full_truth=int(np.sum(y))
    for method in METHODS:
        p=points[method];tr=row['traces'][method];ids=np.array(tr['measurement_ids'],dtype=int);vals=np.array(tr['measurement_values'])
        if p['truth']!=full_truth or not np.array_equal(y[ids],vals):raise ValueError('measurement provenance')
        expected=None
        if method in ('direct_HT','raw_difference','seq_detector_100','split_ridge10','split_offset100','ff_power'):
            if not np.array_equal(ids,order[:b]):raise ValueError('SRS stream mismatch')
        if method in ('direct_HT','raw_difference'):
            q=np.zeros(N) if method=='direct_HT' else frame.q
            expected=np.sum(q)+N/b*np.sum(vals-q[ids])
        elif method=='seq_detector_100':
            terms=[];fits=[];past=0.
            for j,i in enumerate(ids):
                if j==0 or j in UPDATES:
                    q=frame.q.copy() if j==0 else offset_ridge(frame.X,frame.q,ids[:j],vals[:j],100,K)
                    fits.append({'before_draw':j,'fit_ids':ids[:j].tolist(),'prediction_sha256':digest_array(q)})
                M=N-j
                if b<N:terms.append(N*(N-b)/(b*M*(M-1))*(past+np.sum(q[order[j:]])+M*(vals[j]-q[i])))
                past+=vals[j]
            if tr['fits']!=fits:raise ValueError('sequential forecast history')
            expected=float(np.sum(vals)) if b==N else math.fsum(terms)
        elif method.startswith('split_'):
            pilot=b//5;P=order[:pilot];R=order[pilot:];S=R[:b-pilot]
            q=ridge_predictions(frame.features,P,y[P],K,10) if method=='split_ridge10' else offset_ridge(frame.X,frame.q,P,y[P],100,K)
            if tr['pilot_ids']!=P.tolist() or tr['pilot_prediction_sha256']!=digest_array(q):raise ValueError('pilot evidence')
            expected=np.sum(y[P])+np.sum(q[R])+len(R)/len(S)*np.sum(y[S]-q[S])
        elif method=='ff_power':
            expected,lam=power_oracle(frame.q,ids,vals)
            close(tr['lambda'],lam,'power weight')
        elif method in ('stratified_HT','ff_strat_power'):
            groups=[np.flatnonzero(frame.strata==h) for h in range(3)];allocation=allocate_exact([len(g) for g in groups],b)
            if tr['allocation']!=allocation:raise ValueError('stratum allocation')
            chosen=[];expected=0.;records=iter(tr['strata'])
            for h,g in enumerate(groups):
                if not len(g):continue
                s=order[frame.strata[order]==h][:allocation[h]];record=next(records);chosen.extend(s.tolist())
                if record['sample_ids']!=s.tolist():raise ValueError('stratum stream')
                if method=='stratified_HT':expected+=len(g)*float(y[s].mean())
                else:
                    local=np.searchsorted(g,s);part,lam=power_oracle(frame.q[g],local,y[s]);expected+=part
                    close(record['lambda'],lam,'stratum weight')
            if chosen!=ids.tolist():raise ValueError('stratum access ordering')
        elif method.startswith('frame_cf2_'):
            folds=fold_ids(frame);samples=[g[fold_orders[k][:b//2]] for k,g in enumerate(folds)]
            if ids.tolist()!=np.concatenate(samples).tolist():raise ValueError('fold sampling stream')
            if len(tr['folds'])!=2:raise ValueError('fold count')
            expected=0.
            for k,g in enumerate(folds):
                train=samples[1-k];s=samples[k];rec=tr['folds'][k]
                actual_train=[] if method=='frame_cf2_direct' else train.tolist()
                if rec['fold_ids']!=g.tolist() or rec['sample_ids']!=s.tolist() or rec['training_ids']!=actual_train:raise ValueError('fold identity or training stream')
                q=np.zeros(N) if method=='frame_cf2_direct' else offset_ridge(frame.X,frame.q,train,y[train],100,K)
                if rec['prediction_sha256']!=digest_array(q[g]):raise ValueError('fold prediction hash')
                expected+=np.sum(q[g])+len(g)/len(s)*np.sum(y[s]-q[s])
        elif method.startswith(('active_','ppat_')):
            lam={'active_count_lure':0.,'ppat_count_half':.5,'ppat_count_one':1.}[method]
            if tr['lambda']!=lam:raise ValueError('active weight')
            remaining=np.arange(N);centered=frame.q-frame.q.mean();past=0.;pieces=[];fits=[]
            for j,i in enumerate(ids):
                if j==0 or j in UPDATES:
                    mu=frame.q.copy() if j==0 else offset_ridge(frame.X,frame.q,ids[:j],vals[:j],100,K)
                    fits.append({'before_draw':j,'fit_ids':ids[:j].tolist(),'prediction_sha256':digest_array(mu)})
                score=np.sqrt(mu[remaining]*(1-mu[remaining]/K)+(mu[remaining]-lam*centered[remaining])**2)
                prob=np.ones(len(remaining))/len(remaining) if np.sum(score)==0 else .9*score/score.sum()+.1/len(remaining)
                err=float(prob.sum()-1);prob/=prob.sum()
                rec=tr['draws'][j]
                if digest_array(prob)!=rec['probabilities_sha256']:raise ValueError('full propensity vector hash')
                chosen=min(int(np.searchsorted(np.cumsum(prob),uniforms[j],side='right')),len(remaining)-1)
                if remaining[chosen]!=i:raise ValueError('active uniform stream')
                close(rec['p'],prob[chosen],'selected propensity');close(rec['normalization_error'],err,'normalization')
                r=vals[j]-lam*centered[i];lower=np.min(-lam*centered[remaining]/prob)
                scale=np.max((K-lam*centered[remaining])/prob)-lower
                close(tr['conditional_mean_centers'][j],past+lower,'conditional mean center')
                close(tr['conditional_mean_scales'][j],scale,'conditional mean scale')
                close(tr['normalized_observations'][j],(r/prob[chosen]-lower)/scale,'normalized observation')
                if b<N:pieces.append(N/b*(1+(N-b)/(N-j-1)*(1/((N-j)*prob[chosen])-1))*r)
                past+=r;remaining=np.delete(remaining,chosen)
            if tr['fits']!=fits:raise ValueError('active forecast history')
            expected=float(np.sum(vals)) if b==N else math.fsum(pieces)
        if expected is not None:close(p['estimate'],expected,'independent point arithmetic: '+method)
    # The six legacy implementations additionally have direct frozen-code
    # equality fixtures. Their confidence inversion is not redundantly rerun here.
    return {'methods_checked':len(METHODS),'active_draws_replayed':3*b}
