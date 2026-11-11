"""Synthetic tests only. Print results; leave the supplied evidence unchanged."""
from pathlib import Path
from itertools import combinations,product,permutations
import sys,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'30_CODE'))
import paper1_math_v01 as K

checks=[]
def check(name,fn):
    fn();checks.append({'name':name,'status':'PASS'})
def close(a,b,tol=1e-10):
    assert np.allclose(a,b,rtol=tol,atol=tol),(a,b)

def covariance_enumeration():
    rng=np.random.default_rng(491)
    popA=rng.integers(0,7,size=(4,9)).astype(float)
    popC=rng.integers(1,9,size=(3,9)).astype(float)
    frame={'A':4,'C':3};totals=[];covs=[]
    for ia,ic in product(combinations(range(4),2),combinations(range(3),2)):
        y=np.vstack([popA[list(ia)],popC[list(ic)]]);h=['A']*2+['C']*2
        t,c,_,_=K.mixed_covariance(y,h,frame);totals.append(t);covs.append(c)
    close(np.mean(covs,axis=0),np.cov(totals,rowvar=False,ddof=0))
    assert abs(np.mean(covs,axis=0)[0,1])>0

def census_zero():
    y=np.arange(36,dtype=float).reshape(4,9)
    t,c,_,s=K.mixed_covariance(y,['B']*4,{'B':4})
    close(c,np.zeros((9,9)));close(t,y.sum(axis=0));close(s,1.)

def domain_zero_rows():
    y=np.arange(36,dtype=float).reshape(4,9);y[2:,0::3]=0
    t,c,meta,s=K.mixed_covariance(y,['A']*4,{'A':8})
    assert meta['A']['sample_n']==4
    close(t[0::3],2*y[:,0::3].sum(axis=0))
    assert np.linalg.eigvalsh(c).min()>-1e-8

def delta_gradient():
    v=np.array([5,12,20,6,15,18,4,8,10],float);c=np.eye(9)
    r=K.ratio_diagnostic(v,c)['overall'];numerical=[]
    def fun(x):return sum(x[i]*x[i+2]/x[i+1] for i in (0,3,6))
    for j in range(9):
        a=v.copy();b=v.copy();a[j]+=1e-5;b[j]-=1e-5;numerical.append((fun(a)-fun(b))/2e-5)
    close(r['gradient'],numerical,tol=1e-8)
    close(r['variance'],np.dot(numerical,numerical),tol=1e-8)

def envelope_enumeration():
    rng=np.random.default_rng(388)
    for _ in range(20):
        c0=rng.integers(0,5,size=(7,3));c1=rng.integers(0,5,size=(7,3))
        a0=c0+rng.integers(0,2,size=(7,3));a1=c1+rng.integers(0,2,size=(7,3))
        t=np.full((7,3),10);z0=rng.integers(1,10,size=(7,3));z1=rng.integers(1,10,size=(7,3))
        lo,hi,zlo,zhi,t=K.observed_envelope(c0,c1,a0,a1,z0,z1,t)
        w=rng.uniform(.1,3,size=(7,1));pi=np.array([.1,.2,.3,1,1,1,.5])[:,None]
        def R(m,z):return ((w*m/pi).sum(axis=0)*(w*t).sum(axis=0)/(w*z).sum(axis=0)).sum()
        L,U=R(lo,zhi),R(hi,zlo)
        for assignment in product((0,1),repeat=7):
            use=np.array(assignment)[:,None].astype(bool)
            assert L-1e-10<=R(np.where(use,c1,c0),np.where(use,z1,z0))<=U+1e-10
            assert L-1e-10<=R(np.where(use,a1,a0),np.where(use,z1,z0))<=U+1e-10

def matching_exhaustion():
    rng=np.random.default_rng(927)
    for _ in range(30):
        a=rng.uniform(0,4,(4,4));b=rng.uniform(0,4,(3,4));a[:,2:]=a[:,:2]+a[:,2:]+.1;b[:,2:]=b[:,:2]+b[:,2:]+.1
        S=K.overlap_matrix(a,b);allowed=S>=.1;best=(-1,-1.)
        for k in range(4):
            for left in combinations(range(4),k):
                for right in permutations(range(3),k):
                    if all(allowed[i,j] for i,j in zip(left,right)):
                        best=max(best,(k,sum(S[i,j] for i,j in zip(left,right))))
        m=K.scientific_match(a,b,.1);assert len(m)==best[0];close(sum(S[i,j] for i,j in m.items()),best[1])
        assert len(K.audit_match(a,b,.1))==best[0]

def boundary_capture():
    # 26px reference matched by 24px original: original-side height filter would fabricate a miss.
    a=[[0,0,10,26]];b=[[0,1,10,25]]
    assert K.scientific_match(a,b)=={0:0}

def dc_union():
    b=[0,0,10,10];d=[[0,0,3,10],[2,0,6,10]]
    close(K.dc_fraction(b,d),.4);close(K.dc_fraction(b,d,'union'),.6)
    close(K.dc_fraction(b,d,'none'),0.)

def bootstrap_coupling():
    h=np.array(['A','A','A','B','B','C','C']);d=np.array(['x','x','y','x','z','y','z'])
    w=K.bootstrap_weights(h,d,'within_stratum',B=100)
    close(w[:,0],w[:,1]);assert np.any(w[:,0]!=w[:,3])
    z=K.bootstrap_weights(h,d,'whole_drive',B=100)
    close(z[:,0],z[:,3]);close(z[:,2],z[:,5]);close(z,K.bootstrap_weights(h,d,'whole_drive',B=100))

def undefined_not_dropped():
    y=np.array([[2,0,1]*3,[1,2,3]*3],float);w=np.array([[1,0],[0,1],[1,1]],float)
    r=K.bootstrap_ratios(y,np.ones_like(y),w)
    assert r['overall']['upper'] is None and r['overall']['n_undefined']==1 and r['overall']['n_replicates']==3
    assert K.ratio_diagnostic(np.array([1,0,3]*3),np.eye(9))['overall']['point'] is None
    assert K.ratio_diagnostic(np.array([0,1,3]*3),np.eye(9))['overall']['Va_log'] is None

def sparse_matrix():
    y=np.array([[0,1,2]*3,[0,2,5]*3,[0,3,4]*3],float)
    _,c,_,_=K.mixed_covariance(y,['C']*3,{'C':5})
    close(c[0],0);assert c[1,1]>0

def cp_boundaries():
    close(K.cp_upper(0,50),1-.05**(1/50));close(K.cp_upper(50,50),1.)
    assert .27<K.cp_upper(8,50)<.271

def release_guards():
    assert K.safe_public_tree({'population':'full','counts':[1,2,3],'covariance':[[1.,0.],[0.,1.]]})
    for x in [{'rows':[{'image_id':'secret'}]},{'mapping':{}},{'x':float('nan')},{'per_repeat':[]}]:
        try:K.safe_public_tree(x)
        except ValueError:pass
        else:raise AssertionError('UNSAFE_RELEASE_ACCEPTED')

if __name__=='__main__':
    for name,fn in [('exact_stratified_mixed_covariance',covariance_enumeration),('census_zero_variance',census_zero),('domain_zero_units_retained',domain_zero_rows),('finite_difference_delta_gradient',delta_gradient),('2560_observed_view_assignments',envelope_enumeration),('matching_exhaustive',matching_exhaustion),('reference_only_height_gate',boundary_capture),('exact_DontCare_union',dc_union),('bootstrap_cluster_coupling',bootstrap_coupling),('undefined_replicates_not_dropped',undefined_not_dropped),('zero_find_component_not_whole_matrix',sparse_matrix),('CP_boundary_identities',cp_boundaries),('aggregate_release_guards',release_guards)]:check(name,fn)
    result={'passed':len(checks),'failed':0,'checks':checks,'kernel_sha256':hashlib.sha256((ROOT/'30_CODE/paper1_math_v01.py').read_bytes()).hexdigest(),'scope':'Synthetic algebra/software checks only; not interval coverage or human measurement validation.'}
    print(json.dumps(result,indent=2))
