"""Pure Paper 1 computation kernels. No input files, writes, or private records."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

BANDS=('H1','H2','H3')
COMPONENT_ORDER=[f'{q}_{b}' for b in BANDS for q in ('M','Z','T')]
FRAME={'A':1071,'B':118,'C':308}
SEED=3192327639
REPLICATES=2000

def boxes(x):
    return np.asarray(x,dtype=float).reshape(-1,4)

def overlap_matrix(a,b,metric='iou'):
    a,b=boxes(a),boxes(b)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh[:,:,0]*wh[:,:,1]
    aa=np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=1)[:,None]
    bb=np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=1)[None,:]
    den=aa+bb-inter if metric=='iou' else np.minimum(aa,bb) if metric=='ios' else aa
    return np.divide(inter,den,out=np.zeros_like(inter),where=den>0)

def audit_match(a,b,threshold=.3):
    """Frozen AM-3 ordered augmenting-path matcher, not scientific IoU tie rule."""
    score=overlap_matrix(a,b); adjacency=[np.flatnonzero(r>=threshold).tolist() for r in score]
    owner=[-1]*score.shape[1]
    def visit(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if owner[j]<0 or visit(owner[j],seen):owner[j]=i;return True
        return False
    for i in range(len(adjacency)):visit(i,set())
    return {i:j for j,i in enumerate(owner) if i>=0}

def scientific_match(a,b,threshold=.5,metric='iou',greedy=False):
    score=overlap_matrix(a,b,'iou'); eligibility=overlap_matrix(a,b,metric)>=threshold
    if score.size==0:return {}
    if greedy:
        used_a=set();used_b=set();out={}
        for v,i,j in sorted(((-score[i,j],i,j) for i,j in zip(*np.where(eligibility)))):
            if i not in used_a and j not in used_b:out[i]=j;used_a.add(i);used_b.add(j)
        return out
    # One extra match outweighs every possible difference in total IoU.
    big=min(score.shape)+1
    weights=np.where(eligibility,big+score,0)
    r,c=linear_sum_assignment(-weights)
    return {int(i):int(j) for i,j in zip(r,c) if eligibility[i,j]}

def dc_fraction(box,regions,rule='max_single'):
    if rule=='none' or len(regions)==0:return 0.
    b=np.asarray(box,float); area=(b[2]-b[0])*(b[3]-b[1])
    if area<=0:return 1.
    if rule=='max_single':return float(overlap_matrix([b],regions,'ioa').max(initial=0))
    rects=[]
    for d in regions:
        x0,y0=max(b[0],d[0]),max(b[1],d[1]);x1,y1=min(b[2],d[2]),min(b[3],d[3])
        if x1>x0 and y1>y0:rects.append((x0,y0,x1,y1))
    xs=sorted({v for r in rects for v in (r[0],r[2])});total=0.
    for a,c in zip(xs[:-1],xs[1:]):
        ys=sorted((r[1],r[3]) for r in rects if r[0]<=a and r[2]>=c)
        if not ys:continue
        lo,hi=ys[0];span=0.
        for l,h in ys[1:]:
            if l>hi:span+=hi-lo;lo,hi=l,h
            else:hi=max(hi,h)
        total+=(c-a)*(span+hi-lo)
    return total/area

def band(height,shift=0.):
    h=float(height)+shift
    return -1 if h<25 else 0 if h<40 else 1 if h<80 else 2

def cp_upper(k,n,alpha=.05):
    if not(0<=k<=n) or n<=0:raise ValueError('BINOMIAL_COUNTS')
    return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))

def mixed_covariance(y,strata,frame=None):
    y=np.asarray(y,float);strata=np.asarray(strata);frame=FRAME if frame is None else frame
    if y.ndim!=2 or y.shape[1]!=9 or len(y)!=len(strata):raise ValueError('VECTOR_SHAPE')
    if not np.isfinite(y).all():raise ValueError('NONFINITE_VECTOR')
    total=np.zeros(9);cov=np.zeros((9,9));detail={};scales=np.ones((len(y),9))
    for h in sorted(set(strata)):
        ix=strata==h;n=int(ix.sum());N=frame[h]
        if n<2 or n>N:raise ValueError('STRATUM_SIZE')
        f=n/N;d=np.tile([1/f,1.,1.],3);S=np.cov(y[ix],rowvar=False,ddof=1)
        part=n*(1-f)*(d[:,None]*S*d[None,:]);cov+=part;scales[ix]=d
        total+=np.sum(y[ix]*d,axis=0)
        detail[h]={'sample_n':n,'frame_N':N,'f':f,'covariance':part.tolist()}
    return total,cov,detail,scales

def ratio_diagnostic(total,cov):
    total=np.asarray(total,float);cov=np.asarray(cov,float);out={}
    for label,bs in [('overall',range(3))]+[(b,[i]) for i,b in enumerate(BANDS)]:
        g=np.zeros(9);R=0.;valid=True
        for i in bs:
            M,Z,T=total[3*i:3*i+3]
            if T<=0 or Z<=0:valid=False;break
            R+=M*T/Z;g[3*i:3*i+3]=[T/Z,-M*T/Z**2,M/Z]
        if not valid:
            out[label]={'point':None,'variance':None,'gradient':None,'Va_z':None,'Va_log':None,'status':'UNDEFINED_DENOMINATOR'};continue
        v=float(g@cov@g)
        if v < -1e-8:raise ValueError('NEGATIVE_DELTA_VARIANCE')
        v=max(v,0.);zupper=R+1.645*math.sqrt(v)
        logupper=None if R<=0 else R*math.exp(min(700,1.645*math.sqrt(v)/R))
        out[label]={'point':float(R),'variance':v,'gradient':g.tolist(),'Va_z':float(zupper),'Va_log':logupper,'status':'DIAGNOSTIC_ONLY'}
    return out

def bootstrap_weights(strata,drives,mode,seed=SEED,B=REPLICATES):
    strata=np.asarray(strata);drives=np.asarray(drives);rng=np.random.default_rng(seed)
    out=np.zeros((B,len(drives)),dtype=float)
    groups=[np.arange(len(drives))] if mode=='whole_drive' else [np.flatnonzero(strata==h) for h in sorted(set(strata))]
    # Fixed replicate-major, then A/B/C loop, sorted drive IDs, PCG64 integer draws.
    defs=[]
    for ix in groups:
        units=sorted(set(drives[ix]));index={d:k for k,d in enumerate(units)}
        defs.append((ix,np.array([index[d] for d in drives[ix]]),len(units)))
    for r in range(B):
        for ix,indices,K in defs:
            multiplicities=np.bincount(rng.integers(0,K,size=K),minlength=K)
            out[r,ix]=multiplicities[indices]
    return out

def bootstrap_ratios(y,scales,weights):
    totals=weights@(np.asarray(y)*np.asarray(scales));out={}
    M,Z,T=totals[:,0::3],totals[:,1::3],totals[:,2::3]
    valid=(Z>0)&(T>0)
    R=np.divide(M*T,Z,out=np.full(M.shape,np.nan),where=valid)
    for label,cols in [('overall',range(3))]+[(b,[i]) for i,b in enumerate(BANDS)]:
        vals=np.sum(R[:,list(cols)],axis=1);invalid=int((~np.isfinite(vals)).sum())
        # No complete-case quantile: any unidentified replicate makes this diagnostic limit unavailable.
        out[label]={'upper':None if invalid else float(np.quantile(vals,.95,method='linear')),
                    'n_replicates':len(vals),'n_undefined':invalid,
                    'status':'UNDEFINED_REPLICATES' if invalid else 'DIAGNOSTIC_ONLY'}
    return out

def observed_envelope(c0,c1,a0,a1,z0,z1,t):
    c0,c1,a0,a1,z0,z1,t=map(lambda x:np.asarray(x,float),(c0,c1,a0,a1,z0,z1,t))
    if not all(x.shape==t.shape for x in (c0,c1,a0,a1,z0,z1)):raise ValueError('ENVELOPE_SHAPE')
    if np.any(c0>a0) or np.any(c1>a1) or np.any(z0>t) or np.any(z1>t):raise ValueError('ENVELOPE_COUNTS')
    return (np.minimum(c0,c1),np.maximum(a0,a1),np.minimum(z0,z1),np.maximum(z0,z1),t)

def make_y(m,z,t):
    return np.stack((m,z,t),axis=2).reshape(len(m),9)

def safe_public_tree(obj):
    """Exact aggregate schema is enforced by caller; deny identity/row keys recursively."""
    blocked={'image_id','image_ids','task_id','task_ids','marks','coordinates','drive','drive_id','task_to_image','per_image','per_repeat','raw_record','mapping','image_order','drive_order'}
    if isinstance(obj,dict):
        if blocked.intersection(obj):raise ValueError('PROTECTED_RELEASE_KEY')
        for v in obj.values():safe_public_tree(v)
    elif isinstance(obj,list):
        for v in obj:safe_public_tree(v)
    elif isinstance(obj,float) and not math.isfinite(obj):raise ValueError('NONFINITE_RELEASE')
    elif not isinstance(obj,(str,int,float,bool,type(None))):raise ValueError('RELEASE_TYPE')
    return True
