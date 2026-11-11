"""Development synthetic populations; no real records or final evaluation seeds."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PublicFrame:
    features: np.ndarray
    original: np.ndarray
    detector: np.ndarray
    unmatched: np.ndarray
    overlap: np.ndarray
    drive: np.ndarray
    caps: np.ndarray
    observed_heights: np.ndarray
    observed_a: np.ndarray
    observed_d: np.ndarray

@dataclass(frozen=True)
class ConstructionTruth:
    omitted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    objects: np.ndarray

def generate(mechanism,seed,N=1000,K=20):
    if mechanism not in {f'M{i:02d}' for i in range(1,13)}:raise ValueError('unknown mechanism')
    if N%10 or K!=20:raise ValueError('development generator expects ten-image groups and K=20')
    rng=np.random.default_rng(seed);drive=np.repeat(np.arange(N//10),10)
    high=mechanism in ('M02','M04')
    crowd=.30 if high else .12
    exists=rng.random((N,K))<crowd
    heights=rng.choice([30.,60.,120.],size=(N,K),p=[.35,.40,.25])*rng.uniform(.92,1.08,(N,K))
    pa=np.full((N,K),.80);pd=np.full((N,K),.80)
    shared=0.
    if mechanism in ('M03','M04'):shared=.75
    if mechanism=='M05':
        pa=np.where(heights<40,.35,np.where(heights<80,.80,.95))
        pd=np.where(heights<40,.30,np.where(heights<80,.85,.95))
    if mechanism=='M06':
        hard=rng.random(N//10)<.3
        pa[:]=np.where(hard[drive,None],.35,.90)
        pd[:]=np.where(hard[drive,None],.25,.90);shared=.5
    if mechanism=='M07':pd[:]=0.
    if mechanism=='M08':pd[:]=.85
    if mechanism=='M09':pa[:]=.995;pd[:]=.5
    if mechanism=='M10':
        invisible=rng.random((N,K))<.20
        pa[invisible]=0.;pd[invisible]=0.
    u=rng.random((N,K));v=rng.random((N,K))
    if shared:v=np.where(rng.random((N,K))<shared,u,v)
    a=exists&(u<pa);d=exists&(v<pd)
    y=np.sum(exists&~a,axis=1).astype(int)
    # A misleading proxy is generated through drive-specific spurious proposals.
    spurious=np.zeros(N,dtype=int)
    if mechanism=='M07':spurious=rng.poisson(2.,N)
    if mechanism=='M08':
        d=exists&a&(v<.85)
        # Available original-list count predicts this nuisance; no hidden y input.
        spurious=rng.poisson(.2+np.sum(a,axis=1)*2.)
    if mechanism=='M12':spurious=rng.poisson(.7,N)
    qtrue=d&~a;unmatched=np.sum(qtrue,axis=1)+spurious
    original=np.sum(a,axis=1);detector=np.sum(d,axis=1)+spurious;overlap=np.sum(a&d,axis=1)
    qbins=[np.sum(qtrue&(heights<40),axis=1),np.sum(qtrue&(heights>=40)&(heights<80),axis=1),np.sum(qtrue&(heights>=80),axis=1)]
    qbins[1]=qbins[1]+spurious
    X=np.column_stack([unmatched,original,detector,*qbins]).astype(float)
    lower=y.copy();upper=y.copy()
    if mechanism=='M11':
        withheld=rng.binomial(y,.25);lower=y-withheld;upper=np.minimum(K,y+rng.binomial(K-y,.04))
    if mechanism=='M12':
        lower=np.maximum(0,y-rng.binomial(y,.15));upper=np.minimum(K,y+spurious)
    union=a|d
    obs_h=heights[union];obs_a=a[union].astype(int);obs_d=d[union].astype(int)
    if spurious.sum():
        obs_h=np.r_[obs_h,np.repeat(60.,spurious.sum())];obs_a=np.r_[obs_a,np.zeros(spurious.sum(),dtype=int)];obs_d=np.r_[obs_d,np.ones(spurious.sum(),dtype=int)]
    public=PublicFrame(X,original,detector,unmatched,overlap,drive,np.full(N,K,dtype=int),obs_h,obs_a,obs_d)
    truth=ConstructionTruth(y,lower,upper,np.sum(exists,axis=1))
    return public,truth
