"""Statistical helpers for finite TV and cross-time bootstrap bands."""
import numpy as np
from bnn_geometry.diagnostics import ess_mean_raw

def gap_fit(post,prior):
    p=np.sort(np.ravel(post)); q=np.sort(np.ravel(prior)); x=np.union1d(p,q)
    diff=np.searchsorted(p,x,"right")/len(p)-np.searchsorted(q,x,"right")/len(q)
    k=np.argmax(diff)
    return float(x[k]),float(diff[k])

def band(cold,post,seed):
    # Posterior block means and second moments give fast resampling of the reference SD and mean.
    C,N=post.shape
    ess,_=ess_mean_raw(post)
    if not np.isfinite(ess) or ess<=0: raise ValueError("invalid reference ESS")
    b=2**max(0,int(np.ceil(np.log2(max(1,10*C*N/ess)))))
    if N%b or N//b<8: raise ValueError("insufficient reference blocks for simultaneous band")
    origin=post.mean(); z=post-origin
    block=z.reshape(C,-1,b); bm=block.mean(axis=-1); bm2=(block**2).mean(axis=-1)
    sd=post.std(ddof=1); observed=(cold.mean(axis=0)-origin)/sd
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(2000):
        idx=rng.integers(bm.shape[1],size=bm.shape); rows=np.arange(C)[:,None]
        m=bm[rows,idx].mean(); m2=bm2[rows,idx].mean()
        s=np.sqrt((m2-m*m)*C*N/(C*N-1))
        starts=rng.integers(len(cold),size=len(cold))
        draws.append((cold[starts].mean(axis=0)-origin-m)/s)
    half=float(np.quantile(np.max(np.abs(np.asarray(draws)-observed),axis=1),.95))
    return observed,half,b

