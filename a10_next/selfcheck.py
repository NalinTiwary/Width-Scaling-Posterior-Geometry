"""Quick statistical regression checks for the companion, in the project environment."""
import numpy as np
from stats_tools import gap_fit,band
from bnn_geometry.revision.bounds import bound_row
from bnn_geometry.revision.initcheck import time_to_agreement

def main():
    p=np.array([0,0,0,1]); q=np.array([0,0,1,1])
    c,g=gap_fit(p,q)
    assert c==0 and g==.25 and gap_fit(p,p)[1]==0
    rng=np.random.default_rng(17)
    a=rng.normal(size=(32,6)); b=rng.normal(size=(4,2048))
    y,hw,bl=band(a,b,42)
    np.testing.assert_allclose(y,(a.mean(axis=0)-b.mean())/b.std(ddof=1))
    assert np.isfinite(hw) and hw>0 and bl>0
    hit=time_to_agreement([1,2,4,8],[.8,.05,.05,.05],[.1]*4)
    assert hit['time']==2 and not hit['censored']
    assert time_to_agreement([1,2,4],[0]*3,[.4]*3)['censored']
    row=bound_row(label="test",L=2,m=1024,n=16,chi=4)
    assert row['rho_positive'] and not row['passes_input_width_72']
    np.testing.assert_allclose(row['rho'],.49609375)
    print("PASS: two-atom TV, null TV, bootstrap, crossing/censoring and geometry cutoff.")
if __name__=="__main__": main()
