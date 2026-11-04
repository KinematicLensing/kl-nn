"""Sec. 6.3 (2026-10-01): selection response of cuts on the NPE posterior-mean cos i on the Xu3-like stencil."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/jet/home/xwang30/kl-nn/arch')
from cache_contract import load_cache_partitions
from pathlib import Path
root=Path('/ocean/projects/phy250048p/shared/cache/paper_tests_B_inference/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair/paper_tests_B_shear_response_xu3like_5k_stencil_10k_s42_cpu')
cp=load_cache_partitions(root)
est=np.concatenate([np.load(f) for f in cp.files['tf_target_mean_estimates']]); tr=np.concatenate([np.load(f) for f in cp.files['truth']])
print(est.shape)
z=np.load('/ocean/projects/phy250048p/shared/reports/paper-tests/B_inference/t2b_xu3like.tf_equal.cubes.npz')
g=z['ghat']; d=float(z['delta']); R=z['R']; nb=len(g)
cpost=est[:,1,3].reshape(nb,5); ctrue=tr[:,3].reshape(nb,5)
assert np.allclose(est[:,1,:2].reshape(nb,5,2),g,atol=1e-5)
print('cos post vs true corr',np.corrcoef(cpost[:,0],ctrue[:,0])[0,1])
rng=np.random.default_rng(2)
def stat(idx,c):
    sel=cpost[idx]>c; gi=g[idx]; Ri=R[idx]
    def m(x,mask): return x[mask].mean(0)
    Rg=np.trace(Ri[sel[:,0]].mean(0))/2
    rs=[(m(gi[:,0],sel[:,1])[0]-m(gi[:,0],sel[:,2])[0])/(2*d),(m(gi[:,0],sel[:,3])[1]-m(gi[:,0],sel[:,4])[1])/(2*d)]
    return np.array([Rg,np.mean(rs),sel[:,0].mean()])
for c in (0.8,0.6,0.436,0.312):
    v=stat(np.arange(nb),c); reps=np.array([stat(rng.integers(0,nb,nb),c) for _ in range(2000)])
    flips=[(cpost[:,1]>c)!=(cpost[:,2]>c)]
    print('cos_post>%.3f kept %.2f Rg %.3f Rsel %.4f+-%.4f ratio %.3f+-%.3f flips %d'%(c,v[2],v[0],v[1],reps[:,1].std(),v[1]/v[0],(reps[:,1]/reps[:,0]).std(),flips[0].sum()))
