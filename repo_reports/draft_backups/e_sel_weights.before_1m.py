"""Sec. 6.3 (2026-10-01): selection response of floor / no-floor weights on the Xu3-like stencil.
Run: python e_sel_weights.py"""
import numpy as np
z=np.load('/ocean/projects/phy250048p/shared/reports/paper-tests/B_inference/t2b_xu3like.tf_equal.cubes.npz')
g=z['ghat']; sd=z['sd']; d=float(z['delta']); R=z['R']; nb=len(g); print('delta',d,'nb',nb)
sg2=0.5*(sd**2).sum(-1)  # (nb,5) per-component variance avg
floor=np.sqrt(sg2[:,0]).mean()
def W(kind):
    if kind=='equal': return np.ones_like(sg2)
    if kind=='floor': return 1/(sg2+floor**2)
    if kind=='nofloor': return 1/sg2
def stat(idx,kind):
    w=W(kind)[idx]; gi=g[idx]; Ri=R[idx]
    m=lambda x,ww:(ww[:,None]*x).sum(0)/ww.sum()
    Rg=np.trace(np.tensordot(w[:,0]/w[:,0].sum(),Ri,axes=(0,0)))/2
    rs=[(m(gi[:,0],w[:,1])[0]-m(gi[:,0],w[:,2])[0])/(2*d),(m(gi[:,0],w[:,3])[1]-m(gi[:,0],w[:,4])[1])/(2*d)]
    return np.array([Rg,np.mean(rs)])
rng=np.random.default_rng(1)
for kind in ('equal','floor','nofloor'):
    v=stat(np.arange(nb),kind); reps=np.array([stat(rng.integers(0,nb,nb),kind) for _ in range(2000)])
    rat=reps[:,1]/reps[:,0]
    w=W(kind)[:,0]; ess=w.sum()**2/(w**2).sum()
    print(kind,'ESS frac %.3f'%(ess/nb),'Rg %.3f+-%.3f'%(v[0],reps[:,0].std()),'Rsel %.4f+-%.4f'%(v[1],reps[:,1].std()),'ratio %.3f+-%.3f'%(v[1]/v[0],rat.std()))
