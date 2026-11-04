#!/usr/bin/env python3
"""Per-galaxy TF-weighted posterior mean and 1D sd of (g1, g2) for the Xu3
fibrepair cache (read-only). Used by T2b to compare the stencil width
distribution to Xu3 and to check the stencil R(sigma) fit against Xu3 m."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/jet/home/xwang30/kl-nn/arch')
from cache_contract import load_cache_partitions, load_partitioned_array
S = Path('/ocean/projects/phy250048p/shared')
C = S / 'cache/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair/test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded'
OUT = S / 'cache/paper_tests_B_inference/xu3_sigma_summary.npz'
cp = load_cache_partitions(C)
mean, sd = [], []
for i, f in enumerate(cp.files['shear_sample']):
    s = np.load(f, mmap_mode='r'); lw = np.load(cp.files['posterior_tf_log_weight'][i], mmap_mode='r')
    for a in range(0, s.shape[0], 200):
        g = np.asarray(s[a:a+200], float); l = np.asarray(lw[a:a+200], float)
        w = np.exp(l - l.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        mu = np.einsum('nk,nkc->nc', w, g); mean.append(mu)
        sd.append(np.sqrt(np.einsum('nk,nkc->nc', w, (g - mu[:, None])**2)))
    if i % 10 == 0: print(i, flush=True)
truth = np.asarray(load_partitioned_array(cp, 'truth'))
cached = np.asarray(load_partitioned_array(cp, 'tf_target_mean_estimates'))[:, 1, :2]
mean = np.concatenate(mean); sd = np.concatenate(sd)
print('max |recomputed - cached mean|', np.abs(mean - cached).max())
np.savez(OUT, mean=mean, sd=sd, truth=truth,
         image_snr=np.asarray(load_partitioned_array(cp, 'image_snr')),
         central_halpha_snr=np.asarray(load_partitioned_array(cp, 'central_halpha_snr')))
print('wrote', OUT)
