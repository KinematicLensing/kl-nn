# DESI TF test-set production submission — 2026-08-28

> **Superseded on 2026-08-29.** The sample tables recorded below used
> cap-after-selection for raw HLR above 5 arcsec, and their compact test caches
> used equal posterior-candidate mass. Both policies are obsolete. The revised
> generator excludes raw HLR outside the inclusive 0.1--5 arcsec model support,
> and the revised cache applies within-galaxy TF prior-replacement weights while
> retaining uniform truth-population mass before shape-noise precision.
>
> The remaining recovery/cache/report jobs
> `44783799`, `44783802`, `44783839`, `44783845`, `44783847`,
> `44783850`, `44783854`, `44783856`, and `44783860` were canceled.
> At that point no replacement jobs had been submitted. Existing products from
> that chain must not be treated as revised test-set products; the corrected
> canonical replacement submission is recorded at the end of this document.

Submitted at 2026-08-28T22:40:39-04:00 from
`/jet/home/xwang30/kl-nn`.

## Fixed configuration

- Model: `CNN-CNN-Meta-bounded-hybrid-simv3-r90_1m_s42_44603007`
- Samples: `test_100k_simv3_xu1_tf`, `test_100k_simv3_xu3_tf`,
  and `test_100k_simv3_xu5_tf`
- Catalog-sampling seeds: 42001, 42003, and 42005
- Rows per cut: 100,000
- FITS/LMDB chunk size: 2,000 rows; 50 tasks per cut
- Posterior cache: 100 tasks per cut, 1,000 galaxies per task,
  10,000 identity/R90 draws per galaxy, seed 42
- Cache tag: `testset_10k_s42`
- Report:
  `/ocean/projects/phy250048p/shared/reports/desi_tf_test_set_s42.html`

## Slurm dependency graph

| Stage | Xu1 | Xu3 | Xu5 |
| --- | ---: | ---: | ---: |
| Sample tables (shared retry array) | 44739719 | 44739719 | 44739719 |
| FITS arrays | 44739117 | 44739118 | 44739119 |
| LMDB shard arrays | 44739120 | 44739121 | 44739122 |
| LMDB merges | 44739123 | 44739124 | 44739125 |
| Compact cache arrays | 44739126 | 44739233 | 44739234 |

Combined report job: **44739273**, with `afterok` dependencies on all three
compact cache arrays.

## Sampling-array recovery

The original sampling array, job 44739116, failed during shell startup before
catalog sampling began: `set -u` made the site `/etc/bashrc` reject its unset
`BASHRCSOURCED` guard. The launcher now enables nounset only after the module,
shell startup, and Conda activation steps.

At 2026-08-28T22:44:36-04:00, deterministic replacement array 44739719 was
submitted with the same canonical names and seeds and `ALLOW_OVERWRITE=1`.
FITS jobs 44739117, 44739118, and 44739119 were retargeted from the failed
array to `afterok:44739719_*`; their existing LMDB, cache, and combined-report
dependency chain remains unchanged.

All retry tasks completed successfully: Xu1 in 17:39, Xu3 in 11:43, and Xu5 in
02:06. The submitted retry used its original 30-minute, one-CPU allocation.
After observing the large catalog's I/O and page-cache footprint, the launcher
was hardened for future runs to one hour and two CPUs (about 3.8 GB on
RM-shared); this resource-only change does not alter seeds or sample contents.

## Sample-table validation

| Cut | Rows | Seed | Capped HLR rows | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Xu1 | 100,000 | 42001 | 2,209 | `81517ac3cb26cf2c26b65a72500be2e42385c359a0ab6dbf58482a719899488f` |
| Xu3 | 100,000 | 42003 | 2,305 | `2d4056b8888f48ebac70dfe21ee5c398e6a0933afd917c60619fdaae6dabdad5` |
| Xu5 | 100,000 | 42005 | 3,105 | `36565fd37fdf2d07f55734dafdfed5796920b6801d11edf2426f80e77ea4fcb2` |

Production QA confirmed contiguous IDs, unique catalog rows, finite simulator
inputs, `hlr <= 5`, exact manifest checksums, fixed simulation redshift 0.3,
one draw in every uniform-cosi LHS stratum, and one draw in every truncated-TF
PIT stratum. The three cosi and TF-PIT means agree with 0.5 to better than
`2.1e-8`.

At 2026-08-28T23:07:04-04:00, all 150 FITS-generation tasks were running.
Every LMDB, compact-cache, and report job remained pending on its declared
upstream dependency. Logs use the launcher paths under
`/ocean/projects/phy250048p/shared/logs/`.

## FITS timeout incident and cancellation — 2026-08-29

All FITS tasks began at 2026-08-28T23:04:31-04:00. Seventy-seven elements
reached the four-hour launcher limit and were terminated at
2026-08-29T03:04:53-04:00; there were no Python exceptions, OOM events, quota
errors, or science-row validation failures. Completed-element runtimes reached
03:59:11, and progress was strongly clustered by compute node, establishing an
undersized walltime plus node/shared-filesystem waiting as the failure mode.

| Cut | Complete tasks | Timed-out tasks | Valid FITS | Missing FITS | Incomplete parts |
| --- | ---: | ---: | ---: | ---: | --- |
| Xu1 | 13 | 37 | 85,193 | 14,807 | `1,3-5,14-17,22-50` |
| Xu3 | 26 | 24 | 89,924 | 10,076 | `1-6,19-22,31-44` |
| Xu5 | 34 | 16 | 90,673 | 9,327 | `10-25` |

All 265,790 published files have the canonical part/ID assignment and expected
46,080-byte size. Deep checks of the first and last file in every part (300
files) found no structure, metadata, row-ID, or science-fingerprint errors;
there are no temporary or unexpected files. No cleanup is required.

At 2026-08-29T10:56:13-04:00, LMDB jobs 44739120--44739125, cache arrays
44739126/44739233/44739234, and report job 44739273 were explicitly canceled.
No related job remains queued or running, and no replacement was submitted.

The launcher now requests 12 hours and prints its array, raw-job, node, and log
identities at startup. The runbook limits future FITS submissions to ten
concurrent elements per cut and documents fingerprint-validated sparse resume
without changing the 2,000-row part mapping. Logs from the failed arrays are:

- `/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_44739117_<task>.out`
- `/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_44739118_<task>.out`
- `/ocean/projects/phy250048p/shared/logs/generate_simulator_v3_44739119_<task>.out`

## Recovery submission — 2026-08-29

The checkpoint-preserving recovery chain was submitted without regenerating
the sample tables or deleting any valid FITS:

| Stage | Xu1 | Xu3 | Xu5 |
| --- | ---: | ---: | ---: |
| Sparse FITS recovery | 44783799 | 44783800 | 44783802 |
| LMDB shard arrays | 44783839 | 44783843 | 44783845 |
| LMDB merges | 44783847 | 44783848 | 44783850 |
| Compact cache arrays | 44783854 | 44783855 | 44783856 |

Combined report job: **44783860**.

The FITS arrays use the exact incomplete-part lists above, `--time=12:00:00`,
`%10` throttles, `ALLOW_PARTIAL_ARRAY=1`, and temporary exclusion of the two
extreme outlier nodes `r048,r389`. LMDB arrays are also throttled to ten tasks
per cut. Every later job has an `afterok` dependency on its corresponding
upstream array or merge, and the report depends on all three cache arrays.

At 2026-08-29T11:45:01-04:00, ten FITS tasks per cut were running and the
remaining sparse elements were held only by the intended array limit. New logs
showed the self-identifying banner, verified existing prefixes, and successful
generation of the next missing IDs; no startup or integrity error was present.

The copyable submission template and statistical contract are in
[`DESI_TF_TEST_SET_RUNBOOK.md`](DESI_TF_TEST_SET_RUNBOOK.md).

## Corrected canonical replacement submission — 2026-08-29

The corrected chain was submitted from `/jet/home/xwang30/kl-nn` between
2026-08-29T22:10:21-04:00 and 2026-08-29T22:10:39-04:00. Sample, FITS, and
LMDB dataset names remain exactly `test_100k_simv3_xu1_tf`,
`test_100k_simv3_xu3_tf`, and `test_100k_simv3_xu5_tf`; no renamed replacement
datasets were introduced.

| Stage | Xu1 | Xu3 | Xu5 |
| --- | ---: | ---: | ---: |
| Sample tables (shared array) | 44810458 | 44810458 | 44810458 |
| FITS arrays | 44810459 | 44810460 | 44810461 |
| LMDB shard arrays | 44810462 | 44810465 | 44810466 |
| LMDB merges | 44810467 | 44810468 | 44810469 |
| TF-weighted compact cache arrays | 44810470 | 44810471 | 44810472 |

Combined report job: **44810473**, with `afterok` dependencies on all three
compact cache arrays. Its output is
`/ocean/projects/phy250048p/shared/reports/desi_tf_test_set_tfweighted_v2_s42.html`.
The corrected caches use tag `testset_tfweighted_v2_10k_s42` so they cannot mix
with the incompatible v1 equal-candidate Xu3 cache; this does not change the
canonical sample or LMDB dataset names.

Before submission, the exact obsolete LMDB directory
`/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_xu3_tf` was removed
because the merge stage intentionally refuses to overwrite an existing base
database. Xu1 and Xu5 base databases and all shard directories were absent.
The sample array uses `ALLOW_OVERWRITE=1` to replace the three obsolete
CSV/manifest pairs in place. Existing FITS directories were not deleted:
row-fingerprint validation skips valid files and atomically regenerates stale
ones under the same names.

All six production launchers create an isolated per-task child below
`/ocean/projects/phy250048p/shared/tmp`, export it through `TMPDIR`, `TMP`, and
`TEMP`, and delete only that child on exit. A live lifecycle test passed before
submission. Sample tasks 2 and 3 subsequently completed with exit code 0 and
their scratch children disappeared; task 1 was still running at the first
post-submit check. The Xu3 corrected manifest records 100,000 uniformly sampled
joint catalog rows, inclusive raw-HLR eligibility of 0.1--5 arcsec, TF scatter
of 0.1 dex conditional on catalog `rmag`, and simulation redshift 0.3.

## Slow-node cancellation and replacement chain — 2026-08-30

Live progress showed severe node-correlated waiting rather than simulator
errors. Thirteen elements projected to exceed or narrowly miss the 12-hour
limit and were canceled after preserving every atomically published FITS:

- Xu1 array 44810459: tasks `2,3,13,22,24-26`
- Xu3 array 44810460: tasks `12-14`
- Xu5 array 44810461: tasks `22,25,31`

The obsolete pending downstream jobs 44810462, 44810465--44810473 were also
canceled because their `afterok` dependencies on the original FITS arrays
became unsatisfiable. Replacement jobs retain the canonical sample, FITS, and
LMDB dataset names:

| Stage | Xu1 | Xu3 | Xu5 |
| --- | ---: | ---: | ---: |
| Sparse FITS retries | 44833876 | 44833877 | 44833880 |
| Full 1--50 FITS validation/resume barriers | 44833882 | 44833884 | 44833886 |
| LMDB shard arrays | 44834010 | 44834011 | 44834012 |
| LMDB merges | 44834013 | 44834014 | 44834015 |
| TF-weighted compact cache arrays | 44834016 | 44834017 | 44834018 |

Combined report job: **44834019**. Sparse retries wait for the corresponding
canceled elements to become fully terminal. Each full barrier waits for both
its complete original array and sparse retry array with `afterany`, then runs
all 50 parts through fingerprint-validated `--skip-existing`; therefore it
repairs any additional timeout rather than trusting the original array's failed
aggregate state. Replacement LMDB jobs depend with `afterok` only on these
authoritative barriers.

All new FITS work, and every still-pending element that Slurm allowed to be
updated, excludes the observed slow nodes
`r023,r032,r040,r176,r184,r208,r216,r295,r312,r336,r343`. The full barriers
retain the 12-hour limit and ten-task throttle. Valid completed rows are reused;
only missing or fingerprint-mismatched rows are regenerated.

At the first post-resubmission check, all 13 sparse elements were running on
new nodes `r045`, `r236`, and `r338`. Their logs showed fingerprint-valid prefix
reuse followed by generation of the next stale row. Slurm's 400-second
cancellation grace period bypassed normal shell cleanup and left 13 exact
per-task scratch children; each was removed after its owner was terminal. No
orphaned atomic `.*.tmp.fits` file was present in the canceled parts. Active
retry scratch children remain isolated and will be removed when those tasks
exit.

### Second slow-node intervention

A later rate audit identified seven additional elements projected beyond the
12-hour limit: Xu1 tasks `30-31` on `r249`, Xu3 task `44` on `r294`, and Xu5
tasks `47-48` on `r217` plus `49-50` on `r294`. They were canceled and resumed
as sparse arrays 44844402, 44844403, and 44844404, respectively. The existing
full validation barriers were updated in place to require `afterany` on both
their original FITS arrays and these new retries:

- 44833882 waits for 44810459 and 44844402
- 44833884 waits for 44810460 and 44844403
- 44833886 waits for 44810461 and 44844404

Because the downstream LMDB jobs already depend on those barrier IDs, jobs
44834010--44834019 remain valid and were not canceled or duplicated. Nodes
`r217`, `r249`, and `r294` were added to the FITS exclusion set, which is now
`r023,r032,r040,r176,r184,r208,r216,r217,r249,r294,r295,r312,r336,r343`.

#### Correction for historical outlier node `r048`

Xu5 retry elements 44844404_48--50 initially started on `r048`, a historical
extreme-outlier node that was missing from the current exclusion list. They
were canceled and replaced by sparse retry array **44844934** for tasks 48--50;
task 47 remains in 44844404 and is progressing on `r045`. The Xu5 full
validation barrier now waits for all three relevant arrays:

- 44833886 waits `afterany` for 44810461, 44844404, and 44844934

The replacement elements started on `r289` and `r353` and their logs show
normal row-by-row progress. The Xu1 retries 44844402_30--31 are progressing on
`r045`, and Xu3 retry 44844403_44 is progressing on `r338`. No scratch child or
atomic `.*.tmp.fits` file was left by the canceled `r048` attempts.

Nodes `r048` and `r389` were added to the exclusions for the new retry, all
still-pending original FITS elements, and all full validation barriers. The
complete exclusion set for that work is
`r023,r032,r040,r048,r176,r184,r208,r216,r217,r249,r294,r295,r312,r336,r343,r389`.
The barrier job IDs and downstream jobs 44834010--44834019 remain unchanged.
