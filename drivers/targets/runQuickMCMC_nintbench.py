#!/usr/bin/env python
"""UL reruns, 10 independent chains per config (2026-10-15). Copy of runQuickMCMC_broad_prod.py
with --chain and --seed added; physics settings unchanged.
Production driver for the NG15yr broad frequency targeted runs.
One driver for all 8 runs: {NGC3115, NGC1316} x {detection, UL} x
{with, without J1713+0747}. Distance priors frozen 2026-08-17:
NGC3115 dL = 10.0 +/- 1.0 Mpc, NGC1316 dL = 19.3 +/- 1.93 Mpc.
tref in QuickCW/const_mcmc.py set to last TOA MJD 59071.007913.
"""
import numpy as np
np.seterr(all='raise')

import pickle
import os
import argparse

from astropy.coordinates import SkyCoord

import QuickCW.QuickCW_NG15yr_gauss_dL as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

TARGETS = {
    'NGC3115': dict(ra='10h05m13.978s', dec='-07d43m06.89s',
                    dist_mean=10.0, dist_sigma=1.0),
    'NGC1316': dict(ra='03h22m41.7009s', dec='-37d12m28.58s',
                    dist_mean=19.3, dist_sigma=1.93),
}

parser = argparse.ArgumentParser(description="NG15yr broad freq targeted production run")
parser.add_argument("--target", type=str, required=True, choices=list(TARGETS.keys()))
parser.add_argument("--amplitude_prior", type=str, required=True, choices=["detection", "UL"])
parser.add_argument("--drop_j1713", action="store_true")
parser.add_argument("--chain", type=int, required=True)
parser.add_argument("--n_int_block", type=int, default=10000)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n_steps", type=int, default=int(1e8))
parser.add_argument("--fgw_max", type=float, default=1e-7)
parser.add_argument("--save_dir", type=str,
    default="/scratch/na00078/projects/15yr_broad_targeted/nint_bench/h5_files")
args = parser.parse_args()

tgt = TARGETS[args.target]
psr_tag = "noJ1713" if args.drop_j1713 else "withJ1713"
run_name = "%s_%s_%s_c%02d_nint%d" % (args.target, args.amplitude_prior, psr_tag, args.chain, args.n_int_block)
np.random.seed(args.seed)
from numba import njit
@njit
def _numba_seed(s):
    np.random.seed(s)
_numba_seed(args.seed)
print("  seed             =", args.seed)
savefile = os.path.join(args.save_dir, run_name + ".h5")
if os.path.exists(savefile):
    raise RuntimeError("savefile exists, refusing to overwrite: " + savefile)

noisefile = '/scratch/na00078/projects/15yr_data/15yr_v1_1/v1p1_wn_dict.json'
psr_dist_file = '/scratch/na00078/projects/15yr_data/15yrCW/pulsar_distances_15yr.pkl'
if args.drop_j1713:
    data_pkl = '/scratch/na00078/projects/15yr_data/15yrCW/v1p1_de440_pint_bipm2019_unshifted_pdist_wo_J1713.pkl'
    rn_emp_dist_file = '/scratch/na00078/projects/15yr_data/15yr_v1_1/rn_distr_v1p1_wo_J1713.pkl'
else:
    data_pkl = '/scratch/na00078/projects/15yr_data/15yrCW/v1p1_de440_pint_bipm2019_unshifted_pdist.pkl'
    rn_emp_dist_file = '/scratch/na00078/projects/15yr_data/15yr_v1_1/rn_distr_v1p1.pkl'

with open(data_pkl, 'rb') as f:
    psrs = pickle.load(f)
assert all(p.name != 'J1713+0747' for p in psrs) == args.drop_j1713 or not args.drop_j1713
if args.drop_j1713:
    assert all(p.name != 'J1713+0747' for p in psrs), 'J1713 present in wo_J1713 pickle'
else:
    assert any(p.name == 'J1713+0747' for p in psrs), 'J1713 missing from full pickle'

print("RUN MANIFEST")
print("  run_name        =", run_name)
print("  n_psr           =", len(psrs))
print("  amplitude_prior =", args.amplitude_prior)
print("  dist prior      = %.2f +/- %.2f Mpc" % (tgt['dist_mean'], tgt['dist_sigma']))
print("  mc_bounds       = [6.0, 10.0]")
print("  fgw_max         =", args.fgw_max)
print("  n_steps         =", args.n_steps)
print("  savefile        =", savefile)

N = args.n_steps
n_int_block = args.n_int_block
save_every_n = 100000
N_blocks = np.int64(N // n_int_block)
fisher_eig_downsample = 2000
n_status_update = 100
n_block_status_update = np.int64(N_blocks // n_status_update)
assert N_blocks % n_status_update == 0
assert N % save_every_n == 0
assert N % n_int_block == 0

T_max = 3.
n_chain = 4

c = SkyCoord(ra=tgt['ra'], dec=tgt['dec'])
cos_gwtheta = np.cos(np.pi / 2 - c.dec.radian)
gwphi = c.ra.radian

chain_params = ChainParams(
    T_max, n_chain, n_block_status_update,
    freq_bounds=np.array([np.nan, args.fgw_max]),
    n_int_block=n_int_block,
    save_every_n=save_every_n,
    fisher_eig_downsample=fisher_eig_downsample,
    rn_emp_dist_file=rn_emp_dist_file,
    savefile=savefile,
    thin=10,
    prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3,
    dist_jump_weight=0.2, rn_jump_weight=0.3,
    gwb_jump_weight=0.1, common_jump_weight=0.2,
    all_jump_weight=0.2,
    fix_rn=False, zero_rn=False,
    fix_gwb=False, zero_gwb=False,
    cos_gwtheta_bounds=[cos_gwtheta - 1e-8, cos_gwtheta + 1e-8],
    gwphi_bounds=[gwphi - 1e-8, gwphi + 1e-8])

pta, mcc = QuickCW.QuickCW(
    chain_params, psrs,
    amplitude_prior=args.amplitude_prior,
    psr_distance_file=psr_dist_file,
    noise_json=noisefile,
    dist_mean=tgt['dist_mean'],
    dist_sigma=tgt['dist_sigma'],
    mc_bounds=[6.0, 10.0])

mcc.advance_N_blocks(N_blocks)
