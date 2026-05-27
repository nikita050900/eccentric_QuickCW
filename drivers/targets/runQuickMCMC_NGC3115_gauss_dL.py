#!/usr/bin/env python
"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)

Driver: NGC 3115, broad fGW, Gaussian dL prior.
NANOGrav 15yr dataset.

Target: NGC 3115 (The Spindle Galaxy)
    RA  = 10h 05m 13.978s
    Dec = -07d 43m 06.89s
    dL  = 9.7 +/- 0.4 Mpc (Tonry et al. 2001, ApJ 546, 681; SBF)
         distance modulus (m-M) = 29.93 +/- 0.09 mag
    sigma_dL = 10% fractional uncertainty
    SMBH mass ~ 10^9 Msun (Emsellem et al. 1999)
"""

import numpy as np
np.seterr(all='raise')

import pickle
import os
import argparse

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

from enterprise_extensions import deterministic

from astropy.coordinates import SkyCoord

import QuickCW.QuickCW_NG15yr_gauss_dL as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

# ==================================================================
# DATA
# ==================================================================
data_pkl = '/scratch/na00078/projects/15yr_data/15yrCW/v1p1_de440_pint_bipm2019_unshifted_pdist.pkl'

with open(data_pkl, 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)

print(len(psrs))

# ==================================================================
# MCMC CONFIGURATION
# ==================================================================
N = int(1e9)

n_int_block = 10000
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

# ==================================================================
# NOISE FILES
# ==================================================================
noisefile = '/scratch/na00078/projects/15yr_data/15yr_v1_1/v1p1_wn_dict.json'
rn_emp_dist_file = '/scratch/na00078/projects/15yr_data/15yr_v1_1/rn_distr_v1p1.pkl'
psr_dist_file = '/scratch/na00078/projects/15yr_data/15yrCW/pulsar_distances_15yr.pkl'

# ==================================================================
# ARGPARSE
# ==================================================================
save_dir = "/scratch/na00078/projects/15yr_broad_targeted/loki/h5_files"

parser = argparse.ArgumentParser(
    description="NGC 3115 broad freq targeted search, Gaussian dL prior.")
parser.add_argument(
    "--save_filename",
    type=str,
    default="NGC3115_broad_gauss_dL.h5",
    help="Name of the .h5 file to save (default: %(default)s)")
parser.add_argument(
    "--amplitude_prior",
    type=str,
    choices=["detection", "UL"],
    default="detection",
    help="Amplitude prior type: detection or UL (default: %(default)s)")
args = parser.parse_args()

savefile = os.path.join(save_dir, args.save_filename)
amplitude_prior = args.amplitude_prior

print(f"Saving to: {savefile}")
print(f"Using amplitude_prior: {amplitude_prior}")

# ==================================================================
# TARGET: NGC 3115
# ==================================================================
c = SkyCoord(ra='10h05m13.978s', dec='-07d43m06.89s')
cos_gwtheta = np.cos(np.pi / 2 - c.dec.radian)
gwphi = c.ra.radian

print("NGC 3115 sky position:")
print("  cos_gwtheta = %.8f" % cos_gwtheta)
print("  gwphi       = %.8f" % gwphi)

# Gaussian dL prior from SBF measurement
# Tonry et al. 2001, ApJ 546, 681: (m-M) = 29.93 +/- 0.09
dist_mean = 9.7          # Mpc
dist_sigma = 0.1 * dist_mean  # 10% fractional uncertainty

# ==================================================================
# CHAIN PARAMS
# ==================================================================
chain_params = ChainParams(
    T_max, n_chain, n_block_status_update,
    freq_bounds=np.array([np.nan, 1e-7]),
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
    cos_gwtheta_bounds=[cos_gwtheta - 1e-8,
                        cos_gwtheta + 1e-8],
    gwphi_bounds=[gwphi - 1e-8, gwphi + 1e-8])

# ==================================================================
# RUN
# ==================================================================
pta, mcc = QuickCW.QuickCW(
    chain_params, psrs,
    amplitude_prior=args.amplitude_prior,
    psr_distance_file=psr_dist_file,
    noise_json=noisefile,
    dist_mean=dist_mean,
    dist_sigma=dist_sigma,
    mc_bounds=[6.0, 10.0])

mcc.advance_N_blocks(N_blocks)