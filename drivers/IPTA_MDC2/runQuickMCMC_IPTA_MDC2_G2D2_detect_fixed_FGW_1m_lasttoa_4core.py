#!/usr/bin/env python
"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)

Run: G2D2 dataset, Detection mode, fixed fGW.
Loki direct dL sampling pipeline at 100,000,000 saved samples.
tref = last TOA (set in const_mcmc.py).
GWB amplitude prior: U(-18, -11) to match Enterprise.
"""
import sys
import pickle
import numpy as np
np.seterr(all='raise')

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

import QuickCW.QuickCW_IPTA_MDC2_G2D2_detect_fixed_FGW_lastTOA as QuickCW
from QuickCW.QuickMCMCUtils import ChainParams

# pickled pulsar objects
data_pkl = '/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/psr_objects/G2D2_IPTA_MDC2_all_pulsars.pkl'
with open(data_pkl, 'rb') as psr_pkl:
    psrs = pickle.load(psr_pkl)
print("Number of pulsars loaded:", len(psrs))

# tref set in const_mcmc.py (MJD 58175.673402, last TOA, Bjorn's convention)

# 100,000,000 saved samples with thin=10 -> N = 1e9 raw iterations
N = int(1e9)
n_int_block = 10000
save_every_n = 100000
N_blocks = np.int64(N // n_int_block)

fisher_eig_downsample = 2000
n_status_update = 100
n_block_status_update = np.int64(N_blocks // n_status_update)

# divisibility checks
assert N_blocks % n_status_update == 0, "N_blocks must be divisible by n_status_update"
assert N % save_every_n == 0, "N must be divisible by save_every_n"
assert N % n_int_block == 0, "N must be divisible by n_int_block"

# parallel tempering
T_max = 3.0
n_chain = 4

noisefile = '/scratch/na00078/projects/IPTA_MDC2/IPTA_MDC2_data/noise_files/fit_psr_noise_dataset2.json'
rn_emp_dist_file = None
psr_dist_file = None

if len(sys.argv) > 1:
    savefile = sys.argv[1]
else:
    savefile = '/scratch/na00078/projects/IPTA_MDC2/h5_files/G2D2_fixed_detect_loki_100M_lastTOA_4core.h5'
print("Saving to:", savefile)

# sky position fixed to G2D2 injection (per paper Table 1)
cos_gwtheta = np.cos(0.6387905062299246)
gwphi = 3.3335788713091694

# fixed fGW at G2D2 injection value (3.7e-9 Hz), narrow uniform prior
TargFreq = 3.7e-09

chain_params = ChainParams(
    T_max, n_chain, n_block_status_update,
    freq_bounds=np.array([TargFreq - 1e-21, TargFreq + 1e-21]),
    n_int_block=n_int_block,
    save_every_n=save_every_n,
    fisher_eig_downsample=fisher_eig_downsample,
    rn_emp_dist_file=rn_emp_dist_file,
    savefile=savefile,
    thin=10,
    prior_draw_prob=0.2, de_prob=0.6, fisher_prob=0.3,
    dist_jump_weight=0.2, rn_jump_weight=0.3, gwb_jump_weight=0.1,
    common_jump_weight=0.2, all_jump_weight=0.2,
    fix_rn=False, zero_rn=False, fix_gwb=False, zero_gwb=False,
    cos_gwtheta_bounds=[cos_gwtheta - 1e-8, cos_gwtheta + 1e-8],
    gwphi_bounds=[gwphi - 1e-8, gwphi + 1e-8],
)

pta, mcc = QuickCW.QuickCW(
    chain_params, psrs,
    mc_prior='detection',
    psr_distance_file=psr_dist_file,
    noise_json=noisefile,
)

mcc.advance_N_blocks(N_blocks)