"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)

QuickCW module for the t_ref comparison run (CQG paper):
    G2D2 dataset, Detection mode, fixed fGW.
    Direct dL sampling pipeline (Loki Dey).
    tref set to last TOA (Bjorn's Enterprise convention) via driver kwarg.
    GWB gamma: free, U(0, 7).
"""

from time import perf_counter
import json
import pickle
import inspect

import numpy as np

from numba import config
config.THREADING_LAYER = 'omp'
print("Number of cores used for parallel running: ", config.NUMBA_NUM_THREADS)

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import white_signals
from enterprise.signals import gp_signals

from enterprise_extensions import deterministic

import QuickCW.const_mcmc as cm
from QuickCW.QuickMCMCUtils import MCMCChain, ChainParams
from QuickCW.PulsarDistPriors import DMDistParameter, PXDistParameter


def QuickCW(chain_params, psrs, noise_json=None, use_legacy_equad=True,
            include_ecorr=False, mc_prior='detection', gwb_gamma_prior=None,
            psr_distance_file=None, backend_selection=False, tref=None):
    """Set up all essential objects for QuickCW to do MCMC iterations.

    Configuration:
        - G2D2 dataset
        - Detection mode (mc_prior='detection')
        - fixed fGW (set in driver via chain_params.freq_bounds)
        - dL sampled in U(log10(74.646), log10(76.154))  (1% around 75.4 Mpc)
        - GWB gamma: free, U(0, 7)
        - tref: set via kwarg (defaults to cm.tref = MJD 53000 if not provided)

    Parameters
    ----------
    tref : float, optional
        Reference time in seconds. If None, falls back to the QuickCW default
        (cm.tref = MJD 53000 * 86400 s). Pass max_toa to match Bjorn's
        Enterprise runs which use the last TOA as reference epoch.
    """
    # Resolve tref
    if tref is None:
        tref_use = cm.tref
        print("Using QuickCW default tref: %.6f s (MJD %.6f)" % (tref_use, tref_use / 86400.0))
    else:
        tref_use = tref
        print("Using custom tref: %.6f s (MJD %.6f)" % (tref_use, tref_use / 86400.0))

    print("Began Main Loop")
    ti = perf_counter()

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    efac = parameter.Constant()
    equad = parameter.Constant()
    ecorr = parameter.Constant()

    if backend_selection:
        selection = selections.Selection(selections.by_backend)
    else:
        selection = selections.Selection(selections.no_selection)

    if use_legacy_equad:
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
    else:
        efq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)

    if include_ecorr:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    log10_Agw = parameter.Uniform(-20, -11)('gwb_log10_A')

    if gwb_gamma_prior is None:
        gwb_gamma_prior = np.array([0, 7])
    gamma_gw = parameter.Uniform(gwb_gamma_prior[0], gwb_gamma_prior[1])('gwb_gamma')

    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl, components=chain_params.gwb_comps,
                                    Tspan=Tspan, name='gw')

    tm = gp_signals.TimingModel()

    if include_ecorr:
        if use_legacy_equad:
            s_base = ef + eq + ec + rn + crn + tm
        else:
            s_base = efq + ec + rn + crn + tm
    else:
        if use_legacy_equad:
            s_base = ef + eq + rn + crn + tm
        else:
            s_base = efq + rn + crn + tm

    cos_gwtheta = parameter.Uniform(chain_params.cos_gwtheta_bounds[0],
                                    chain_params.cos_gwtheta_bounds[1])('0_cos_gwtheta')
    gwphi = parameter.Uniform(chain_params.gwphi_bounds[0],
                              chain_params.gwphi_bounds[1])('0_gwphi')

    if np.isnan(chain_params.freq_bounds[0]):
        print('Found lower frequency bound of nan - Setting it to 1/T.')
        chain_params.freq_bounds[0] = 1 / Tspan
    log10_fgw = parameter.Uniform(np.log10(chain_params.freq_bounds[0]),
                                  np.log10(chain_params.freq_bounds[1]))('0_log10_fgw')

    m_min = 7.03
    m_max = 10
    if mc_prior == 'detection':
        log10_mc = parameter.Uniform(m_min, m_max)('0_log10_mc')
    elif mc_prior == 'UL':
        log10_mc = parameter.LinearExp(m_min, m_max)('0_log10_mc')
    else:
        raise NotImplementedError(
            "mc_prior provided not implemented. "
            "Use 'detection' for uniform in log-mc or 'UL' for uniform in mc.")

    phase0 = parameter.Uniform(0, 2 * np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

    p_phase = parameter.Uniform(0, 2 * np.pi)

    # Direct dL sampling: 1% tolerance around 75.4 Mpc
    dist_min = 74.646  # [Mpc]
    dist_max = 76.154  # [Mpc]
    log10_dist = parameter.Uniform(np.log10(dist_min), np.log10(dist_max))('0_log10_dist')

    if psr_distance_file is None:
        if np.any(np.array([psr.pdist[0] for psr in psrs]) == 0):
            raise ValueError(
                "Some pulsar distances are zero. Provide psr_distance_file or use "
                "psr objects with nonzero distance.")
        p_dist = parameter.Normal(0, 1)
        cw_wf = deterministic.cw_delay(
            cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
            log10_dist=log10_dist, log10_fgw=log10_fgw, phase0=phase0,
            psrTerm=True, p_phase=p_phase, p_dist=p_dist, evolve=True,
            psi=psi, cos_inc=cos_inc, tref=tref_use)
        cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')
        s = s_base + cw
        models = [s(psr) for psr in psrs]
    else:
        if (np.any(np.array([psr.pdist[0] for psr in psrs]) > 0)) | \
           np.any(np.array([psr.pdist[1] for psr in psrs]) != 1):
            raise ValueError(
                "Mode using parallax/DM pulsar distance priors requires zero-mean, "
                "unit-variance psr objects. Use psr objects matching that or set "
                "psr_distance_file=None.")
        with open(psr_distance_file, 'rb') as fp:
            pulsar_distances = pickle.load(fp)
        models = []
        for psr in psrs:
            cw_delay_args = dict(
                cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                log10_dist=log10_dist, log10_fgw=log10_fgw, phase0=phase0,
                psrTerm=True, p_phase=p_phase, evolve=True,
                psi=psi, cos_inc=cos_inc, tref=tref_use)
            CWSignal_args = dict(psrTerm=True, name='cw0')
            cw = per_pulsar_prior(psr, pulsar_distances, cw_delay_args, CWSignal_args)
            s = s_base + cw
            models.append(s(psr))

    t1 = perf_counter()
    print("Begin Loading Pulsar Timing Array from Enterprise at %8.3fs" % (t1 - ti))
    pta = signal_base.PTA(models)
    t1 = perf_counter()
    print("Finished Loading Pulsar Timing Array from Enterprise at %8.3fs" % (t1 - ti))

    with open(noise_json, 'r') as fp:
        noisedict = json.load(fp)

    pta.set_default_params(noisedict)
    if chain_params.verbosity > 1:
        print(pta.summary())
    if chain_params.verbosity > 0:
        print("List of parameters in the model with their priors:")
        print(pta.params)

    max_toa = np.max(psrs[0].toas)
    for i in range(len(psrs)):
        max_toa = max(max_toa, np.max(psrs[i].toas))

    mcc = MCMCChain(chain_params, psrs, pta, max_toa, noisedict, ti)
    return pta, mcc


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def per_pulsar_prior(enterprise_pulsar: Pulsar, pulsar_distances: dict,
                     cw_delay_args: dict = None, CWSignal_args: dict = None):
    if cw_delay_args is None:
        cw_delay_args = get_default_args(deterministic.cw_delay)
    if CWSignal_args is None:
        CWSignal_args = get_default_args(deterministic.CWSignal)

    if 'DM' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = DMDistParameter(pulsar_distances[enterprise_pulsar.name][0],
                                 pulsar_distances[enterprise_pulsar.name][1])
    elif 'PX' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = PXDistParameter(pulsar_distances[enterprise_pulsar.name][0],
                                 pulsar_distances[enterprise_pulsar.name][1])

    cw_wf = deterministic.cw_delay(p_dist=p_dist, **cw_delay_args)
    cw = deterministic.CWSignal(cw_wf, **CWSignal_args)
    return cw