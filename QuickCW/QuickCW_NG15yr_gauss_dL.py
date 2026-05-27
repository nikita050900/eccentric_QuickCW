"""C 2021 Bence Becsy
MCMC for CW fast likelihood (w/ Neil Cornish and Matthew Digman)

QuickCW module for NANOGrav 15yr broad frequency targeted CW searches.
    Gaussian dL prior via parameter.Normal on log10_dist.
    GWB gamma: free, Uniform(0, 7).

Gaussian dL prior
-----------------
Replaces log10_h parameterization with log10_dist. For an EM measured
luminosity distance dL_mean with uncertainty sigma_dL, the physically
motivated prior is p(dL) = N(dL_mean, sigma_dL). When sigma_dL/dL_mean
is small (< ~15%), this maps to a Gaussian on log10(dL) via standard
uncertainty propagation:

    mu_log10  = log10(dL_mean)
    sigma_log10 = sigma_dL / (dL_mean * ln(10))

Reference: Steve Taylor (private communication, 2025) recommends
treating the EM distance as a Gaussian centered on the measured
value with sigma equal to the measurement uncertainty.
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


def QuickCW(chain_params, psrs, noise_json=None, use_legacy_equad=False,
            include_ecorr=True, amplitude_prior='detection',
            gwb_gamma_prior=None, psr_distance_file=None,
            backend_selection=True,
            dist_mean=None, dist_sigma=None, mc_bounds=None):
    """Set up all essential objects for QuickCW to do MCMC iterations.

    Parameters
    ----------
    chain_params : ChainParams
        MCMC configuration object.
    psrs : list
        Enterprise pulsar objects.
    noise_json : str
        Path to white noise dictionary JSON.
    use_legacy_equad : bool
        Use old convention for equad. False for NANOGrav 15yr.
    include_ecorr : bool
        Include ECORR white noise. True for NANOGrav.
    amplitude_prior : str
        'detection' for uniform in log amplitude (Bayes factor),
        'UL' for uniform in amplitude (upper limits).
    gwb_gamma_prior : array or None
        Prior range on GWB spectral index gamma.
        None defaults to [0, 7].
    psr_distance_file : str or None
        File with parallax and DM distance info for pulsars.
        None uses Gaussian prior from psr objects.
    backend_selection : bool
        Use enterprise Selection based on backend.
        True for real data, False for simulated.
    dist_mean : float
        Mean luminosity distance in Mpc for the Gaussian prior.
    dist_sigma : float
        1 sigma uncertainty on luminosity distance in Mpc.
    mc_bounds : list or None
        [m_min, m_max] for log10 chirp mass prior bounds.
        None defaults to [6.0, 10.0].

    Returns
    -------
    pta : enterprise PTA object
    mcc : MCMCChain object
    """
    if dist_mean is None or dist_sigma is None:
        raise ValueError(
            "dist_mean and dist_sigma must be provided for the "
            "Gaussian dL prior (both in Mpc).")

    mu_log10_dist = np.log10(dist_mean)
    sigma_log10_dist = dist_sigma / (dist_mean * np.log(10))
    print("Gaussian dL prior: mean = %.3f Mpc, sigma = %.3f Mpc "
          "(%.1f%%)" % (dist_mean, dist_sigma,
                        100.0 * dist_sigma / dist_mean))
    print("Mapped to log10(dL): mu = %.6f, sigma = %.6f" %
          (mu_log10_dist, sigma_log10_dist))

    if mc_bounds is None:
        mc_bounds = [6.0, 10.0]
    m_min, m_max = mc_bounds
    print("Chirp mass prior bounds: log10(Mc) in [%.2f, %.2f]" %
          (m_min, m_max))

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
        eq = white_signals.TNEquadNoise(log10_tnequad=equad,
                                        selection=selection)
    else:
        efq = white_signals.MeasurementNoise(efac=efac,
                                             log10_t2equad=equad,
                                             selection=selection)

    if include_ecorr:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                        selection=selection, name='')

    log10_A = parameter.Uniform(-20, -11)
    gamma = parameter.Uniform(0, 7)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(pl, components=30)

    log10_Agw = parameter.Uniform(-20, -11)('gwb_log10_A')

    if gwb_gamma_prior is None:
        gwb_gamma_prior = np.array([0, 7])
    gamma_gw = parameter.Uniform(gwb_gamma_prior[0],
                                 gwb_gamma_prior[1])('gwb_gamma')

    cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
    crn = gp_signals.FourierBasisGP(cpl,
                                    components=chain_params.gwb_comps,
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

    cos_gwtheta = parameter.Uniform(
        chain_params.cos_gwtheta_bounds[0],
        chain_params.cos_gwtheta_bounds[1])('0_cos_gwtheta')
    gwphi = parameter.Uniform(
        chain_params.gwphi_bounds[0],
        chain_params.gwphi_bounds[1])('0_gwphi')
    print("cos_gwtheta is:", cos_gwtheta)
    print("gwphi is:", gwphi)

    if np.isnan(chain_params.freq_bounds[0]):
        print('Found lower frequency bound of nan. Setting it to 1/T.')
        chain_params.freq_bounds[0] = 1 / Tspan
    log10_fgw = parameter.Uniform(
        np.log10(chain_params.freq_bounds[0]),
        np.log10(chain_params.freq_bounds[1]))('0_log10_fgw')

    phase0 = parameter.Uniform(0, 2 * np.pi)('0_phase0')
    psi = parameter.Uniform(0, np.pi)('0_psi')
    cos_inc = parameter.Uniform(-1, 1)('0_cos_inc')

    p_phase = parameter.Uniform(0, 2 * np.pi)

    log10_dist = parameter.Normal(
        mu_log10_dist, sigma_log10_dist)('0_log10_dist')

    if amplitude_prior == 'detection':
        log10_mc = parameter.Uniform(m_min, m_max)('0_log10_mc')
    elif amplitude_prior == 'UL':
        log10_mc = parameter.LinearExp(m_min, m_max)('0_log10_mc')
    else:
        raise NotImplementedError(
            "amplitude_prior not implemented. Use 'detection' for "
            "uniform in log amplitude or 'UL' for uniform in "
            "amplitude prior.")

    if psr_distance_file is None:
        if np.any(np.array([psr.pdist[0] for psr in psrs]) == 0):
            raise ValueError(
                "Some pulsar distances are zero. Provide "
                "psr_distance_file or use psr objects with "
                "nonzero distance.")

        p_dist = parameter.Normal(0, 1)

        cw_wf = deterministic.cw_delay(
            cos_gwtheta=cos_gwtheta, gwphi=gwphi,
            log10_mc=log10_mc, log10_dist=log10_dist,
            log10_fgw=log10_fgw, phase0=phase0,
            psrTerm=True, p_phase=p_phase, p_dist=p_dist,
            evolve=True, psi=psi, cos_inc=cos_inc,
            tref=cm.tref)
        cw = deterministic.CWSignal(cw_wf, psrTerm=True, name='cw0')

        s = s_base + cw
        models = [s(psr) for psr in psrs]
    else:
        if (np.any(np.array([psr.pdist[0] for psr in psrs]) > 0)) | \
           np.any(np.array([psr.pdist[1] for psr in psrs]) != 1):
            raise ValueError(
                "Mode using parallax/DM pulsar distance priors "
                "requires zero mean, unit variance psr objects.")

        with open(psr_distance_file, 'rb') as fp:
            pulsar_distances = pickle.load(fp)

        models = []
        for psr in psrs:
            cw_delay_args = dict(
                cos_gwtheta=cos_gwtheta, gwphi=gwphi,
                log10_mc=log10_mc, log10_dist=log10_dist,
                log10_fgw=log10_fgw, phase0=phase0,
                psrTerm=True, p_phase=p_phase, evolve=True,
                psi=psi, cos_inc=cos_inc, tref=cm.tref)
            CWSignal_args = dict(psrTerm=True, name='cw0')
            cw = per_pulsar_prior(psr, pulsar_distances,
                                  cw_delay_args, CWSignal_args)
            s = s_base + cw
            models.append(s(psr))

    t1 = perf_counter()
    print("Begin Loading PTA from Enterprise at %8.3fs" % (t1 - ti))
    pta = signal_base.PTA(models)
    t1 = perf_counter()
    print("Finished Loading PTA from Enterprise at %8.3fs" % (t1 - ti))

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


def per_pulsar_prior(enterprise_pulsar, pulsar_distances,
                     cw_delay_args=None, CWSignal_args=None):
    if cw_delay_args is None:
        cw_delay_args = get_default_args(deterministic.cw_delay)
    if CWSignal_args is None:
        CWSignal_args = get_default_args(deterministic.CWSignal)

    if 'DM' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = DMDistParameter(
            pulsar_distances[enterprise_pulsar.name][0],
            pulsar_distances[enterprise_pulsar.name][1])
    elif 'PX' in pulsar_distances[enterprise_pulsar.name]:
        p_dist = PXDistParameter(
            pulsar_distances[enterprise_pulsar.name][0],
            pulsar_distances[enterprise_pulsar.name][1])

    cw_wf = deterministic.cw_delay(p_dist=p_dist, **cw_delay_args)
    cw = deterministic.CWSignal(cw_wf, **CWSignal_args)
    return cw