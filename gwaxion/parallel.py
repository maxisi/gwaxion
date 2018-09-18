import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool

import numpy as np
from functools import partial

from . import physics

# ######################################################################################
# UTILS

# first, some voodoo to allow for nested multiprocesses
# see https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# ######################################################################################
# APPLICATION-SPECIFIC

# def find_best_gw_notimes(mbh_chi, ncpus=2, alpha_thresh=0.01, alpha_step=0.01, **kwargs):
#     """ Get amplitude and frequency of peak GW emission for BH with properties
#     determined by kwargs.
# 
#     Returns
#     -------
#     hmax: float
#         peak amplitude
#     fmax: float
#         peak GW frequency
#     amax: float
#         peak alpha
#     """
#     mbh, chi = mbh_chi
#     # construct alphas
#     alpha_max = physics.get_alpha_max(chi)
#     if alpha_max < alpha_thresh:
#         return [0]*3
#     else:
#         alphas = np.arange(alpha_thresh, alpha_max, alpha_step)
#         # collect peak values
#         pool = multiprocessing.Pool(ncpus)
#         h0r_fs = pool.map(partial(physics.get_gw, m_bh=mbh, chi_bh=chi, **kwargs), alphas)
#         (hmax, fmax), amax = max(zip(h0r_fs, alphas))
#         return hmax, fmax, amax

def find_best_gw(mbh_chi, ncpus=2, alpha_thresh=0.01, alpha_step=0.01, **kwargs):
    """ Get amplitude and frequency of peak GW emission for BH with properties
    determined by kwargs.

    Returns
    -------
    hmax: float
        peak amplitude
    fmax: float
        peak GW frequency
    Ti: float
        number instabilitiy time for peak
    amax: float
        peak alpha
    """
    mbh, chi = mbh_chi
    # construct alphas
    alpha_max = physics.get_alpha_max(chi)
    if alpha_max < alpha_thresh:
        return [0]*5
    else:
        alphas = np.arange(alpha_thresh, alpha_max, alpha_step)
        # collect peak values
        pool = multiprocessing.Pool(ncpus)
        outputs = pool.map(partial(physics.get_gw, m_bh=mbh, chi_bh=chi, times=True,
                                      **kwargs), alphas)
        (hmax, fmax, tinst, tgw), amax = max(zip(outputs, alphas))
        return hmax, fmax, tinst, tgw, amax

def get_peak_row(mbh_chi, distance=1, **kwargs):
    hrmax, fmax, amax = find_best_gw(mbh_chi, **kwargs)
    mbh, chi = mbh_chi
    return {'mbh': mbh, 'chi': chi, 'h0': hrmax/distance, 'fgw': fmax, 'alpha': amax}

def get_peak_row_time(mbh_chi, distance=1, **kwargs):
    hrmax, fmax, tinst, tgw, amax = find_best_gw(mbh_chi, **kwargs)
    mbh, chi = mbh_chi
    return {'mbh': mbh, 'chi': chi, 'h0': hrmax/distance, 'fgw': fmax, 'tinst': tinst,
            'tgw': tgw, 'alpha': amax}


# ######################################################################################
# EXAMPLE
#
# # create mbh_chi array
# mbh_chis = []
# for mbh in mbhs_array:
#     for chi in chis_array:
#         mbh_chis.append([mbh, chi])
# 
# # run over Ms and chis
# pool = MyPool(NCPUS_0)
# rows = pool.map(partial(get_row, distance=distance), mbh_chis)
# df_max = pd.DataFrame(rows)
# df_max.to_hdf(dfpath, 'table', mode='w')
