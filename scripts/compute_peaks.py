#! /usr/bin/env python

import os, subprocess
import numpy as np
import pandas as pd
# from scipy.interpolate import interp1d, interp2d
# from scipy.optimize import leastsq, root, brentq, fsolve
import tqdm

import multiprocessing
from functools import partial

#import pycbc.waveform
#import cosmolopy.distance as cd

from gwaxion import physics

# mplparams = {
#     'text.usetex': True,  # use LaTeX for all text
#     'axes.linewidth': 1,  # set axes linewidths to 0.5
#     'axes.grid': False,  # add a grid
#     'axes.labelweight': 'normal',
#     'font.family': 'serif',
#     'font.size': 24,
#     'font.serif': 'Computer Modern Roman'
# }
# matplotlib.rcParams.update(mplparams)
# 
# cosmo = {'omega_M_0':0.308, 'omega_lambda_0':0.692, 'omega_k_0':0.0, 'h':0.678}


# # Horizon
# 
# To find the horizon we want to set $\rho = 1$, where
# \begin{equation*}
# \rho \equiv \frac{h_\oplus}{h_{\rm th}(f_\oplus)} = \left( \frac{h_0}{h_{95}} d_0\, {\rm ASD}_{95} \right) \frac{1}{d_z\, {\rm ASD}[f_0\left(1+z\right)^{-1}]}\, ,
# \end{equation*}
# and $h_\oplus$ and $f_\oplus$ are the GW amplitude and frequency measured at Earth, namely
# \begin{align}
# h_\oplus &= h_0 \frac{d_0}{d_z}\, ,\\
# f_\oplus &= f_0 \left(1+z\right)^{-1}\, ,
# \end{align}
# with $f_0$ the source-frame GW frequency and $h_0$ the strain at luminosity distance $d_0$.
# 
# The denominator in the second factor above can be interpolated for $f_0$ and $z$.

# In[3]:

def h95_viterbi(asd_f, T=None):
    # From Lilli: 95% detection efficiency h0 is 4.7e-26 at 201.2 Hz and an ASD of 4E-24 Sqrt[Hz], for T=80 days
    h95ref = 4.7E-26
    f95ref = 201.2
    asd95ref = 4E-24
    Tref = 80*physics.DAYSID_SI
    T = T if T is not None else Tref
    return h95ref * (asd_f/asd95ref) * (Tref/T)**0.25

def euclidean_horizon(h0, f=None, asd_interp=None, dref=5, T=None):
    """ Computes horizon (Mpc) assuming no redshift."""
    asd = asd_interp(f)
    h95 = h95_viterbi(asd, T=T)
    return h0 * dref / h95

def z_to_dl(z):
    return cd.luminosity_distance(z, **cosmo)

## find redshift for a given luminosity distance
def dl_to_z(dl, z0=1):
    return leastsq(lambda z: dl - cd.luminosity_distance(z, **cosmo), z0)[0]

# ## estimate the horizon for recursive evaluation in the main code
# def horizon_dist_eval(h_ratio, d0, z0):
#     guess_dist = d0*h_ratio
#     guess_redshift, res = leastsq(findzfromDL, z0, args=guess_dist)
#     return guess_redshift[0], guess_dist



# In[4]:

# define h and z factors

h95ref = 4.7E-26
asd95ref = 4E-24
log_d0 = np.log10(5)
log_h95ref = np.log10(h95ref)
log_asd95ref = np.log10(asd95ref)

def log_h_factor(h0):
    log_h0 = np.log10(h0)
    return log_h0 + log_d0 + log_asd95ref - log_h95ref

def log_z_factor(f0, z, asd_interp):
    dz = cd.luminosity_distance(z, **cosmo)
    fz = f0/(1. + z)
    return np.log10(dz * asd_interp(fz))

# horizon-finding functions
def compute_log_zH_brentq(h0, f0, asd_interp, log_z_min=-10, log_z_max=2):
    try:
        log_zH = brentq(lambda log_z: log_h_factor(h0) - log_z_factor(f0, 10**log_z, asd_interp), log_z_min, log_z_max)
    except ValueError:
        log_zH = np.nan
    return log_zH
            
def compute_log_zH_leastsq(h0, f0, asd_interp, log_z_guess=0):
    log_zH = leastsq(lambda log_z: log_h_factor(h0) - log_z_factor(f0, 10**log_z, asd_interp), log_z_guess)[0]
    return log_zH


# first, some voodoo to allow for nested multiprocesses
# see https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic

import multiprocessing
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool

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


# In[16]:

NCPUS_0 = 8 
NCPUS_1 = 8

def get_gws(a, lgw=2, l=1, m=1, nr=0, **kwargs): 
    cloud = physics.BosonCloud.from_parameters(l, m, nr, alpha=a, evolve_params={'y_0': 1E-8}, **kwargs) 
    return cloud.gw(lgw).h0r, cloud.gw(lgw).f

def scan_alphas(mbh_chi):
    mbh, chi = mbh_chi
    # construct alphas
    alpha_max = physics.get_alpha_max(chi)
    if alpha_max < 0.01:
        return [0]*2
    else:
        alphas = np.arange(0.01, alpha_max, 0.01)
        # collect peak values
        pool = multiprocessing.Pool(NCPUS_1)
        h0r_fs = pool.map(partial(get_gws, m_bh=mbh, chi_bh=chi), alphas)
        return max(h0r_fs)

def get_row(mbh_chi, distance=1):
    hrmax, fmax = scan_alphas(mbh_chi)
    mbh, chi = mbh_chi
    return {'mbh': mbh, 'chi': chi, 'h0': hrmax/distance, 'fgw': fmax}


# ## Source frame
# 
# Load source-frame quantities from disk, or create if not found

# In[17]:

dfpath = 'peak_DE.hdf5'
rewrite = True

n_mass = 200
n_chi = 200

mbhs_array = np.logspace(-1, 4, n_mass)
chis_array = np.linspace(1E-4, 1, n_chi, endpoint=False)

distance = 5E6 * physics.PC_SI

# create mbh_chi array
mbh_chis = []
for mbh in mbhs_array:
    for chi in chis_array:
        mbh_chis.append([mbh, chi])

print "Running over Ms and chis..."
pool = MyPool(NCPUS_0)
rows = pool.map(partial(get_row, distance=distance), mbh_chis)
df_max = pd.DataFrame(rows)
df_max.to_hdf(dfpath, 'table', mode='w')
print "File saved: %r" % dfpath

