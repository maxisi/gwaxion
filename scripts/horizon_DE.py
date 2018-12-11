#! /usr/bin/env python

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import leastsq, brentq
import tqdm

import multiprocessing
from functools import partial

import cosmolopy.distance as cd

from gwaxion import physics
from gwaxion.parallel import *

mplparams = {
    'text.usetex': True,  # use LaTeX for all text
    'axes.linewidth': 1,  # set axes linewidths to 0.5
    'axes.grid': False,  # add a grid
    'axes.labelweight': 'normal',
    'font.family': 'serif',
    'font.size': 24,
    'font.serif': 'Computer Modern Roman'
}
matplotlib.rcParams.update(mplparams)

cosmo = {'omega_M_0':0.308, 'omega_lambda_0':0.692, 'omega_k_0':0.0, 'h':0.678}


# #################################################################################
# FUNCTIONS
# #################################################################################

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

# In[2]:

def h95_viterbi(asd_f, T=None, ndet=1):
    # From Lilli: 95% detection efficiency h0 is 4.7e-26 at 201.2 Hz and an ASD of 4E-24 Sqrt[Hz], for T=80 days
    h95ref = 4.7E-26
    f95ref = 201.2
    asd95ref = 4E-24
    Tref = 80*physics.DAYSID_SI
    ndetref = 2
    T = T if T is not None else Tref
    return h95ref * (asd_f/asd95ref) * (Tref/T)**0.25 * (ndetref/float(ndet))**0.5

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

# define h and z factors

orientation_factor = 2.8  # factor to convert from avg. orientation to optimal orientation
h95ref = 4.7E-26 / orientation_factor
asd95ref = 4E-24
tdriftref = 8*physics.DAYSID_SI

log_d0 = np.log10(5)
log_h95ref = np.log10(h95ref)
log_asd95ref = np.log10(asd95ref)
log_tdriftref = np.log10(tdriftref)

def log_h_factor(h0):
    log_h0 = np.log10(h0)
    return log_h0 + log_d0 + log_asd95ref - log_h95ref

def log_z_factor(f0, z, asd_interp):
    dz = cd.luminosity_distance(z, **cosmo)
    fz = f0/(1. + z)
    return np.log10(dz * asd_interp(fz))

def log_tdrift_factor(tdrift_src, z):
    # tdrift_src is the tdrift corresponding to fdot in the *source* frame
    # to get the tdrift for fdot in the *detector* frame must redshift
    log_tdrift_det = np.log10(tdrift_src * (1. + z))
    return 0.25*(log_tdrift_det - log_tdriftref)

# horizon-finding functions
def compute_log_zH_brentq(h0, f0, asd_interp, tdrift=tdriftref, log_z_min=-10, log_z_max=2):
    try:
        def remainder(log_z):
            z = 10**log_z
            return log_h_factor(h0) + log_tdrift_factor(tdrift, z) - log_z_factor(f0, z, asd_interp)
        log_zH = brentq(remainder, log_z_min, log_z_max)
    except ValueError:
        log_zH = np.nan
    return log_zH

# # horizon-finding functions
# def compute_log_zH_brentq(h0, f0, asd_interp, log_z_min=-10, log_z_max=2):
#     try:
#         log_zH = brentq(lambda log_z: log_h_factor(h0) - log_z_factor(f0, 10**log_z, asd_interp), log_z_min, log_z_max)
#     except ValueError:
#         log_zH = np.nan
#     return log_zH
#             
# def compute_log_zH_leastsq(h0, f0, asd_interp, log_z_guess=0):
#     log_zH = leastsq(lambda log_z: log_h_factor(h0) - log_z_factor(f0, 10**log_z, asd_interp), log_z_guess)[0]
#     return log_zH


# #################################################################################
# LOAD DATA
# #################################################################################

# ## Source frame
# 
# Load source-frame quantities from disk, or create if not found

# In[64]:

dfpath = 'peak_DE.hdf5'
#dfpath = 'peak_DE_both-times.hdf5'
rewrite = False

NCPUS_0 = 8
NCPUS_1 = 8

if os.path.exists(dfpath) and not rewrite:
    print "Loading file: %r" % dfpath
    df_max = pd.read_hdf(dfpath, 'table', mode='r')
        
    n_mass = len(set(df_max['mbh']))
    n_chi = len(set(df_max['chi']))

    print n_mass, n_chi
else:
    print "Finding peak numerically..."
    n_mass = 200
    n_chi = 200
    print n_mass, n_chi
    
    mbhs_array = np.logspace(0, 4, n_mass)
    chis_array = np.linspace(1E-4, 1, n_chi, endpoint=False)

    distance = 5E6 * physics.PC_SI

    # create mbh_chi array
    mbh_chis = []
    for mbh in mbhs_array:
        for chi in chis_array:
            mbh_chis.append([mbh, chi])
    
    # run over Ms and chis
    pool = MyPool(NCPUS_0)
    rows = pool.map(partial(get_peak_row_time, distance=distance, ncpus=NCPUS_1), mbh_chis)
    df_max = pd.DataFrame(rows)
    df_max.to_hdf(dfpath, 'table', mode='w')
    
# condition data
df_cond = df_max[(df_max['fgw']>=0.01) & (df_max['fgw']<=1E6) & (df_max['h0']>1E-40)].copy()


# add a column with boson masses
def get_boson(row):
    return row['fgw']*np.pi*physics.HBAR_SI / physics.EV_SI
#     alpha = physics.Alpha(m_bh=row['mbh'], alpha=row['alpha'])
#     return alpha.m_b_ev

df_cond['mu'] = df_cond.apply(lambda row: get_boson(row), axis=1)
df_max['mu'] = df_cond['mu']

# add a columns with alpha
def get_alpha(row):
    return physics.Alpha(m_bh=row['mbh'], m_b=row['boson'], ev=True, msun=True).alpha

# add a columns with fdot and corresponding Tdrift
def get_fdot(row):
    return (3E-14)*(10. / row['mbh'])**2 * (row['alpha']/0.1)**19 * row['chi']**2

def get_tdrift(row, delta_f=7.23):
    # fdot* Tdrift = delta_f
    # delta_f = 1/(2.*Tdrift)
    # Tdrift = sqrt(2.*fdot)
    return 1./np.sqrt(2.*row['fdot'])

if 'alpha' not in df_cond.keys():
    df_cond['alpha'] = df_cond.apply(lambda row: get_alpha(row), axis=1)
    df_max['alpha'] = df_cond['alpha']

df_cond['fdot'] = df_cond.apply(lambda row: get_fdot(row), axis=1)
df_max['fdot'] = df_cond['fdot']

df_cond['tdrift'] = df_cond.apply(lambda row: get_tdrift(row), axis=1)
df_max['tdrift'] = df_cond['tdrift']


# #################################################################################
# Create ASDs
# #################################################################################

## ALIGO DESIGN
# from lalsimulation import SimNoisePSDaLIGOZeroDetHighPower
# 
# flow = 0.001
# fhig = 1E6
# freqs = np.linspace(flow, fhig, int(5E6))
# 
# psd_des = np.array([SimNoisePSDaLIGOZeroDetHighPower(f) for f in freqs])
# asd_des = psd_des**0.5
# 
# psd_des_interp = interp1d(freqs, psd_des)
# asd_des_interp = interp1d(freqs, asd_des, bounds_error=False, fill_value=np.inf)

vals = np.loadtxt('noise_curves_T1500293-v10/aLIGODesign_T1800044.txt')
fs, asd_des_array = vals[:,0], vals[:,1]

# create interpolant (in logspace)
log_asd_des_interp = interp1d(np.log10(fs), np.log10(asd_des_array), fill_value='extrapolate')

def asd_des_interp(f): lasd=log_asd_des_interp(np.log10(f)); return 10**lasd

asds_dict = {'design': asd_des_interp}


## VOYAGER
vals = np.loadtxt('noise_curves_T1500293-v10/voyager.txt')
fs, asd_voy_array = vals[:,0], vals[:,1]

# this curve has an artificial turn around 5.02 Hz so discard those points before
# extrapolation
f_cut = 6
asd_voy_array = asd_voy_array[fs>f_cut]
fs = fs[fs>f_cut]

# create interpolant (in logspace)
log_asd_voy_interp = interp1d(np.log10(fs), np.log10(asd_voy_array), fill_value='extrapolate')

def asd_voy_interp(f): lasd=log_asd_voy_interp(np.log10(f)); return 10**lasd
asds_dict['voy'] = asd_voy_interp


## COSMIC EXPLORER

vals = np.loadtxt('noise_curves_T1500293-v10/ce.txt')
fs, asd_ce_array = vals[:,0], vals[:,1]

# create interpolant (in logspace)
log_asd_ce_interp = interp1d(np.log10(fs), np.log10(asd_ce_array), fill_value='extrapolate')
def asd_ce_interp(f): lasd=log_asd_ce_interp(np.log10(f)); return 10**lasd
asds_dict['ce'] = asd_ce_interp


## EINSTEIN TELESCOPE

vals = np.loadtxt('noise_curves_T1500293-v10/et_d.txt')
fs, asd_et_array = vals[:,0], vals[:,1]

# this curve has an artificial turn around 1.01 Hz so discard those points before extrapolation
f_cut = 1.5
asd_et_array = asd_et_array[fs>f_cut]
fs = fs[fs>f_cut]

# create interpolant (in logspace)
log_asd_et_interp = interp1d(np.log10(fs), np.log10(asd_et_array), fill_value='extrapolate')
def asd_et_interp(f): lasd=log_asd_et_interp(np.log10(f)); return 10**lasd
asds_dict['et'] = asd_et_interp


# #################################################################################
# COMPUTE HORIZONS AND PLOT
# #################################################################################

from gwaxion import utilities

X = df_max['mbh'].reshape(n_chi, n_mass)
Y = df_max['chi'].reshape(n_chi, n_mass)

Z_boson = utilities.smooth_data(df_max['mu'].reshape(n_chi, n_mass), sigma=3)
Z_boson = np.ma.masked_array(Z_boson, mask=((Y<0.15)))

force_update = False

zmin, zmax = 1E-5, 1E5
norm = matplotlib.colors.LogNorm(vmin=zmin, vmax=zmax)
ticks = ticker.LogLocator(numticks=20, base=10, numdecs=8)
print "Colorbar max: %r" % zmax
    
for name, asd_interp in asds_dict.iteritems():
    print "Processing: %r" % name
    if force_update or 'log_zH_%s' % name not in df_max.keys():
        print "Computing horizons..."
        log_zHs = []
        with tqdm.tqdm(range(len(df_cond))) as pbar:
            for h0, f0, tdrift in zip(df_cond['h0'], df_cond['fgw'], df_cond['tdrift']):
                log_zHs.append(compute_log_zH_brentq(h0, f0, asd_interp, tdrift=tdrift))
                pbar.update()        
        log_zHs = np.array(log_zHs)

        # do the following in two steps so pandas takes care of matching
        df_cond['log_zH_%s' % name] = log_zHs
        df_max['log_zH_%s' % name] = df_cond['log_zH_%s' % name]
    log_zHs = df_max['log_zH_%s' % name]
    
    ## PLOT
    mask = ((df_max['fgw']<0.01) | (df_max['fgw']>1E6) | (df_max['h0']<1E-40))
    logzs = np.ma.masked_array(log_zHs, mask=mask)
    Z = z_to_dl(10**logzs).reshape(n_chi, n_mass)
    
    fig, ax = plt.subplots(1, figsize=(11,8))#, figsize=(16,8))
    
    ## smooth the contours
    Z_denoised = utilities.smooth_data(Z, vmin=zmin)

    # manually set the levels
    lev_exp = np.arange(np.floor(np.log10(zmin)-1),
                       np.ceil(np.log10(zmax)+1))
    levs = np.power(10, lev_exp)
    
    cm = ax.contourf(X, Y, Z_denoised, levs, cmap='magma', norm=norm)

    ## let's add boson mass contours
    cs = ax.contour(X, Y, Z_boson, colors='white', locator=ticks, alpha=0.9)
    fmt = lambda v: r"$10^{%i}$" % np.log10(v)
    locs = None#[(2E3, 0.5), (2E2, 0.5), (2E1, 0.5), (2E0, 0.5)]
    plt.clabel(cs, cs.levels[1:], inline=1, fmt=fmt, fontsize=24, use_clabeltext=True, manual=locs)

    ## let's add fdot_det contours
    fdot_det = df_max['fdot']*((1. + 10**logzs)**(-2))
    Z = utilities.smooth_data(fdot_det.reshape(n_chi, n_mass), sigma=3)
    Z = np.ma.masked_array(Z, mask=((Y<0.15)))
    cs = ax.contour(X, Y, Z, colors=('gray',), levels=(1E-8, 10,), alpha=0.2, extend='both')
    ax.contourf(X, Y, Z, colors='none', levels=[1E-8, 1], alpha=0.1)
    
    ax.annotate(r"$\dot f > 10^{-8}$ Hz/s", xy=(0.1, 0.7), xycoords="axes fraction", 
                ha='left', va='bottom', fontsize=20, color='0.6', rotation=21)

    # plot vertical line at 60 Msun
    ax.axvline(60, c='w', ls=':', lw=2)
    ax.axhline(0.7, c='w', ls=':', lw=2)
    
    # add colorbar
    cb = plt.colorbar(cm, label=r'Horizon (Mpc)')
    cb.ax.tick_params(labelsize=18) 
    
    ax.set_xlabel(r'$M_i$')
    ax.set_ylabel(r'$\chi_i$')
    ax.patch.set_facecolor("black")
    
    ax.set_xscale('log')
    
    ax.set_xlim(1, 1E4)
    ax.set_ylim(0, 1)
    
    figpath = 'cont_chi_mbh_range_%s_DE_tdrift.pdf' % name
    fig.savefig(figpath, bbox_inches='tight', dpi=400)
    print "Figure saved: %r" % figpath

    # print peak properties 
    max_loc = df_cond['log_zH_%s' % name].idxmax()
    max_z = 10**df_cond['log_zH_%s' % name].max()
    print "\n Furthest horizon: %.1f Mpc (z=%.2e)" % (z_to_dl(max_z), max_z)
    print pd.DataFrame([df_cond.loc[max_loc][['mbh', 'chi', 'alpha', 'boson', 'h0', 'fgw', 'fdot', 'tinst']]])
    test_z = 10**compute_log_zH_brentq(5E-26, 200, asd_interp)
    # print "\nGW150914 horizon: %.1f Mpc (z=%.2e)" % (z_to_dl(test_z), test_z)
    # test_z = 10**compute_log_zH_leastsq(5E-26, 200, asd_interp)
    # print "GW150914 horizon: %.1f Mpc (z=%.2e)" % (z_to_dl(test_z), test_z)

    # print "Computing examples..."
    # # EXAMPLES
    # distance = 5E6*physics.PC_SI
    # mbh_chis = [(3, 0.9), (10, 0.9), (60, 0.7), (60, 0.9), (200, 0.85), (300, 0.95)]
    # df = pd.DataFrame([get_peak_row_time(mbh_chi, distance=distance) for mbh_chi in mbh_chis])
    # df['mu'] = df.apply(lambda row: get_boson(row), axis=1)
    # df['fdot'] = df.apply(lambda row: get_fdot(row), axis=1)
    # df['tdrift'] = df.apply(lambda row: get_tdrift(row), axis=1)
    # log_zHs = []
    # for h0, f0, tdrift in zip(df['h0'], df['fgw'], df['tdrift']):
    #     log_zHs.append(compute_log_zH_brentq(h0, f0, asd_interp, tdrift=tdrift))
    # log_zHs = np.array(log_zHs)
    # df['zH_%s' % name] = 10**log_zHs
    # df['H_%s' % name] = z_to_dl(10**log_zHs)
    # print df

    print "-------- (end of %s) --------\n" % name

## END
df_max.to_hdf(dfpath, 'table', mode='w')
print "File saved: %r" % dfpath
