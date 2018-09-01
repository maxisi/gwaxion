#! /usr/bin/env python

import os, subprocess
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import leastsq
import tqdm

import pycbc.waveform
import cosmolopy.distance as cd

from gwaxion import physics

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


# ## Preamble


# # PSDs
# 
# ## aLIGO design

# In[3]:

from lalsimulation import SimNoisePSDaLIGOZeroDetHighPower

flow = 0.01
fhig = 1E5
freqs = np.linspace(flow, fhig, 4000)

psd_des = np.array([SimNoisePSDaLIGOZeroDetHighPower(f) for f in freqs])
asd_des = psd_des**0.5

psd_des_interp = interp1d(freqs, psd_des)
asd_des_interp = interp1d(freqs, asd_des, bounds_error=False, fill_value=np.inf)

# fig, ax = plt.subplots(1)
# ax.plot(freqs, asd_des)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(flow, fhig)
# ax.set_xlabel(r'$f$ (Hz)')
# ax.set_ylabel(r'ASD')
# plt.show(fig)


# # Interpolant
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

# In[4]:

# define z-factor

def log_z_factor(f0, z, asd_interp):
    dz = cd.luminosity_distance(z, **cosmo)
    fz = f0/(1. + z)
    return np.log10(dz * asd_interp(fz))


# In[5]:

# create data for z-factor interpolant

f0_array = np.logspace(1, 4, 500)
z_array = np.logspace(-4, 3, 500)

f0s, zs = [], []
for f0 in f0_array:
    for z in z_array:
        f0s.append(f0)
        zs.append(z)
f0s = np.array(f0s)
zs = np.array(zs)
        
log_zfactors = log_z_factor(f0s, zs, asd_des_interp)


# In[6]:

log_zf_ma = np.ma.masked_array(log_zfactors, np.isinf(log_zfactors))
mask = log_zf_ma.mask
log_zf_ma = log_zf_ma[~mask]
f0s_ma = f0s[~mask]  # np.ma.masked_array(f0s, np.isinf(log_zfactors))
zs_ma = zs[~mask]  # np.ma.masked_array(zs, np.isinf(log_zfactors))


# # In[7]:
# 
# # plot interpolant data
# 
# fig, ax = plt.subplots(1)
# # plot contours
# cm = ax.hexbin(f0s_ma, zs_ma, C=log_zf_ma, cmap='magma', xscale='log', yscale='log')
# 
# # add colorbar
# cb = plt.colorbar(cm, label=r'$\log z$-factor')
# cb.ax.tick_params(labelsize=18) 
# 
# plt.xlabel(r'$f_{\rm GW}$')
# plt.ylabel(r'$z$')
# ax.patch.set_facecolor("grey")
# 
# plt.xlim(10, 1E4)
# plt.ylim(1E-4, 1E3)
# 
# cm.set_rasterized(True)
# 
# plt.show()
# plt.close()


# In[10]:

# create z-factor interpolant

log_zf_interp = interp2d(f0s_ma, zs_ma, log_zf_ma, bounds_error=False, fill_value=np.inf)

import pickle
fpath = "logzfinterp.p"
pickle.dump(sampler.flatchain, open(fpath, "wb" ))

# ## Source frame
# 
# Load source-frame quantities from disk, or create if not found

# In[91]:

# dfpath = 'peak.hdf5'
# if os.path.exists(dfpath):
#     df_max = pd.read_hdf(dfpath, 'table', mode='r')
# else:
#     n_mass = 200
#     n_chi = 200
#     n_alpha = 1000
# 
#     distance = 5E6 * physics.PC_SI
# 
#     mbhs_array = np.logspace(0, 4, n_mass)
#     chis_array = np.linspace(1E-4, 1, n_chi)
# 
#     alphas = np.linspace(0, 0.5, n_alpha)
#     rows = []
#     for mbh in mbhs_array:
#         for chi in chis_array:
#             h0s, fgws = physics.h0_scalar_brito(mbh, alphas, chi_i=chi, d=distance)
#             hmax = np.nanmax(h0s)
#             fmax = fgws[h0s==hmax][0]
#             amax = alphas[h0s==hmax][0]
#             rows.append({'mbh': mbh, 'chi': chi, 'h0': hmax, 'fgw': fmax, 'alpha': amax})
#     df_max = pd.DataFrame(rows)
#     df_max.to_hdf(dfpath, 'table', mode='w')
# 
# 
# # In[6]:
# 
# 
# 
# 
# # ### 3G detectors
# 
# # In[ ]:
# 
# # load Voyager ASD
# vals = np.loadtxt('noise_curves_T1500293-v10/voyager.txt')
# freqs, asd_voy_array = vals[:,0], vals[:,1]
# 
# # create interpolant
# asdv = interp1d(freqs, asd_voy_array)
# 
# fmin, fmax = freqs.min(), freqs.max()
# 
# asd = asdv(freqs)
# 
# fig, ax = plt.subplots(1)
# ax.plot(freqs, asd)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(fmin, fmax)
# ax.set_xlabel(r'$f$ (Hz)')
# ax.set_ylabel(r'ASD')
# plt.show(fig)
# 
# print fmin, fmax
# 
# 
# # ## Peak emission
# 
# # In[ ]:
# 
# # # create an array of BH masses
# 
# # n_mass = 5000
# # n_chi = 1000
# # n_alpha = 2000
# 
# # distance = 5E6 * physics.PC_SI
# 
# # mbhs_array = np.linspace(1, 1E4, n_mass)
# # chis_array = np.linspace(1E-4, 1, n_chi)
# 
# # alphas = np.linspace(0, 0.5, n_alpha)
# 
# # df_list = []
# 
# 
# # dfpath = 'peak.hdf5'
# # if os.path.exists(dfpath):
# #     df_max = pd.read_hdf(dfpath, 'table', mode='r')
# # else:
# #     rows = []
# #     for mbh in mbhs_array:
# #         for chi in chis_array:
# #             h0s, fgws = physics.h0_scalar_brito(mbh, alphas, chi_i=chi, d=distance)
# #             hmax = np.nanmax(h0s)
# #             fmax = fgws[h0s==hmax][0]
# #             amax = alphas[h0s==hmax][0]
# #             rows.append({'mbh': mbh, 'chi': chi, 'h0': hmax, 'fgw': fmax, 'alpha': amax})
# #     df_max = pd.DataFrame(rows)
# #     df_max.to_hdf('peak.hdf5', 'table', mode='w')
# 
# 
# # In[ ]:
# 
# # h0_bound = df_max[(df_max['mbh']>10) & (df_max['mbh']<1E3)]['h0']
# # print h0_bound.min()
# # print h0_bound.max()
# 
