# Copyright (C) 2018 Maximiliano Isi
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy as np
from scipy.integrate import quad

class SpinWeightedSpheroidalHarmonic(object):
    def __init__(self, c, l, m, s):
        """ Compute spin-weighted-spheroidal harmonics (SWSHs).

        NOTE: treats the frequency as a *free* parameter (via `c`), i.e. does
        not assume the usual QNM frequencies for a given (n, l, m).

        Uses Leaver's method from:
        E.Leaver, "An analytic representation for the quasi-normal modes of
        Kerr black holes". Proc.R.Soc.Lond.A402 : 285 (1985) 

        See also details in:
        E. Berti, V. Cardoso and M. Casals, "Eigenvalues and eigenfunctions of
        spin-weighted spheroidal harmonics in four and higher dimensions",
        Phys. Rev. D73:024013 (2006); Erratum-ibid.D73:109902 (2006).
        [arXiv: gr-qc/0511111]
    
        Arguments
        ---------
        c : float
            SWSH argument, usually `c = a*w` for `a` the dimensionless BH spin
            and `w` the dimensionless wave angular frequency (0 <= a < 1).
    
        l : int
            Azimuthal quantum number.
    
        m : int
            Magnetic quantum number (-l <= m <= l).
    
        s : int
            Spin quantum number (+-2 for GWs).
        """
        # check arguments
        if l < 0:
            raise ValueError("invalid azimuthal number")
        if not ((-l <= m) and (m <= l)):
            raise ValueError("invalid magnetic number")
        # save main parameters
        self.c = c
        self.l = l
        self.m = m
        self.s = s
        # initialize results
        self._eigenvalue = None
        self.eigenvalue_properties = {
            'nmax': None,
            'converged': None,
            'mode': None
        }
        self._eigenfunction = None
        self.eigenfunction_properties = {
            'nmax': None
        }
        self._swsh = None
        # auxiliary quantities
        self._km = 0.5*np.abs(m - s)
        self._kp = 0.5*np.abs(m + s)
        # analytic expression for the eigenvalue Alm valid for c=0, and 
        # starting point of numerical computation otherwise.
        self._sep0 = l*(l + 1.) - s*(s + 1.) - (2.*m*s**2)*c/(l*(l + 1.))

    def __call__(self, theta, phi):
        return self.swsh(theta, phi)


    # ---------------------------------------------------------------------
    # EIGENVALUE COMPUTATION

    def _alpha(self, n):
        # Eq. (20a) in Leaver
        return -2*(n + 1)*(n + 2*self._km + 1)
    
    def _beta(self, n, sep):
        # Eq. (20b) in Leaver
        km = self._km
        kp = self._kp
        c = self.c
        s = self.s
        return n*(n - 1) + 2*n*(km + kp + 1 - 2*c) - (2*c*(2*km + s + 1) -\
               (km + kp)*(km + kp + 1)) - (c**2 + s*(s + 1) + sep)

    def _gamma(self, n):
        # Eq. (20c) in Leaver
        return 2*self.c*(n + self._km + self._kp + self.s)
    
    def _Leaver31ang(self, sep):
        rp = 0
        n = self.eigenvalue_properties['nmax']
        while(n>0):
            rp = self._gamma(n)/(self._beta(n, sep) - self._alpha(n)*rp)
            n += -1
        return rp

    def _Leaver33ang(self, sep):
        return self._beta(0, sep)/self._alpha(0) - self._Leaver31ang(sep)
        
    def _compute_eigenvalue(self, nmax=100, mode='root'):
        """ Numerically solve continued fraction equation to find Alm 
        eigenvalue (angular separation) as in Eq. (21) in Leaver.

        This is not necessary if `c=0`, as we have analytic solution.
        """
        self.eigenvalue_properties['nmax'] = nmax
        self.eigenvalue_properties['mode'] = mode
        if mode=='root':
            from scipy.optimize import root
            solution = root(self._Leaver33ang, self._sep0)
            sep = solution['x'][0]
            self.eigenvalue_properties['converged'] = solution['success']
        elif mode == 'newton':
            from scipy.optimize import newton
            sep = newton(self._Leaver33ang, self._sep0)
        elif mode == 'brentq':
            from scipy.optimize import brentq
            sep = brentq(self._Leaver33ang, self._sep0*0.9, self._sep0*1.1)
        else:
            # try whether `mode` is one of the methods accepted by `root` 
            from scipy.optimize import root
            sep = root(self._Leaver33ang, self._sep0, method=mode)
        return sep

    def compute_eigenvalue(self, *args, **kwargs):
        """ Compute angular separation eigenvalue Alm.
        
        If `c=0` returns analytic solution. Otherwise, numerically solve 
        continued fraction equation to find Alm as in Eq. (21) in Leaver.

        Arguments
        ---------
        nmax: int (optional)
            Maximum number of steps in the continued fraction recursion.
            (More for higher accuracy.)
        mode: str (optional)
            Numerical root-finding method (e.g. 'root', 'newton')
        """
        # compute eigenvalue (unless c=0, in which case use analytic expr)
        if self.c == 0:
            self._eigenvalue = self._sep0
        else:
            self._eigenvalue = self._compute_eigenvalue(*args, **kwargs)
        # update derivative products if they exist
        if self._swsh is not None:
            nmax_funct = self.eigenfunction_properties['nmax']
            self.compute_eigenfunction(nmax=nmax_funct)
            self._produce_swsh()
        return self._eigenvalue

    @property
    def eigenvalue(self):
        if self._eigenvalue is None:
            self.compute_eigenvalue()
        return self._eigenvalue

    # ---------------------------------------------------------------------
    # EIGENFUNCTION COMPUTATION
    def compute_eigenfunction(self, nmax=20):
        """ Compute spheroidal harmonic Slm(x), with `x = cos(theta)`, where
        `theta` is the polar angle.

        Uses recursive formula from Eq. (18) in Leaver [Eq. (2.5) in Berti].
        The function is normalized such that
            \int_{-1}^{1} |Slm(x)|^2 dx = 1

        NOTE: does not include azimuthal dependence (i.e. not spin-weighted).

        Arguments
        ---------
        nmax : int
            iteration depth for sum in Eq. (18), higher for greater accuracy.
            (default: 20)

        Returns
        -------
        slm_normed : funct
            normalized spheroidal harmonic.
        """
        self.eigenfunction_properties['nmax'] = nmax
        alpha = self._alpha
        beta = lambda n: self._beta(n, self.eigenvalue)
        gamma = self._gamma
        c = self.c
        # initialize coefficients
        an = np.zeros(nmax + 1)
        an[0] = 1
        an[1] = - beta(0) / alpha(0)
        # recursively compute the rest
        for i in range(1, nmax):
            an[i+1] = (-beta(i)*an[i] - gamma(i)*an[i-1]) / alpha(i)
        # compute eigenfunction using Leavers equation [x = cos(theta)]
        # Eq. (18) in Leaver [or Eq. (2.5) in Berti et al.]
        slm = lambda x: np.exp(c*x) * (1+x)**self._km * (1-x)**self._kp * \
                        np.sum(an*(1+x)**np.arange(nmax+1))
        # normalize
        norm_integrand = lambda x: 2*np.pi*slm(x)*np.conj(slm(x))
        norm = np.sqrt(quad(norm_integrand, -1., 1.)[0])
        slm_normed = lambda x: slm(x)/norm
        self._eigenfunction = slm_normed
        # update derivative products if they exist already
        if self._swsh is not None:
            self._produce_swsh()
        return slm_normed

    @property
    def eigenfunction(self):
        """ Spheroidal harmonic function.

        Arguments
        ---------
        x : float
            polar parameter `x = cos(theta)` for `theta` the polar angle.

        Returns
        -------
        Slm : float
            value of spheroidal harmonic at specified value of cos(theta).
        """
        if self._eigenfunction is None:
            self.compute_eigenfunction()
        return self._eigenfunction

    def sh(self, theta):
        """ Spheroidal harmonic (SH), Slm, as a function of polar angle.

        NOTE: this is equivalent to
            eigenfunction(np.cos(theta))

        Arguments
        ---------
        theta : float
            polar angle.

        Returns
        -------
        Slm : float
            value of SH evaluated at theta.
        """
        return self.eigenfunction(np.cos(theta))

    def _produce_swsh(self):
        self._swsh = lambda th, phi: np.exp(1j*self.m*phi)*self.sh(th)

    @property
    def swsh(self):
        """ Spin-weighted spheroidal harmonic (SWSH), Ylm.

        Arguments
        ---------
        theta : float
            polar angle.
        phi : float
            azimuthal angle.

        Returns
        -------
        Ylm : float
            value of SWSH evaluated at specified values theta and phi.
        """
        if self._swsh is None:
            self._produce_swsh()
        return self._swsh

