# MIT License

# Copyright (c) 2018 Maximiliano Isi, Richard Brito

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.special import factorial
from . import leavers
from .utilities import *

try:
    # if LAL is available, import constants for better accuracy (?)
    from lal import C_SI, G_SI, HBAR_SI, PC_SI, MSUN_SI, DAYSID_SI, YRSID_SI 
except ImportError:
    G_SI = 6.674e-11  # m^3 kg^-1 ss^-2
    C_SI = 299792458  # m ss^-1
    HBAR_SI = 1.054571e-34  # kg m^2 ss^-2*ss
    PC_SI = 3.08567758e16  # m
MSUN_SI = 1.9891e30  # kg

EV_SI = 1.602176565e-19  # kg (m ss^-1)^2    
MPL_SI = np.sqrt(HBAR_SI * C_SI / G_SI)


# ###########################################################################
# FUNCTIONS

def get_gw(alpha, lgw=2, l=1, m=1, nr=0, distance=1, times=False, **kwargs): 
    """ Get amplitude and frequency of GW emission for given alpha, and BH properties
    determined by kwargs.
    """
    cloud = BosonCloud.from_parameters(l, m, nr, alpha=alpha, **kwargs) 
    output = [cloud.gw(lgw).h0r/distance, cloud.gw(lgw).f]
    if times:
        output.append(cloud.number_growth_time)
        output.append(cloud.get_life_time(lgws=[lgw]))
    return output

def get_alpha_max(chi, m=1):
    # max SR alpha is just Obh_nat, see Eq. (7) in paper
    return 0.5*m*chi/(1 + np.sqrt(1 - chi**2))

def get_final_spin(alpha, m=1):
    return 4.*alpha*m / (4.*alpha**2 + m**2)

def qcd_axion_mass(fa):
    '''Mass of QCD axion as a function of symmetry breaking scale.

    See e.g. Eq. (2) in [arXiv:1004.3558]

    Arguments
    ---------
    fa: float
        Peccei-Quinn axion symmetry breaking scale in eV.

    Returns
    -------
    mua: float
        QCD axion mass in eV.
    '''
    return 6E-10 * (1E16 * 1E9 / fa)

def hydrogenic_level(n, alpha):
    ''' Return hydrogenic levels: (En / E0)

    Arguments
    ---------
    n: float
        *principal* quantum number `n = nr + l + 1`, for `nr` the radial
        and `l` the azimuthal quantum numbers.

    alpha: float
        Fine-structure constant.

    Returns
    -------
    level: float
        dimensionless level (En / E0).
    '''
    return 1 - 0.5 * (alpha/n)**2

# TODO: make these static methods of BlackHoleBoson?
def h0_scalar_brito(m_i, alpha, chi_i=0.9, d=1, l=1, m=1, lgw=None, mgw=None,
                    msun=True):
    """GW strain for scalar boson from Brito et al. [PhysRevD.96.064050]

    Default GW emission at (lgw=2*l, mgw=2*m).

    Arguments
    ---------
    m_i: float, array
        initial black-hole mass (in solar masses if `msun`, else kg).
    alpha: float, array
        fine-structure constant.
    chi_i: float, array
        initial black-hole spin parameter (dimensionless).
    d: float, array
        distance to source in meters (def. 1).
    msun: bool
        expect black-hole mass in solar masses.

    Returns
    -------
    h0: float, array
        gravitational-wave amplitude (strain).
    fgw:
        gravitational-wave frequency (Hz).
    """
    # dimensionless boson frequency for (l=1, m=1, nr=0)
    # same as `alpha*hydrogenic_level(1+1, alpha)` (see above)
    mwr = alpha*hydrogenic_level(l + 1, alpha)
    # dimensionless gravitational wave frequency
    mwgw = 2*mwr
    # black-hole final dimensionless spin, from Eq. (25)
    chi_f = 4*mwr/(1 + 4*mwr**2)
    # black-hole final mass, from Eq. (26)
    m_i = m_i*MSUN_SI if msun else m_i
    m_f = m_i*(1 - mwr*(chi_i - chi_f))
    m_c = m_i - m_f
    # dimensionfull GW frequency (rescaling `mwgw` by final BH mass)
    fgw = C_SI**3 * mwgw / (2*np.pi*G_SI*m_f)
    # numerical fit to GW emitted power for (lgw=2, mgw=2)
    zabs = Zabs.fast_fit(alpha, lgw=lgw or 2*l, mgw=mgw or 2*m)
    # Eq. (39) in Brito et al. with units restored
    h0 = G_SI*2*zabs*m_c / (d * C_SI**2 * mwgw**2)
    return h0, fgw

def h0_scalar_approx(alpha, f=None, m_bh=None, m_b=None, d=1,
                     msun=True, ev=True, chi=None):
    """ Analytic approximation to the peak BHB scalar strain from Arvanitaki+.

    Arguments
    ---------
    alpha: float
        gravitational fine-structure constant.
    f: float
        signal frequency
    """
    if f is not None:
        # `f_gw = 2*f_boson` implies:
        m_b = HBAR_SI*np.pi*f
        ev = True
    a = Alpha(alpha=alpha, m_bh=m_bh, m_b=m_b, msun=msun, ev=ev)
    if f is None:
        f = a.fgw
    h0 = 1E-24 * (a.alpha/0.1)**8 * (PC_SI*1E3/d) * (1E-12/a.m_b_ev)
    if chi is not None:
        # add spin correction
        h0 *= (chi - get_final_spin(a.alpha)) / 0.1
    return h0, f

def h0_vector_approx(alpha, f=None, m_bh=None, m_b=None, d=PC_SI*1E3, 
                     msun=True, ev=True):
    """ Analytic approximation to the peak BHB vector strain from Arvanitaki+.

    Arguments
    ---------
    alpha: float
        gravitational fine-structure constant.
    f: float
        signal frequency
    """
    if f is not None:
        # `f_gw = 2*f_boson` implies:
        m_b = HBAR_SI*np.pi*f
        ev = True
    a = Alpha(alpha=alpha, m_bh=m_bh, m_b=m_b, msun=msun, ev=ev)
    if f is None:
        f = a.m_b_ev / (HBAR_SI*np.pi)
    h0 = 5E-21 * (a.alpha/0.1)**6 * (PC_SI*1E3/d) * (1E-12/a.m_b_ev)
    return h0, f

def tinst_approx(m, alpha, chi, msun=True):
    if msun:
        m *= MSUN_SI
    t =  27. * DAYSID_SI * (m/(10*MSUN_SI)) * (0.1/alpha)**9 / chi
    return t

def tgw_approx(m, alpha, chi, msun=True):
    if msun:
        m *= MSUN_SI
    t = (6.5E4) * YRSID_SI * (m/(10*MSUN_SI)) * (0.1/alpha)**15 / chi
    return t
    
# TODO make these Cloud methods?
def get_sr_cond(x, jx, T0, epsilon, m=1):
    return jx - x**2 * (m*x - jx) * 4*T0**2/(m*epsilon)**2


# ###########################################################################
# CLASSES

class BlackHole(object):
    def __init__(self, mass, chi=None, a=None, j=None, msun=False):
        """ Black hole of given mass and spin.

        Can pass spin using the dimensionless parameter `chi`, or the 
        Kerr parameter (dimensions of length) `a`, or the angular momentum
        itself, `j`.
        
        Defaults to no spin (`chi=0`) and fails if more than one spin
        parameters are provided.

        Arguments
        ---------
        mass: float
            mass in kg (or in MSUN, if `msun` is True).

        chi: float
            dimensionless spin, `chi = (c^2/G)(a/M)`, in (0, 1) [opt].

        a: float
            Kerr parameter, `a = J/(Mc)`, in meters [opt].
        
        j: float
            black-hole angular momentum, `J`, in SI units [opt].

        msun: bool
            whether `mass` is given in solar masses (def. False).
        """
        # MASS
        if msun:
            self.mass = mass * MSUN_SI
            self.mass_msun = mass
            mass = self.mass
        else:
            self.mass = mass
            self.mass_msun = mass / MSUN_SI
        # LENGTHSCALE
        self.rg = G_SI * mass / C_SI**2
        self.rs = 2 * self.rg
        # TIMESCALE
        self.tg = self.rg / C_SI
        # SPIN
        if sum([int(spin_param is None) for spin_param in [chi, a, j]]) < 2:
            raise ValueError("can only take one spin parameter: chi, a, or J.")
        elif a is not None:
            chi = a / self.rg
        elif j is not None:
            chi = j / (self.mass * C_SI * self.rg)
        elif chi is None:
            # no spin provided, default to Schwarzschild
            chi = 0
        self.chi = chi
        self.a = self.rg * self.chi
        self.angular_momentum = self.mass * C_SI * self.a
        # RADII in natural units (G=M=c=1)
        self.rp_natural = 1 + np.sqrt(1 - self.chi**2)
        self.rm_natural = 1 - np.sqrt(1 - self.chi**2)
        self.rp = self.rg * self.rp_natural
        self.rm = self.rg * self.rm_natural
        # ANGULAR VELOCITY
        self.omega_horizon_natural = chi / (2. * self.rp_natural)
        self.omega_horizon = chi * C_SI / (2. * self.rp)  # = oh_nat * c / rg
        # AREA
        self.area = 4*np.pi*(self.rp**2 + self.a**2)
        self.area_natural = 8 * np.pi * self.rp_natural
        # SR-SPECIFIC
        self._h0r_fits = {}

    def sigma(self, r, theta):
        """ Kerr auxiliary length Sigma, function of radius and polar angle
        in Boyer Lindquist coordinates.
        
        Arguments
        ---------
        r : float
            BL radius (m)
        theta : float
            BL polar angle (rad).

        Returns
        -------
        sigma : float
            sigma (m).
        """
        return r**2 + self.a**2 * np.cos(theta)**2

    def delta(self, r):
        """ Kerr auxiliary length Delta, function of radius 
        in Boyer Lindquist coordinates.
        
        Arguments
        ---------
        r : float
            BL radius (m)

        Returns
        -------
        delta : float
            delta (m).
        """
        return r**2 - self.rs * r + self.a**2

    def omega(self, r, theta):
        """ Frame dragging angular frequency (rad) for Boyer-Lindquist radius
        and polar angle.
        """
        num = C_SI * self.rs * r * self.a
        den = self.sigma(r, theta)*(r**2 + self.a**2) + \
              self.rs * r * self.a**2 * np.sin(theta)**2
        omega = num / den
        return omega
    
    def ergoshphere(self, theta, natural=False):
        """Innner and outer ergosphere radii for given polar angle.
        """
        if natural:
            rs = 2
            a = self.chi
        else:
            rs = self.rs
            a = self.a
        rEp = 0.5 * (rs + np.sqrt(rs**2 - 4* a**2 * np.cos(theta)**2))
        rEm = 0.5 * (rs - np.sqrt(rs**2 - 4* a**2 * np.cos(theta)**2))
        return rEp, rEm

    # --------------------------------------------------------------------
    # UTILITIES

    def scan_alphas(self, l=1, m=1, nr=0, delta_alpha=0.001, alpha_min=0.001,
                    alpha_max=None, lgw=None, verbose=False, ncpus=1, **kwargs):
        alpha_max = alpha_max or get_alpha_max(self.chi, m=m)
        alphas = np.arange(alpha_min, alpha_max, delta_alpha)
        h0rs, fgws = [], []
        iterable = alphas
        # determine whether to show progress bar
        if verbose:
            try:
                from tqdm import tqdm
                iterable = tqdm(alphas)
            except ImportError:
                print("WARNING: need tqdm for verbosity.")
        # iterate
        for alpha in iterable:
            cloud = BosonCloud.from_parameters(l, m, nr, m_bh=self.mass_msun,
                                               chi_bh=self.chi, alpha=alpha,
                                               **kwargs)
            h0rs.append(cloud.gw(lgw).h0r)
            fgws.append(cloud.gw(lgw).f)
        return np.array(h0rs), np.array(fgws), alphas

    def best_alpha(self, *args, **kwargs):
        h0rs, fgws, alphas = self.scan_alphas(*args, **kwargs)
        h0r_max = h0rs.max()
        i_max = np.where(h0rs==h0r_max)[0][0]
        return h0r_max, fgws[i_max], alphas[i_max]

    # NOTE: currently, this is essentially just refitting Zabs, which is a bit
    # dumb... however, the total mass of the cloud is also used, which might
    # be computed numerically in the future to make the code more precise---so 
    # leave this as is.
    def h0r_fit(self, f, **kwargs):
        l = int(kwargs.pop('l', 1))
        m = int(kwargs.pop('m', 1))
        lgw = int(kwargs.pop('lgw', 2*l))
        if (l, m, lgw) not in self._h0r_fits:
            from scipy.interpolate import interp1d
            h0rs, fgws, _ = self.scan_alphas(l=l, m=m, lgw=lgw, **kwargs)
            self._h0r_fits[(l, m, lgw)] = interp1d(fgws, h0rs, fill_value=0,
                                                   bounds_error=False)
        return self._h0r_fits[(l, m, lgw)](f)

    def fgw(self, alpha=None, l=1, nr=0, m_b=None, ev=True):
        a = Alpha(m_bh=self.mass_msun, alpha=alpha, m_b=m_b, ev=ev)
        level_correction = hydrogenic_level(l+nr+1, a.alpha)
        return level_correction * a.fgw


class Boson(object):
    def __init__(self, mass, spin=0, ev=False):
        """ A boson field of given mass and spin.

        Arguments
        ---------
        mass: float
            mass in kg (or in eV, if `ev`).
        spin: int
            spin-weight (0, 1, 2).
        ev: bool
            mass provided in eV.
        """
        if ev:
            # the `mass` parameter was actually the energy
            self.energy_ev = mass
            self.energy = mass * EV_SI
            self.mass = self.energy / C_SI**2
            mass = self.mass
        else:
            self.mass = mass
            self.energy = mass * C_SI**2
            self.energy_ev = self.energy / EV_SI 
            # this last quantity is called `mu` by Arvanitaki et al.
        self.spin = spin
        self.omega = self.energy / HBAR_SI
        self.reduced_compton_wavelength = HBAR_SI / (mass*C_SI)
        self.compton_wavelength = 2*np.pi*self.reduced_compton_wavelength
        # Other quantities
        self.mu_brito = C_SI * self.mass / HBAR_SI


class Alpha(object):
    def __init__(self, m_bh=None, m_b=None, alpha=None, msun=True, ev=True,
                 tolerance=1E-10):
        """ Gravitational fine-structure constant.
        
        Can be initialized with any two of (m_bh, m_b, alpha) to compute the
        third quantity. If the three numbers are provided, will check they
        are consistent (and fail with `ValueError` if not).

        Arguments
        ---------
        m_bh: float
            black-hole mass (in MSUN, or SI if `msun` is False).
        m_b: float
            boson rest mass (eV, or SI if `ev` is False).
        alpha: float
            gravitational fine-structure constant (dimensionless).
        msun: bool
            BH mass provided in solar masses, rather than SI (def True).
        ev: bool
            boson mass provided in eV, rather than SI (def True).

        """
        if m_bh is not None and msun:
            m_bh = MSUN_SI*m_bh
        if m_b is not None and ev:
            m_b = m_b * EV_SI / C_SI**2
        self.m_bh = m_bh
        self.m_b = m_b
        if all([p is not None for p in [m_bh, m_b, alpha]]):
            # check consistency
            alpha_new = self.compute(self.m_bh, self.m_b)
            if abs(alpha - alpha_new) > tolerance:
                raise ValueError("alpha incompatible with BH & boson masses.")
        elif all([p is not None for p in [m_bh, alpha]]):
            # compute boson mass
            self.m_b = HBAR_SI * C_SI * alpha / (G_SI * self.m_bh)
        elif all([p is not None for p in [m_b, alpha]]):
            # compute BH mass
            self.m_bh = HBAR_SI * C_SI * alpha / (G_SI * self.m_b)
        self.alpha = alpha if alpha is not None else self.compute(self.m_bh, self.m_b)
        
    @cached_property
    def m_bh_msun(self):
        return self.m_bh / MSUN_SI

    @cached_property
    def m_b_ev(self):
        return self.m_b * C_SI**2 / EV_SI

    @cached_property
    def fgw(self):
        return self.m_b_ev * EV_SI / (HBAR_SI*np.pi)

    @staticmethod
    def compute(m_bh, m_b):
        return G_SI * m_bh * m_b / (HBAR_SI * C_SI)


class BlackHoleBoson(object):
    def __init__(self, bh, boson):
        """ System composed of a black-hole and a boson.

        To create from parameters use `.from_parameters()` class method.

        Arguments
        ---------
        bh: BlackHole
            black-hole instance.
        boson: Boson
            boson instance.
        """
        self.bh = bh
        self.boson = boson
        # Fine-structure constant `G M m / (hbar c) = rg / lambda_bar_c`
        self.alpha = self.bh.rg / self.boson.reduced_compton_wavelength
        self.clouds = {}
        self._has_waveform = False

    # --------------------------------------------------------------------
    # CLASS METHODS

    @classmethod
    def from_parameters(cls, **kwargs):
        """ Create black-hole boson system from parameters.

        Can pass any two of the mass parameters (m_bh, m_b, alpha) and
        any one of the BH spin parameters (chi_bh, a_bh, j_bh).

        The BH mass is assumed to be in units of MSUN, unless `msun=False` in
        which case SI units are expected.

        The boson mass is assumed to be in units of eV, unless `ev=False` in
        which case SI units are expected.

        The spin parameters (a_bh, j_bh) are always expected in SI units, while
        `chi_bh` is dimensionless.

        Arguments
        ---------
        m_bh: float
            black-hole mass (in MSUN, or SI if `msun` is False).
        chi_bh: float
            dimensionless BH spin.
        m_b: float
            boson rest mass (eV, or SI if `ev` is False).
        boson_spin: int
            spin of boson field (0 for scalar, 1 for vector).
        alpha: float
            fine-structure constant.
        msun: bool
            BH mass provided in solar masses, rather than SI (def True).
        ev: bool
            boson mass provided in eV, rather than SI (def True).
        """
        s_b = kwargs.pop('boson_spin', 0)
        bh_spin_kwargs = {k.strip('_bh'): kwargs.pop(k, None) for k in
                          ['chi_bh', 'a_bh', 'j_bh']}
        alpha = Alpha(**kwargs)
        bh = BlackHole(alpha.m_bh, msun=False, **bh_spin_kwargs)
        boson = Boson(alpha.m_b, spin=s_b, ev=False)
        return cls(bh, boson)

    # --------------------------------------------------------------------
    # UTILITIES

    def _sr_factor(self, m):
        """ Super-radiance term for magnetic quantum number `m`.
        Super-radiance takes place if this value is non-negative.

        Arguments
        ---------
        m: int
            magnetic quantum number.

        Returns
        -------
        sr_factor: float
            m * Omega_bh - omega_boson  (dimensionless)
        """
        sr_factor_natural = m*self.bh.chi - 2*self.alpha*self.bh.rp_natural
        # Note `sr_factor_natural` is dimensionless because:
        #     sr_factor_natural = (2 rp / c)*sr_factor
        # with `sr_factor` the dimensionful factor (rad/s):
        #     sr_factor = m*self.bh.omega_horizon - self.boson.omega
        return sr_factor_natural

    def is_superradiant(self, m):
        return self._sr_factor(m) >= 0

    # --------------------------------------------------------------------
    # FREQUENCY

    def _level(self, n):
        return hydrogenic_level(n, self.alpha)

    def level_energy(self, n, units='ev'):
        ''' Return real part of hydrogenic energy eigenvalues.

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.
        '''
        units = units.lower()
        if units == 'ev':
            mu = self.boson.energy_ev
        elif units == 'si':
            mu = self.boson.energy
        elif units == 'none':
            mu = 1
        return mu * self._level(n)

    def level_omega_natural(self, n):
        ''' Return dimensionless hydrogenic eigen-frequencies.

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.

        Returns
        -------
        omega_dimless: float
            dimensionless angular frequency of nth eigenmode in rad/s.
        '''
        return self.alpha * self.level_energy(n, units='none')

    def level_omega_re(self, n, method='hydrogen'):
        ''' Return real part of energy eigen-frequencies in rad/s.

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.

        Returns
        -------
        omega: float
            angular frequency of nth eigenmode in rad/s.
        '''
        if method=='hydrogen':
            w = self.boson.omega * self._level(n)
        elif method=='numeric':
            NotImplementedError("numeric frequency solutuions unavailable.")
        else:
            raise ValueError("unrecognized method %r." % method)
        return w

    def level_frequency(self, n, *args, **kwargs):
        ''' Return real part of hydrogenic energy eigen-frequencies in Hz.

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.

        Returns
        -------
        frequency: float
            frequency of n-th eigenmode in Hz.
        '''
        return self.level_omega_re(n, *args, **kwargs) / (2*np.pi)

    # --------------------------------------------------------------------
    # GROWTH-RATE

    def _clmn(self, l, m, nr):
        """ Factor in computation of imaginary frequency.

        gamma factor of Eq. (28) in Detweiler (1980), or equivalently
        Clmn factor of Eq. (18) in Arvanitaki & Dubovsky (2011).

        Without the spin correction, this gives a prefactor of 1/24.
        for the (1, 1, 0) scalar state.

        Arguments
        ---------
        l: int
            azimuthal quantum number.
        m: int
            magnetic quantum number.
        nr: int
            radial quantum number.

        Returns
        -------
        clmn: float
            gamma/cnlm factor.
        """
        chi = self.bh.chi
        sr_factor = self._sr_factor(m)
        # Factor 1
        f1 = factorial(2*l + nr + 1) * 2**(4*l + 2)
        f1 /= factorial(nr) * (nr + l + 1)**(2*l + 4) 
        # Factor 2
        f2 = (factorial(l) / (factorial(2*l)*factorial(2*l + 1)))**2
        # Factor 3
        js = np.arange(1, l+1)
        factors = js**2 * (1 - chi**2) + sr_factor**2
        # i.e. \[ j^2 (1-a^2/rg^2) + 4 rp^2 (m wp - mu)^2 \],
        # with Arvanitaki+'s defns: `wp = omega_bh_natural / rg`, `alpha=mu rg`
        f3 = np.product(factors)
        return f1*f2*f3

    def level_omega_im(self, l, m, nr, method='detweiler'):
        ''' Return imag part of hydrogenic energy eigen-frequencies in rad/s.
        Importantly, this *half* of the occupancy number grow rate.
        
        Can be computed in different regimes:

        'detweiler'
            This is from an analytic approximation in the nonrelativistic limit
            `alpha << 1`. References:
                Eq. (28) in Detweiler (1980)
                Eq. (18) in Arvanitaki & Dubovsky (2011)
                Eq. (8) in Brito et al. (2017)

        'zouros'
            This is from an analytic approximation in the WKB regime
            `alpha >> 1`. References:
                Zouros & Eardley (1979)
                Eq. (27) in Arvanitaki & Dubovsky (2011)
        'dolan'
            Numerical method for the intermediate regime `alpha ~ 1`.
                Dolan (2007)

        Arguments
        ---------
        l: int
            azimuthal quantum number.
        m: int
            magnetic quantum number.
        nr: int
            radial quantum number.

        Returns
        -------
        omega: float
            angular frequency of nth eigenmode in rad/s.
        '''
        method = method.lower()
        n = nr + l + 1
        w0 = self.boson.omega
        a = self.alpha
        if method == 'detweiler':
            # this agrees with Eq. (8) in arXiv:1706.06311
            # sr = 2 rp^2 (m Obh - w) / c
            sr = self._sr_factor(m)
            # multiply by 0.5 to get field-amplitude growth
            # that way, here clmn = 1/48. for (1, 1, 0)
            clmn = 0.5*self._clmn(l, m, nr)
            omega_im = w0 * a**(4*l +4) * sr * clmn
        elif method == 'zouros':
            number = 2. - np.sqrt(2)
            omega_im = 1E-7 * (C_SI/self.bh.rg) * np.exp(-2*np.pi*a*number)
        elif method == 'dolan':
            raise NotImplementedError("method 'dolan' not implented yet.")
        else:
            e = "unrecognized method %r (valid options are: 'detweiler')"\
                % method
            raise ValueError(e)
        return omega_im

    def fgw(self, n):
        ''' Returns main gravitational-wave frequency for level `n`.

        fgw = 2*fre = 2*(wre/2pi) = wre/pi

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.

        Returns
        -------
        fgw: float
            GW frequency in Hz.
        '''
        return self.level_omega_re(n) / np.pi

    def max_growth_rate(self, l_min=0, nr_min=0, l_max=5, nr_max=5, **kwargs): 
        """ Search for level with fastest superradiant growth rate.

        Arguments
        ---------
        l_min: int
            minimum azimuthal quantum number (def. 0).
        nr_min: int
            minimum radial quantum number (def. 0).
        l_max: int
            maximum azimuthal quantum number (def. 5).
        nr_max: int
            maximum radial quantum number (def. 5).

        Returns
        -------
        l_best: int
            azimuthal quantum number of fastest-growing level.
        m_best: int
            magnetic quantum number of fastest-growing level.
        nr_best: int
            radial quantum number of fastest-growing level.
        rate_best: float
            growth rate (Hz) of fastest-growing level.
        """
        # Given the way the SR rate scales with `l`, we want the smallest `l`,
        # with highest `m` that satisfies SR condition.
        # TODO: is this true for vectors?
        if self.alpha > 0.5:
            # cannot satisfy SR condition
            return 0, 0, 0, 0
        else:
            # TODO: this can be optimized
            ls, ms, nrs, rates = [], [], [], []
            for l in range(l_min, l_max+1):
                for m in range(0, l+1):
                    for nr in range(nr_min, nr_max+1):
                        rate = self.level_omega_im(l, m, nr, **kwargs)
                        ls.append(l)
                        ms.append(m)
                        nrs.append(nr)
                        rates.append(rate)
            i = rates.index(max(rates))
            return ls[i], ms[i], nrs[i], rates[i]

    # --------------------------------------------------------------------
    # CLOUDS

    def _add_cloud(self, l, m, nr):
        cloud = BosonCloud(self, l, m, nr)
        self.clouds[(int(l), int(m), int(nr))] = cloud

    def best_cloud(self, *args, **kwargs):
        """ Retrieve (or create) cloud with fastest SR growth-rate.

        All arguments passed to growth-rate function.

        Returns
        -------
        cloud: Cloud
            cloud object for given quantum numbers.
        """
        l, m, nr, _ = self.max_growth_rate(*args, **kwargs)
        key = (int(l), int(m), int(nr))
        if key not in self.clouds:
            self._add_cloud(*key)
        return self.clouds[key]

    def cloud(self, l, m, nr, update_waveform=True):
        """ Retrieve (or create) cloud of given level.

        Arguments
        ---------
        l: int
            azimuthal quantum number.
        m: int
            magnetic quantum number.
        nr: int
            radial quantum number.

        Returns
        -------
        cloud: Cloud
            cloud object for given quantum numbers.
        """
        key = (int(l), int(m), int(nr))
        if key not in self.clouds:
            self._add_cloud(*key)
            if self._has_waveform and update_waveform:
                self.waveform = self.create_waveform()
        return self.clouds[key]

    # --------------------------------------------------------------------
    # GWS

    def create_waveform(self, lmns=None, lgw_max=None): 
        """ Produce waveform (hp, hc) by adding contributions from clouds with 
        quantum numners `lmns`, up to GW azimuthal number `lgw_max`.

        If no `lmns` are specified, will use all clouds present to produce waform.
        If no `lgw_max` is specified, will only use the minimum (`l_gw=2*l_cloud`)
        for each cloud.

        Arguments
        ---------
        lmns: list
            optional list of tuples with cloud quantum numbers, 
            e.g. [(1,1,0), (2,1,2), ...] (def. all precomputed clouds).
        lgw_max: int
            maximum GW azimuthal number (2*l_cloud <= l_gw <= lgw_max).

        Returns
        -------
        hp: function
            plus polarization (function of theta, phi and time)
        hc: function
            cross polarization (function of theta, phi and time)
        """
        if lmns is None:
            lmns = self.clouds.keys()
        hps, hcs = [], []
        # loop over cloud lmn's
        for lmn in lmns:
            c = self.cloud(*lmn)
            # loop over GW l's if specified, otherwise just set `l_gw=2*l_c`
            lgw_max_loc = lgw_max or 2*c.l
            for lgw in np.arange(2*c.l, lgw_max_loc+1):
                hps.append(c.gw(lgw).hp)
                hcs.append(c.gw(lgw).hc)
        if len(hps) == 0:
            raise ValueError("no matching clouds to produce waveform!")
        def hp(*args, **kwargs):
            return np.sum([hp(*args, **kwargs) for hp in hps])
        def hc(*args, **kwargs):
            return np.sum([hp(*args, **kwargs) for hp in hps])
        return hp, hc
            
    @cached_property
    def waveform(self):
        wf = self.create_waveform()
        self._has_waveform = True
        return wf

    def hp(self, *args, **kwargs):
        r = kwargs.pop('r', 1)
        return self.waveform[0](*args, **kwargs) / r

    def hc(self, *args, **kwargs):
        r = kwargs.pop('r', 1)
        return self.waveform[1](*args, **kwargs) / r


class BosonCloud(object):
    def __init__(self, bhb, l, m, nr, evolve=True, evolve_params=None, from_final=False):
        """ Boson cloud around a black hole, corresponding to single level.

        Arguments
        ---------
        bhb: BlackHoleBoson
            black-hole-boson object
        l: int
            azimuthal quantum number.
        m: int
            magnetic quantum number.
        nr: int
            radial quantum number.
        evolve: bool
            whether to compute final BH mass and spin numerically or approximate.
        from_final: bool
            interpret `bhb` as the post-SR, rahter than pre, black-hole-boson system.
        """
        # check `bhb` is of the right type
        try:
            bhb.boson.reduced_compton_wavelength
            bhb.bh.rg
        except AttributeError:
            raise ValueError("'bhb' must be `BlackHoleBoson` instance, not %r"
                             % type(boson))
        if from_final:
            self._bh_initial = None
            self._bhb_initial = None
            self._bhb_final = bhb
            self._bh_final = self.bhb_final.bh
        else:
            self._bhb_initial = bhb
            self._bh_initial = self.bhb_initial.bh
            self._bh_final = None
            self._bhb_final = None
        # check consistency of quantum numbers
        if (0 <= l) & (np.abs(m) <= l) & (0 <= nr) & isinstance(nr*l*m, int):
            self.n = nr + l + 1  # principal quantum number
            self.nr = nr
            self.l = l
            self.m = m
        else:
            raise ValueError("invalid quantum numbers (l, m, nr) = (%r, %r, %r)"
                             % (l, m, nr))
        # set cloud properties
        self.age = 0  # age is the actual age of the cloud after evolving
        self._growth_time = None
        self._life_time = None  # lifetime is the characteristic duration of the cloud
        self._is_superradiant = None
        self._mass = None
        self._mass_msun = None
        # set gravitational-wave properties
        self._zabs = {}
        self._gws = {}
        self._fgw = None
        # solve DEs for cloud evolution, or approximate final values
        self.evolve = evolve  
        self._evolve_params = evolve_params or {'y_0': 1E-8}

    # --------------------------------------------------------------------
    # CLASS METHODS

    @classmethod
    def from_parameters(cls, l, m, nr, evolve=True, evolve_params=None,
                        from_final=False, **kwargs):
        bhb = BlackHoleBoson.from_parameters(**kwargs)
        return cls(bhb, l, m, nr, evolve=evolve, evolve_params=evolve_params,
                   from_final=from_final)

    # TODO: optimize
    def _backtrack_instability(self, **kwargs):
        # get guesses for initial BH parameters if provided
        m_i_0 = kwargs.pop('m_i_0', 1.11*self.bhb_final.bh.mass)
        m_i_frac_0 = m_i_0 / self.bhb_final.bh.mass
        chi_i_0 = kwargs.pop('chi_i_0', 1.)
        initial_guess = np.array([m_i_frac_0, chi_i_0])
        def evolve(mfrac_chi):
            m_i_frac, chi_i = mfrac_chi
            bh_i = BlackHole(m_i_frac*self.bhb_final.bh.mass, chi=chi_i)
            bhb_i = BlackHoleBoson(bh_i, self.bhb_final.boson)
            cloud = BosonCloud(bhb_i, self.l, self.m, self.nr, **kwargs)
            m_f = cloud.bhb_final.bh.mass
            chi_f = cloud.bhb_final.bh.chi
            dm_frac = (m_f - self.bhb_final.bh.mass)/self.bhb_final.bh.mass
            dchi = chi_f - self.bhb_final.bh.chi
            return np.sqrt(dm_frac**2 + dchi**2)
        from scipy.optimize import minimize
        res = minimize(evolve, initial_guess, method='L-BFGS-B',
                       bounds=[(1., 1.4), (self.bhb_final.bh.chi, 0.99999)])
        return res

    def _evolve_instability(self, y_0=1E-8, dtau=None, max_steps=1E6, tolerance=1e-3,
                            dtau_adapt_frac=0.1, dtau_adapt_thresh=0.05,
                            m_accretion_rate=0, j_accretion_rate=0, bhb_0=None):
        """ Solve cloud evolution equations, ignoring cloud angular momentum and GW power.
    
        See documentation for `evolve_bhb` for more information.
    
        Arguments
        ---------
    
        Returns
        -------
    
        """
        bhb_0 = bhb_0 or self.bhb_initial
        l = self.l
        m = self.m
        nr = self.nr
        # initial mass
        M0 = bhb_0.bh.mass
        T0 = G_SI*M0/C_SI**3
        # time step
        epsilon = 1./bhb_0.boson.omega
        dtau = dtau or self.amplitude_growth_time/1000.
        # other dimensionfull constants
        alpha = M0
        beta = epsilon*M0*C_SI**2
        gamma = M0*C_SI**2 / epsilon
        # initial state
        x_0 = 1
        jx_0 = bhb_0.bh.angular_momentum / beta  # same as alpha_0/2
        n = l + nr + 1
        wR_0 = bhb_0.level_omega_re(n) * epsilon
        wI_0 = bhb_0.level_omega_im(l, m, nr) * epsilon
        # dimensionless accretion rates
        dimless_m_accretion_rate = m_accretion_rate * C_SI**2 / gamma
        dimless_j_accretion_rate = j_accretion_rate * epsilon / beta
        # initialize arrays
        xs = [x_0]
        ys = [y_0]
        jxs = [jx_0]
        inv_wRs = [1./wR_0]
        wIs = [wI_0]
        sr_conds = [get_sr_cond(x_0, jx_0, T0, epsilon, m=m)]
        times = [0]
        if not bhb_0.is_superradiant(m):
            # print "WARNING: initial BHB not superradiant for m = %i (alpha=%.6f)" \
            #       % (m, bhb_0.alpha)
            bhb_new = bhb_0
        else:
            # evolve
            for i in range(int(max_steps)):
                # update x & j
                dx = - 2*wIs[i]*ys[i]*dtau
                x_new = xs[i] + dx + dimless_m_accretion_rate*dtau
                y_new = ys[i] - dx 
                # update jx & jy
                jx_new = jxs[i] - 2*m*wIs[i]*inv_wRs[i]*ys[i]*dtau +\
                         dimless_j_accretion_rate*dtau
                # update w's
                # TODO: optimize this to not rely on BH and BHB objects
                bh = BlackHole(x_new*alpha, j=jx_new*beta)
                bhb_new = BlackHoleBoson(bh, bhb_0.boson)
                inv_wR_new = 1./ (bhb_new.level_omega_re(n) * epsilon)
                wI_new = bhb_new.level_omega_im(l, m, nr) * epsilon
                # append
                xs.append(x_new)
                ys.append(y_new)
                jxs.append(jx_new)
                inv_wRs.append(inv_wR_new)
                wIs.append(wI_new)
                times.append(times[i] + dtau)
                # compute SR condition
                sr_cond = get_sr_cond(x_new, jx_new, T0, epsilon, m=m)
                sr_conds.append(sr_cond)
                # decide whether to terminate
                frac_decrease = (sr_conds[i] - sr_cond)/float(sr_cond)
                change_ratio = float(sr_cond)# / sr_conds[0]
                if frac_decrease < tolerance and (change_ratio < tolerance):
                    break
                elif frac_decrease < dtau_adapt_thresh:
                    # adapt time step
                    dtau *= 1. + dtau_adapt_frac
                elif frac_decrease > dtau_adapt_thresh:
                    # adapt time step
                    dtau *= 1. - dtau_adapt_frac
        cs = tuple([np.array(l) for l in [xs, jxs, ys, inv_wRs, wIs, sr_conds, times]])
        return bhb_new, cs

    # --------------------------------------------------------------------
    # PROPERTIES

    @property
    def is_superradiant(self):
        """ Indicates whether this energy level (l, m, n) is superradiant.
        """
        if self._is_superradiant is None:
            self._is_superradiant = self.bhb_initial.is_superradiant(self.m)
        return self._is_superradiant

    @property
    def amplitude_growth_time(self):
        """ Field amplitude superradiant-instability timescale: `1/Im(omega)`.
        """
        if self._growth_time is None:
            self._growth_time = 1./self.bhb_initial.level_omega_im(self.l, self.m, 
                                                                self.nr)
        return self._growth_time
    
    @property
    def number_growth_time(self):
        """ Occupation number superradiant-instability timescale: `0.5/Im(omega)`.
        """
        return 2.*self.amplitude_growth_time

    @property
    def bh_initial(self):
        """ Black-hole before start of superradiant cloud growth.
        """
        if self._bh_initial is None:
            if self.evolve:
                self._bh_initial = self.bhb_initial.bh
            else:
                # initial BH angular momentum from Eq. (25) in Brito et al.
                # approximating the final alpha with the initial alpha
                rg = self.bh_final.rg
                w = self.bhb_final.level_omega_re(self.n)
                chi_f = self.bhb_final.bh.chi
                m_i = (2*(C_SI*rg*w + np.sqrt((C_SI*rg*w)**2 - 
                                              (chif*C_SI*rg*w)**2)))/\
                      (chif*C_SI**2)
                # initial BH spin from Eq. (26) in Brito et al.
                w_nat = self.bhb_final.level_omega_natural(self.n)
                chi_i = (m_i + chi_f*m_i*w_nat - m_f) / (m_i*w_nat)
                self._bh_initial = BlackHole(m_i, chi_i) 
        return self._bh_initial

    @property
    def bhb_initial(self):
        """ Black-hole-boson before start of superradiant cloud growth.
        """
        if self._bhb_initial is None:
            if self.evolve:
                self._bhb_initial, _ = self._backtrack_instability(evolve_params=self._evolve_params)
                # note that the value of y_0 does not affect final state,
                # but a higher value means fewer steps
            else:
                self._bhb_initial = BlackHoleBoson(self.bh_initial,
                                                   self.bhb_final.boson)
        return self._bhb_initial


    @property
    def bh_final(self):
        """ Black-hole left at the end of superradiant cloud growth.
        """
        if self._bh_final is None:
            if self.evolve:
                self._bh_final = self.bhb_final.bh
            else:
                # final BH angular momentum from Eq. (25) in Brito et al.
                # approximating the final alpha with the initial alpha
                rg = self.bh_initial.rg
                w = self.bhb_initial.level_omega_re(self.n)
                chi_f = 4*C_SI*self.m*rg*w / ((C_SI*self.m)**2 + 4*(rg*w)**2)
                # final BH mass from Eq. (26) in Brito et al.
                w_nat = self.bhb_initial.level_omega_natural(self.n)
                m_f = self.bh_initial.mass*(1 - w_nat*(self.bh_initial.chi - chi_f))
                self._bh_final = BlackHole(m_f, chi_f) 
        return self._bh_final

    @property
    def bhb_final(self):
        """ Black-hole-boson left at the end of superradiant cloud growth.
        """
        if self._bhb_final is None:
            if self.evolve:
                self._bhb_final, _ = self._evolve_instability(**self._evolve_params)
                # note that the value of y_0 does not affect final state,
                # but a higher value means fewer steps
            else:
                self._bhb_final = BlackHoleBoson(self.bh_final,
                                                 self.bhb_initial.boson)
        return self._bhb_final

    @property
    def mass(self):
        """ Maximum cloud mass (kg), reached at end of superradiant stage.
        """
        if self._mass is None:
            self._mass = self.bh_initial.mass - self.bh_final.mass
            self._mass_msun = self._mass / MSUN_SI
        return self._mass

    @property
    def mass_msun(self):
        """ Maximum cloud mass (MSUN), reached at end of superradiant stage.
        """
        if self._mass_msun is None:
            self.mass
        return self._mass_msun

    # --------------------------------------------------------------------
    # GW PROPERTIES

    @property
    def fgw(self):
        """ Gravitational-wave frequency (Hz).
        """
        if self._fgw is None:
            self._fgw = 2.*self.bhb_final.level_frequency(self.n)
        return self._fgw

    def zabs(self, lgw=None):
        lgw = lgw or 2*self.l
        # the GW is allowed to have any l_gw >= 2*l_cloud, but only m_gw = 2*m_cloud
        if lgw not in self._zabs:
            self._zabs[lgw] = Zabs(lgw, 2*self.m)(self.bhb_final.alpha)
        return self._zabs[lgw]

    def gw(self, lgw=None):
        lgw = lgw or 2*self.l
        if lgw not in self._gws:
            if lgw < 2*self.l:
                raise ValueError("Must have `l_gw >= 2*l_cloud = %i`."
                                 % 2*self.l)
            # intrinsic amplitude, 1m away from the source (`h0r = h0*r`).
            wgw = 2*np.pi*self.fgw
            m_bh = self.bh_final.mass
            m_c = self.mass
            h0r = (C_SI**4/G_SI) * 2.*self.zabs(lgw=lgw)*m_c / (wgw*m_bh)**2
            # SWSH spin parameter: dimensionless (spin x omega_gw)
            c = self.bh_final.chi * 2*np.pi*self.fgw * self.bh_final.tg
            self._gws[lgw] = GravitationalWaveMode(self.fgw, c=c, l=lgw,
                                                   m=2*self.m, h0r=h0r)
        return self._gws[lgw]

    def get_life_time(self, lgws=None):
        """ GW timescale, adding lgws listed in argument (def. just 2*l_cloud) .
        """
        lgws = self._gws.keys() if lgws is None else lgws
        if len(lgws)==0:
            lgws = [2*self.l] 
        powers = []
        for lgw in lgws:
            powers.append(self.gw(lgw=lgw).power)
        rest_energy = self.mass*C_SI**2
        tgw = rest_energy / sum(powers)
        return tgw
    

class Zabs(object):
    # Numerical fits to fine-structure constant alpha (`a`) provided by R Brito
    # (set up this way to make it easier to add fits dynamically later.)
    # NOTE: these fits assume `chi = chi_f`, could generalize to arbitrary spin
    _FITS = {
        (2, 2): lambda a: 0.7904787874157165*a**8 - 2.9417505987440284*a**9 +\
                          2.803119859556814*a**10,
        (3, 2): lambda a: 1.08158476738751*a**10 - 0.4006416305003071*a**12,
    }

    def __init__(self, l, m):
        self.l = int(l)
        self.m = int(m)
        self._alpha_fit = None

    @property
    def alpha_fit(self):
        if self._alpha_fit is None:
            key = (self.l, self.m)
            if key in self._FITS:
                self._alpha_fit = self._FITS[key]
            else:
                # TODO: add ability to produce fits dynamically here
                raise NotImplementedError("no exisiting fit for (%i, %i)" % 
                                          (self.l, self.m))
        return self._alpha_fit

    def __call__(self, alpha):
        return self.alpha_fit(alpha)

    @staticmethod
    def fast_fit(a, lgw=2, mgw=2):
        return Zabs._FITS[lgw, mgw](a)


class GravitationalWaveMode(object):
    def __init__(self, f, l=2, m=2, h0r=1, r0=1, c=0):
        """ A single (l, m) gravitational wave.

        Arguments
        ---------
        f: float
            signal frequency
        l: int
            azimuthal number (def. 2)
        m: int
            magnetic number (def. 2)
        c: float
            dimensionless spheroidal-harmonics parameter `c=a*omega` (def. 0).
        h0r: float
            intrinsic amplitude at fiducial distance, `h0r = h0(r0)` (def. 1).
        r0: float
            reference distance `r0` in meters (def. 1).
        """
        self.c = c
        self.l = l
        self.m = m
        self.f = f
        self.omega = 2*np.pi*self.f
        self.h0r = h0r
        self.r0 = r0
        # total radiated power in this mode (ATTENTION: +/- m both included)
        self.power = C_SI**3*(self.omega * self.h0r)**2 / (8.*np.pi*G_SI)
        # others
        self._swsh = None
        self._polarizations = None

    @property
    def swshs(self):
        if self._swsh is None:
            c =  self.c
            l = self.l
            m = self.m
            s = -2  # spin-weight of GWs
            self._swsh = (leavers.SpinWeightedSpheroidalHarmonic(c, l, m, s),
                           leavers.SpinWeightedSpheroidalHarmonic(c, l, -m, s))
        return self._swsh
    
    @property
    def polarizations(self):
        if self._polarizations is None:
            wgw = self.omega
            m = self.m
            swsh_p, swsh_m = self.swshs
            def hp(theta, phi, t):
                return np.cos(wgw*t + m*phi)*(swsh_p(theta, phi) +
                                              swsh_m(theta, phi)).real
            def hc(theta, phi, t):
                return np.sin(wgw*t + m*phi)*(swsh_p(theta, phi)-
                                              swsh_m(theta, phi)).real
            self._polarizations = (hp, hc)
        return self._polarizations

    def hp(self, *args, **kwargs):
        """ Plus polarization (unit amplitude).

        Arguments
        ---------
        theta: float
            inclination angle
        phi: float
            orbital phase (azimuthal angle)
        t: float, array
            times.
        r: float
            distance from source in meters (def. 1).

        Returns
        -------
        hp: array
            plus polarization waveform for given times.
        """
        r = kwargs.pop('r', self.r0)
        return (self.r0 / r)*self.polarizations[0](*args, **kwargs)

    def hc(self, *args, **kwargs):
        """ Cross polarization (unit amplitude).

        Arguments
        ---------
        theta: float
            inclination angle
        phi: float
            orbital phase (azimuthal angle)
        t: float, array
            times.
        r: float
            distance from source in meters (def. 1).

        Returns
        -------
        hp: array
            cross polarization waveform for given times.
        """
        r = kwargs.pop('r', self.r0)
        return (self.r0 / r)*self.polarizations[1](*args, **kwargs)
