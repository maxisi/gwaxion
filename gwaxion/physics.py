import numpy as np
from scipy.misc import factorial

try:
    # if LAL is available, import constants for better accuracy (?)
    from lal import C_SI, G_SI, HBAR_SI, PC_SI, MSUN_SI
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


# ###########################################################################
# CLASSES

class BlackHole(object):
    def __init__(self, mass, chi, msun=False):
        """ Black hole.

        Arguments
        ---------
        mass: float
            mass in kg (or in MSUN, if `msun` is True).

        chi: float
            dimensionless spin, `chi=(c^2/G)(a/M)` for `a=J/(Mc)`, in (0, 1).

        msun: bool
            whether `mass` is given in solar masses.
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
        # SPIN
        self.chi = chi  # dimensionless
        self.a = self.rg * self.chi
        self.angular_momentum = self.mass * C_SI * self.a
        # RADDI in natural units (G=M=c=1)
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

    # @property
    # def omegah(self):
    #     if self._omegah is None:
    #         self._omegah = self.omega(self.rp, 0)
    #     return self._omegah


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
    def __init__(self, m_bh=0, m_b=0, alpha=0, msun=True, ev=True,
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
        if msun:
            self.m_bh_msun = m_bh
            m_bh *= MSUN_SI
        if ev:
            self.m_b_ev = m_b
            m_b *= EV_SI / C_SI**2
        self.m_bh = m_bh
        self.m_b = m_b
        alpha_new = G_SI * self.m_bh * self.m_b / (HBAR_SI * C_SI)
        if m_bh and m_b and alpha:
            # check consistency
            if abs(alpha - alpha_new) < tolerance:
                raise ValueError("alpha incompatible with BH & boson masses.")
        elif m_bh and alpha:
            # compute boson mass
            self.m_b = HBAR_SI * C_SI * alpha / (G_SI * self.m_bh)
            self.m_b_ev = self.m_b * C_SI**2 / EV_SI 
        elif m_b and alpha:
            # compute BH mass
            self.m_bh = HBAR_SI * C_SI * alpha / (G_SI * self.m_b)
            self.m_bh_msun = self.m_bh / MSUN_SI
        self.alpha = alpha or alpha_new 


class BlackHoleBoson(object):
    def __init__(self, m_bh, chi_bh, m_b, boson_spin=0, msun=True, ev=True):
        """ System composed of a black-hole and a boson.

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
        msun: bool
            BH mass provided in solar masses, rather than SI (def True).
        ev: bool
            boson mass provided in eV, rather than SI (def True).
        """
        self.bh = BlackHole(m_bh, chi_bh, msun=msun)
        self.boson = Boson(m_b, spin=boson_spin, ev=ev)
        # Length ratio is unity when `2*pi*R = lambda_c`
        # NOTE: in Brito et al., the length ratio is denoted by 'Mmu'
        # self.length_ratio = self.bh.rs / self.boson.reduced_compton_wavelength
        # Fine-structure constant `G M m / (hbar c) = rg / lambda_bar_c`
        self.alpha = self.bh.rg / self.boson.reduced_compton_wavelength

    @classmethod
    def from_alpha(cls, **kwargs):
        chi = kwargs.pop('chi_bh')
        s = kwargs.pop('boson_spin', 0)
        a = Alpha(**kwargs)
        return cls(a.m_bh, chi, a.m_b, boson_spin=s, msun=False, ev=False)

    def _level(self, n):
        return hydrogenic_level(n, self.alpha)

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

    def level_omega_re(self, n):
        ''' Return real part of hydrogenic energy eigen-frequencies in rad/s.

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
        return self.boson.omega * self._level(n)

    def level_frequency(self, n):
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
        return self.level_omega_re(n) / (2*np.pi)

    # --------------------------------------------------------------------
    # GROWTH-RATE

    def _clmn(self, l, m, nr):
        """ Factor in computation of imaginary frequency.

        gamma factor of Eq. (28) in Detweiler (1980), or equivalently
        Clmn factor of Eq. (18) in Arvanitaki & Dubovsky (2011).

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
            sr = self._sr_factor(m)
            clmn = self._clmn(l, m, nr)
            omega_im = w0 * a*(4*l +4) * sr * clmn
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

    def omegagw_dimensionless(self, n):
        return 2*self.level_omega_dimensionless(n)

    def final_bh_spin(self, n, dimensionless=True):
        ''' Returns saturation BH spin: that is, spin reached after SR condition
        no longer satistified for this energy level [Eq. (25) in Brito et al.]

        Arguments
        ---------
        n: float
            *principal* quantum number `n = nr + l + 1`, for `nr` the radial
            and `l` the azimuthal quantum numbers.

        Returns
        -------
        '''
        if dimensionless:
            mwr = self.level_omega_dimensionless(n)
            chif = 4*mwr / (1 + 4*mwr**2)
        else:
            raise NotImplementedError
        return chif

    def final_bh_mass(self, n):
        mwr = self.level_omega_dimensionless(n)
        return self.bh.mass * (1- mwr*(self.bh.chi - self.final_bh_spin(n)))

    def cloud_mass(self, n):
        return self.bh.mass - self.final_bh_mass(n)


class BosonCloud(object):
    def __init__(self, bhb, l, m, nr, msun=True, ev=True):
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
        msun: bool
            provide cloud mass in solar masses (def. True).
        """
        # check `bhb` is of the right type
        self.bhb = bhb
        try:
            self.bhb.boson.reduced_compton_wavelength
            self.bhb.bh.rg
        except AttributeError:
            raise ValueError("'bhb' must be `BlackHoleBoson` instance, not %r"
                             % type(boson))
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
        self.growth_time = 1./self.bhb.level_omega_im(self.l, self.m, self.nr)
        self.is_superradiant = self.bhb.is_superradiant(m)
        # set gravitational-wave properties
        self.fgw = 2*self.bhb.level_frequency(self.n)
        self.lgw = 2*l
        self.mgw = 2*m

    @classmethod
    def from_parameters(cls, l, m, nr, **kwargs):
        bhb = BlackHoleBoson.from_alpha(**kwargs)
        return cls(bhb, l, m, nr)
