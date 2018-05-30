import numpy as np

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
        # SPIN (a = rs*chi/2)
        self.chi = chi  # dimensionless
        self.angular_momentum = mass**2 * chi * (G_SI/C_SI)
        self.a = self.angular_momentum / (mass * C_SI)  # dimensions of length
        # RADII
        self.rs = 2 * G_SI * mass / C_SI**2
        self.rp = (self.rs + np.sqrt(self.rs**2 - 4*self.a**2))/2. 
        self.rm = (self.rs - np.sqrt(self.rs**2 - 4*self.a**2))/2. 
        # RADDI in natural units (G=M=c=1)
        self.rp_natural = 1 + np.sqrt(1 - self.chi**2)  # = 2 rp / rs
        self.rm_natural = 1 - np.sqrt(1 - self.chi**2)
        # ANGULAR VELOCITY
        self.omega_horizon = chi * C_SI / (2. * self.rp)
        self.omega_horizon_natural = chi / (2. * self.rp_natural)
        # also, omegah_natural = omegah * rs / (2c)
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
        self.spin_weight = spin
        self.reduced_compton_wavelength = HBAR_SI / (mass*C_SI)
        self.compton_wavelength = 2*np.pi*self.reduced_compton_wavelength
        # Other quantities
        self.mu_brito = C_SI * self.mass / HBAR_SI


class BlackHoleBoson(object):
    def __init__(self, m_bh, chi_bh, m_b, spin_weight=0, msun=True, ev=True):
        self.bh = BlackHole(m_bh, chi_bh, msun=msun)
        self.boson = Boson(m_b, spin=spin_weight, ev=ev)
        # Fine-structure constant
        self.alpha = (0.5*self.bh.rs) * self.boson.energy / (HBAR_SI * C_SI)

