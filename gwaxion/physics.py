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
    def __init__(self, mass, chi):
        """ Black hole.

        Arguments
        ---------
        mass: float
            mass in kg.

        chi: float
            dimensionless spin, `chi=(c^2/G)(a/M)` for `a=J/(Mc)`, in (0, 1).
        """
        # MASS
        self.mass = mass
        self.mass_msun = mass / MSUN_SI
        # SPIN
        self.chi = chi
        self.angular_momentum = mass**2 * chi * (G_SI/C_SI)
        self.a = self.angular_momentum / (mass * C_SI)
        # RADII
        self.rs = 2 * G_SI * mass / C_SI**2
        self.rp_dimensionless = 1 + np.sqrt(1 - self.chi**2)
        self.rp = (self.rs + np.sqrt(self.rs**2 - 4*self.a**2))/2. 
        self.rm = (self.rs - np.sqrt(self.rs**2 - 4*self.a**2))/2. 
        self.rm_dimensionless = 1 - np.sqrt(1 - self.chi**2)
        # ANGULAR VELOCITY
        self.omegah_dimensionless= chi / (2. * self.rp_dimensionless)
        # other quantities computed below
        self._omegah = None

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
    
    @property
    def omegah(self):
        if self._omegah is None:
            self._omegah = self.omega(self.rp, 0)
        return self._omegah
