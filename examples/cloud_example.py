#! /usr/bin/env python

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
from gwaxion import physics

# Let's create a black-hole--boson system starting from a given BH mass (50 MSUN),
# initial dimensionless BH spin (0.9) and a fine structure constant (alpha = 0.2)

bhb = physics.BlackHoleBoson.from_parameters(m_bh=50, alpha=0.2, chi_bh=0.9)

# Print some properties of the system

print "BLACK HOLE:"
print "----------"
print "The initial black-hole has a mass of %.1f MSUN (%.1e kg)."\
      % (bhb.bh.mass_msun, bhb.bh.mass)
print "It also has a dimensionless spin of chi=%.2f, which corresponds to an "\
      "angular momentum J=%.1e Js, and Kerr parameter a=%.1e m."\
      % (bhb.bh.chi, bhb.bh.angular_momentum, bhb.bh.a)
print "The outer radius is %.1e m, with a horizon angular frequency of %.1f rad/s."\
      % (bhb.bh.rp, bhb.bh.omega_horizon)

print "\nBOSON:"
print "-----"
print "The boson has a rest-mass of %1.e kg, which corresponds to an energy of "\
      "%.1e eV. Its spin is %i."\
      % (bhb.boson.mass, bhb.boson.energy_ev, bhb.boson.spin)

print "\nJOINT SYSTEM:"
print "------------"
print "The black-hole--boson system has a fine-structure constant of %.2f.\n" \
      % bhb.alpha

# Let's compute the growth rate for a bunch of energy levels (l, m, nr)'s
print "The superradiant growth rates of some of its energy levels are:"
print "(l, m, nr, rate in Hz)"
for l in range(0, 3):
    for m in range(0, l+1):
        for nr in range(0, 3):
            print "(%i, %i, %i, %.1e)" % (l, m, nr, bhb.level_omega_im(l, m, nr))

# Check if the m=1 mode is superradiant
print "The `m = 1` level is superradiant: %r" % bhb.is_superradiant(1)

# Obtain the fastest growing level
print "The fastest level is: (%i, %i, %i, %.1e)" % bhb.max_growth_rate()

# Let's now add a cloud to populate this level
# in this case, this equivalent to `bhb.cloud(1, 1, 0)`
print "\nCLOUDS:"
print "------"
cloud = bhb.best_cloud()

# The cloud is now stored in the BlackHoleBoson object
print "The system now has one cloud:"
print bhb.clouds

# Print some cloud properties
mass_fraction = cloud.mass / cloud.bhb_initial.bh.mass
print "\nAfter superradiant growth, the cloud has mass %.1f MSUN (%.1e kg)."\
      % (cloud.mass_msun, cloud.mass)
print "This is %.1f%%  of the original BH mass." % (mass_fraction*100)

# Note that the cloud object contains a pointer to the original BlackHoleBoson
# system under `cloud.bhb_initial`. It also has a similar object for the black
# hole that remains after superradiant growth:
print "The mass and spin of the final black hole are: (%.1f MSUN, %.2f)"\
      % (cloud.bhb_final.bh.mass_msun, cloud.bhb_final.bh.chi)

print "\nGRAVITATIONAL WAVES:"
print "----------------"
print "The cloud will produce GWs at %.2f Hz" % cloud.fgw

# Create a waveform and plot it
hp, hc = cloud.waveform
inclination = np.pi/4
time = np.arange(0, 0.05, 1E-4)

from matplotlib import pyplot as plt
plt.plot(time, hp(inclination, time), label=r'$+$')
plt.plot(time, hc(inclination, time), label=r'$\times$')
plt.xlabel("Time (s)")
plt.ylabel(r"$h(t)$")
plt.title("GWs from BH (%.f MSUN) and scalar (%.1e eV)" 
          % (cloud.bhb_final.bh.mass_msun, cloud.bhb_final.boson.energy_ev))
plt.legend(loc='upper right')
plt.show()

