import numpy as np

import astropy.units as u
from mcfacts.physics.point_masses import si_from_r_g, si_from_r_g_optimized


# units = [1, u.g, u.kg, u.solMass, u.jupiterMass, u.earthMass]
units = [1]

# and the array case
vals = [1.0, 5435.46345, 48.5, 137.0, 0.00001, np.array([0.0, 1.0, 5435.46345, 48.5, 137.0, 0.00001])]

smbh_mass = 1.0

def test_si_from_rg():
    for val, unit in zip(vals, units):
        # notable difference: r_g_from units returns unitless values, while optimized 
        # returns values in meters. Since the vast majority of uses of this function return

        orig = si_from_r_g(smbh_mass, val * unit)
        opt = si_from_r_g_optimized(smbh_mass, val * unit)

        assert(np.allclose(orig, opt, rtol=1e-6))


