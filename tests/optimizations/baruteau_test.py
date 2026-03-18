from mcfacts.physics.binary.evolve import bin_harden_baruteau, bin_harden_baruteau_optimized
import numpy as np
import pandas as pd
import astropy.units as u
import pytest

# def test_baruteau():


# parse array out of the csv file
def parse_array(cell):
    if isinstance(cell, str) and cell.startswith('['):
        cleaned = cell.strip('[]')
        return np.fromstring(cleaned, sep=' ')
    return cell

# parse scalar value out of csv file
def parse_value(cell):
    if isinstance(cell, str):
        # check if it's an array
        if '[' in cell:
            cleaned = cell.replace('[', '').replace(']', '')
            return np.fromstring(cleaned, sep=' ')
        
        # try to extract number from string with units (e.g., "147662503805.01248 m")
        try:
            # split by whitespace and take the first part (the number)
            numeric_part = cell.split()[0]
            return float(numeric_part)
        except (ValueError, IndexError):
            return cell
    return cell


def setup_baruteau_params():
    """Return input parameters read from CSV."""
    inputs = pd.read_csv("tests/optimizations/baruteau_inputs.csv", header=None)
    params = []
    for _, row in inputs.iterrows():
        params.append(tuple(parse_value(row[i]) for i in range(11)))
    return params


@pytest.mark.parametrize(
    "bin_mass_1, bin_mass_2, bin_sep, bin_ecc, bin_time_to_merger_gw, bin_flag_merging, "
    "bin_time_merged, smbh_mass, timestep_duration_yr, time_gw_normalization, time_passed",
    setup_baruteau_params(),
)
def test_sweep_optimized_matches_original(
    bin_mass_1,
    bin_mass_2,
    bin_sep,
    bin_ecc,
    bin_time_to_merger_gw,
    bin_flag_merging,
    bin_time_merged,
    smbh_mass,
    timestep_duration_yr,
    time_gw_normalization,
    time_passed,
):

    out_sep, out_flag_merging, out_time_merged, out_time_to_merger_gw = bin_harden_baruteau(
        np.array([bin_mass_1]),
        np.array([ bin_mass_2 ]),
        np.array([ bin_sep ]),
        np.array([ bin_ecc ]),
        np.array([ bin_time_to_merger_gw ]),
        np.array([ bin_flag_merging ]),
        np.array([ bin_time_merged ]),
        smbh_mass,
        timestep_duration_yr,
        time_gw_normalization,
        time_passed,
        r_g_in_meters=None
    )
    out_sep_opt, out_flag_merging_opt, out_time_merged_opt, out_time_to_merger_gw_opt = bin_harden_baruteau_optimized(
        np.array([ bin_mass_1 ]),
        np.array([ bin_mass_2 ]),
        np.array([ bin_sep ]),
        np.array([ bin_ecc ]),
        np.array([ bin_time_to_merger_gw ]),
        np.array([ bin_flag_merging ]),
        np.array([ bin_time_merged ]),
        smbh_mass,
        timestep_duration_yr,
        time_gw_normalization,
        time_passed,
        r_g_in_meters=None
    )

    assert np.allclose(out_sep, out_sep_opt, rtol=1e-6)

    assert np.allclose(out_flag_merging, out_flag_merging_opt, rtol=1e-6)
    assert np.allclose(out_time_merged, out_time_merged_opt, rtol=1e-6)
    assert np.allclose(out_time_to_merger_gw, out_time_to_merger_gw_opt, rtol=1e-6)




# # units = [1, u.g, u.kg, u.solMass, u.jupiterMass, u.earthMass]
# units = [1]
#
# # and the array case
# vals = [1.0, 5435.46345, 48.5, 137.0, 0.00001, np.array([0.0, 1.0, 5435.46345, 48.5, 137.0, 0.00001])]
#
# smbh_mass = 1.0
#
# def test_si_from_rg():
#     for val, unit in zip(vals, units):
#         # notable difference: r_g_from units returns unitless values, while optimized 
#         # returns values in meters. Since the vast majority of uses of this function return
#
#         orig = si_from_r_g(smbh_mass, val * unit)
#         opt = si_from_r_g_optimized(smbh_mass, val * unit)
#
#         assert(np.allclose(orig, opt, rtol=1e-6))
#


