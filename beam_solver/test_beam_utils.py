import os
import numpy as np
import healpy as hp
import pyuvdata
import beam_utils as bt
import nose.tools as nt
from beam_solver.data import DATA_PATH

cstbeam1 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_151MHz.txt')
cstbeam2 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_152MHz.txt')
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

def test_gengaussbeam():
    beam0 = np.zeros((31, 31))
    bmx, bmy = np.indices(beam0.shape)
    beam0 = np.exp(-((bmx - float(6)) ** 2 + (bmy - float(6)) ** 2) / (2 * float(2) ** 2))
    beam = bt.get_gaussbeam(6, 2)
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = np.exp(-((bmx - float(6.0)) ** 2 + (bmy - float(6.0)) ** 2) / (2 * float(2) ** 2))
    beam = bt.get_gaussbeam(6.0, 2)
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = np.exp(-((bmx - float('6.0')) ** 2 + (bmy - float('6.0')) ** 2) / (2 * float(2) ** 2))
    beam = bt.get_gaussbeam('6.0', 2)
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = np.exp(-((bmx - float(6.0)) ** 2 + (bmy - float(6.0)) ** 2) / (2 * float('2') ** 2))
    beam = bt.get_gaussbeam(6.0, '2')    
    np.testing.assert_almost_equal(beam, beam0)    

    # checks for errors
    nt.assert_raises(ValueError, bt.get_gaussbeam, 'a', 2)
    nt.assert_raises(ValueError, bt.get_gaussbeam, 6, 'a')
    nt.assert_raises(TypeError, bt.get_gaussbeam, 6, 2.0, 31.0)
    nt.assert_raises(TypeError, bt.get_gaussbeam, 6, 2.0, '31')

def test_fitsbeam():
    uvb = pyuvdata.UVBeam()
    uvb.read_beamfits(beamfits)
    uvb.peak_normalize()
    data_array = uvb.data_array
    beam0 = data_array[0, 0, 0, 12, :]    
    beam = bt.get_fitsbeam(beamfits, 148e6)
    nt.assert_equal(beam.shape, (12288,))
    np.testing.assert_almost_equal(beam, beam0)
    beam = bt.get_fitsbeam(beamfits, 148.e6)
    nt.assert_equal(beam.shape, (12288,))
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = data_array[0, 0, 1, 12, :]
    beam = bt.get_fitsbeam(beamfits, 148e6, pol='yy')
    nt.assert_equal(beam.shape, (12288,))
    np.testing.assert_almost_equal(beam, beam0)
    wgt = 2e-6
    beam0 = (data_array[0, 0, 0, 12, :] * wgt + data_array[0, 0, 0, 13, :] * wgt) / (wgt + wgt)
    beam = bt.get_fitsbeam(beamfits, 150e6)
    nt.assert_equal(beam.shape, (12288,))
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = data_array[0, 0, 0, 12, :]
    beam0 = hp.ud_grade(beam0, 64)
    beam = bt.get_fitsbeam(beamfits, 148e6, nside=64)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)

    # check for errors
    nt.assert_raises(TypeError, cstbeam1, 150e6)
    nt.assert_raises(TypeError, beamfits, '150e6')

def test_cstbeam():
    uvb1 = pyuvdata.UVBeam()
    uvb1.read_cst_beam(cstbeam1, beam_type='power', frequency=151e6,
                          telescope_name='TEST', feed_name='bob',
                          feed_version='0.1', feed_pol='xx',
                          model_name='E-field pattern - Rigging height 4.9m',
                          model_version='1.0')
    uvb1.interpolation_function = 'az_za_simple'
    uvb1.to_healpix(64)
    uvb1.peak_normalize()
    data_array1 = uvb1.data_array
    uvb2 = pyuvdata.UVBeam()
    uvb2.read_cst_beam(cstbeam2, beam_type='power', frequency=151e6,
                          telescope_name='TEST', feed_name='bob',
                          feed_version='0.1', feed_pol='xx',
                          model_name='E-field pattern - Rigging height 4.9m',
                          model_version='1.0')
    uvb2.interpolation_function = 'az_za_simple'
    uvb2.to_healpix(64)
    uvb2.peak_normalize()
    data_array2 = uvb2.data_array
    beam0 = data_array1[0, 0, 0, 0, :]
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)       
    beam = bt.get_cstbeam(cstbeam1, [151e6], 151e6)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2], [151e6, 152e6], 151e6)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = data_array1[0, 0, 1, 0, :]
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6, 'yy')
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)
    beam0 = data_array1[0, 0, 1, 0, :]
    beam0 = hp.ud_grade(beam0, 32)
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6, 'yy', nside=32)
    nt.assert_equal(beam.shape, (12288,))
    np.testing.assert_almost_equal(beam, beam0)
    wgt = 0.5e-6
    beam0 = (data_array1[0, 0, 0, 0, :] * wgt + data_array2[0, 0, 0, 0, :] * wgt) / (wgt + wgt)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2], [151e6, 152e6], 151.5e6)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2, cstbeam1], [151e6, 152e6, 153e6], 151.5e6)
    nt.assert_equal(beam.shape, (49152,))
    np.testing.assert_almost_equal(beam, beam0)

    # check for errors
    nt.assert_raises(ValueError, bt.get_cstbeam, [cstbeam1, cstbeam2], [151e6], 151e6)
    nt.assert_raises(ValueError, bt.get_cstbeam, [cstbeam1, cstbeam2, cstbeam1], [151e6, 152e6, 151e6], 151e6)
    nt.assert_raises(TypeError, bt.get_cstbeam, cstbeam1, 151e6, 151e6)
    nt.assert_raises(ValueError, bt.get_cstbeam, [beamfits], [151e6], 151e6)

