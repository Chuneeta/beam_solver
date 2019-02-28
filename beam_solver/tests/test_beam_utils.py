import os
import numpy as np
import healpy as hp
import pyuvdata
import nose.tools as nt
import beam_solver.beam_utils as bt
#import beam_utils as bt
from beam_solver.data import DATA_PATH

cstbeam1 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_151MHz.txt')
cstbeam2 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_152MHz.txt')
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

def test_get_gaussbeam():
    beam0 = np.zeros((31, 31))
    bmx, bmy = np.indices(beam0.shape)
    beam0 = np.exp(-((bmx)** 2 + (bmy)**2) / (2 * 2.0**2))
    beam = bt.get_gaussbeam(2, 2)
    np.testing.assert_almost_equal(beam, beam0)

def test_get_LM():
    x, y = bt.get_LM(31, center=(0,0), res=0.5)
    nt.assert_equal(x[0][0], 0.0)
    nt.assert_equal(y[0][0], 0.0)
    x, y = bt.get_LM(31, center=(15,15), res=0.5)
    nt.assert_equal(x[15][16], 0.0)
    nt.assert_equal(y[16][15], 0.0)    

def test_get_top():
    x, y, z = bt.get_top(10, center=(0,0), res=1)
    np.testing.assert_almost_equal(z, np.sqrt(1 - x**2 - y**2))

def test_get_src_tracks():
    beamval = bt.get_src_tracks((0,0,0), 1, 2)
    nt.assert_equal(beamval, 1)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(0, 0, 100)
    z = np.linspace(0, 0, 100)
    sigma = 2
    beamval = bt.get_src_tracks((x, y, z), 2, sigma)
    np.testing.assert_almost_equal(beamval, 2 * ( np.exp(-x**2 / (2 * sigma**2))))

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
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6)
    nt.assert_equal(beam.shape, (49152,))
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6, nside=32)
    nt.assert_equal(beam.shape, (12288,))
   
def test_interp():
    data = np.linspace(0, 1, 100)
    data = data.reshape((1, len(data)))
    interp_beam = bt._interp_freq(data, [150e6], 150e6) 
    np.testing.assert_almost_equal(interp_beam, data[0,:])

