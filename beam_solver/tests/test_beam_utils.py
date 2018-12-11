import os
from beam_solver import beam_utils as bt
import nose.tools as nt
from beam_solver.data import DATA_PATH

cstbeam1 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_151MHz.txt')
cstbeam2 = os.path.join(DATA_PATH, 'HERA_4.9m_E-pattern_152MHz.txt')
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

def test_gengaussbeam():
    beam = bt.get_gaussbeam(6, 2)
    beam = bt.get_gaussbeam(6.0, 2)
    beam = bt.get_gaussbeam('6.0', 2)
    beam = bt.get_gaussbeam(6.0, '2')    

    # checks for errors
    nt.assert_raises(ValueError, bt.get_gaussbeam, 'a', 2)
    nt.assert_raises(ValueError, bt.get_gaussbeam, 6, 'a')
    nt.assert_raises(TypeError, bt.get_gaussbeam, 6, 2.0, 31.0)
    nt.assert_raises(TypeError, bt.get_gaussbeam, 6, 2.0, '31')

def test_fitsbeam():
    beam = bt.get_fitsbeam(beamfits, 150e6)
    beam = bt.get_fitsbeam(beamfits, 150.e6)
    beam = bt.get_fitsbeam(beamfits, 150e6, pol='yy')
    beam = bt.get_fitsbeam(beamfits, 151.2e6)
    
    # check for errors
    nt.assert_raises(TypeError, cstbeam1, 150e6)
    nt.assert_raises(TypeError, beamfits, '150e6')

def test_cstbeam():
    beam = bt.get_cstbeam([cstbeam1], [151e6], 151e6)
    beam = bt.get_cstbeam(cstbeam1, [151e6], 151e6)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2], [151e6, 152e6], 151e6)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2], [151e6, 152e6], 151.5e6)
    beam = bt.get_cstbeam([cstbeam1, cstbeam2, cstbeam1], [151e6, 152e6, 153e6], 151.5e6)

    # check for errors
    nt.assert_raises(ValueError, bt.get_cstbeam, [cstbeam1, cstbeam2], [151e6], 151e6)
    nt.assert_raises(ValueError, bt.get_cstbeam, [cstbeam1, cstbeam2, cstbeam1], [151e6, 152e6, 151e6], 151e6)
    nt.assert_raises(TypeError, bt.get_cstbeam, cstbeam1, 151e6, 151e6)
    nt.assert_raises(ValueError, bt.get_cstbeam, [beamfits], [151e6], 151e6)
