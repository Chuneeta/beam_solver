import os
from beam_solver.data import DATA_PATH
import extract as et
import nose.tools as nt
import numpy as np
DATA_PATH = '/Users/Ridhima/Documents/ucb_projects/beam_characterization/beam_solver/beam_solver/data/'

fitsfile = os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')
beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

def test_get_centre_radec():
    cra, cdec = et.get_centre_radec(fitsfile)
    nt.assert_almost_equal(cra, 22.9371375, places=2)
    nt.assert_almost_equal(cdec, -30.675116666666668, places=2)
    nt.assert_raises(IOError, et.get_centre_radec, '2458115.23736.xx.fits')
    nt.assert_raises(TypeError, et.get_centre_radec, beamfile)
	
def test_get_flux():
    stats = et.get_flux(fitsfile, 74.26237654, -52.0209015)
    nt.assert_almost_equal(stats['flux'], 0.75)
    stats = et.get_flux(fitsfile, 41.91116875, -43.2292595)
    nt.assert_almost_equal(stats['flux'], 0.5)
    stats = et.get_flux(fitsfile, 22.47460079, -30.27372862)
    nt.assert_almost_equal(stats['flux'], 1)
    stats = et.get_flux(fitsfile, 9.8393989, -17.40763737)
    nt.assert_almost_equal(stats['flux'], 0.5)
    stats = et.get_flux(fitsfile, 356.25426296, -0.3692835)
    nt.assert_almost_equal(stats['flux'], 0.6)
    stats = et.get_flux(fitsfile, 356.26, -0.37)
    nt.assert_almost_equal(stats['flux'], 0.6)
    
    nt.assert_raises(KeyError, et.get_flux, beamfile, 30.01713089, -30.88211818)
    nt.assert_raises(TypeError, et.get_flux, fitsfile, '30.0', -30.88211818)
    nt.assert_raises(TypeError, et.get_flux, fitsfile, 30.0, '-30.8')
    nt.assert_raises(ValueError, et.get_flux, fitsfile, 30.0, -30.88211818, flux_type='int')

