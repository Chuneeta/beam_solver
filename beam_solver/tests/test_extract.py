import os
from beam_solver.data import DATA_PATH
import beam_solver.extract as et
import nose.tools as nt

fitsfile = os.path.join(DATA_PATH, '2458115.23736.xx.fits')
beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

ras = [30.01713089, 27.72922349]
decs = [-30.88211818, -29.53377208]

def test_get_centre_radec():
    cra, cdec = et.get_centre_radec(fitsfile)
    #print cra, cdec
    #nt.assert_raises(IOError, et.get_centre_radec, '2458115.23736.xx.fits')
    #nt.assert_raises(TypeError, et.get_centre_radec, beamfile)

"""	
def test_get_flux():
    et.get_flux(fitsfile, 30.01713089, -30.88211818)
    et.get_flux(fitsfile, 30.01713089, -30.88211818, flux_type='total')
    et.get_flux(fitsfile, 30.01713089, 30.88211818)
    et.get_flux(fitsfile, 30.01713089, 30.88211818, flux_type='total')

    nt.assert_raises(KeyError, et.get_flux, beamfile, 30.01713089, -30.88211818)
    nt.assert_raises(TypeError, et.get_flux, fitsfile, '30.0', -30.88211818)
    nt.assert_raises(TypeError, et.get_flux, fitsfile, 30.0, '-30.8')
    nt.assert_raises(ValueError, et.get_flux, fitsfile, 30.0, -30.88211818, flux_type='int')
"""
