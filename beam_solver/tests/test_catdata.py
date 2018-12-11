import h5py
import os
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
from beam_solver import catdata as cd
from beam_solver import beam_utils as bt

# beamfile
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')

# xx fitsfiles
fitsfile1_xx = os.path.join(DATA_PATH, '2458115.23736.xx.fits')
fitsfile2_xx = os.path.join(DATA_PATH, '2458115.24482.xx.fits')
fitsfiles_xx = [fitsfile1_xx, fitsfile2_xx]

# yy fitsfiles
fitsfile1_yy = os.path.join(DATA_PATH, '2458115.23736.yy.fits')
fitsfile2_yy = os.path.join(DATA_PATH, '2458115.24482.yy.fits')
fitsfiles_yy = [fitsfile1_yy, fitsfile2_yy]

# right ascension and declination values
ras = [30.01713089, 27.72922349, 36.75248962, 34.2415497, 78.3776346, 74.03785837]
decs = [-30.88211818, -29.53377208, -30.63958257, -29.93990039, -30.48595805, -30.08651873]

class Test_catData():
    def test_get_unique(self):
        catd = cd.catData()
        catd.get_unique(ras, decs)
        catd.get_unique(ras[0], decs[0])

        # checking for assertions
        nt.assert_raises(AssertionError, catd.get_unique, ras, decs[0:4])
        
        # checking for valueerrors
       	nt.assert_raises(TypeError, catd.get_unique, 'a', decs[0])

    def test_gen_catalog(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        catd.gen_catalog(fitsfiles_yy, ras, decs)
        catd.gen_catalog(np.array(fitsfiles_xx), ras, decs)        
        catd.gen_catalog(fitsfiles_xx, np.array(ras), np.array(decs))
        srcdict = catd.gen_catalog(fitsfiles_xx, ras, decs, return_data=True)

        # check for errors
        nt.assert_raises(AssertionError, catd.gen_catalog, fitsfiles_xx, ras[0:2], decs)
        nt.assert_raises(TypeError, catd.gen_catalog, fitsfiles_xx, ras[0], decs[0])
        nt.assert_raises(TypeError, catd.gen_catalog, fitsfiles_xx, ['3', '4'], decs[0:2])
        nt.assert_raises(TypeError, catd.gen_catalog, fitsfiles_xx, ras[0:2], [30, '-31.5'])
        nt.assert_raises(IndexError, catd.gen_catalog, fitsfiles_xx[0], ras, decs)
        nt.assert_raises(AttributeError, catd.gen_catalog, [0, 1], ras, decs)
	
    def test_gen_polcatalog(self):
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        catd.gen_polcatalog(np.array(fitsfiles_xx), fitsfiles_yy, ras, decs)
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, np.array(ras), np.array(decs)) 

        # check for errors
        nt.assert_raises(AssertionError, catd.gen_polcatalog, fitsfiles_xx, fitsfiles_yy, ras[0:2], decs)
        nt.assert_raises(AssertionError, catd.gen_polcatalog, fitsfiles_xx[0:1], fitsfiles_yy, ras, decs)
        nt.assert_raises(TypeError, catd.gen_polcatalog, fitsfiles_xx, fitsfiles_yy, ras[0], decs[0])
        nt.assert_raises(TypeError, catd.gen_polcatalog, fitsfiles_xx, fitsfiles_yy, ['3', '4'], decs[0:2])
        nt.assert_raises(TypeError, catd.gen_polcatalog, fitsfiles_xx, fitsfiles_yy, ras[0:2], [30, '-31.5'])
        nt.assert_raises(IndexError, catd.gen_polcatalog, fitsfiles_xx[0], fitsfiles_yy[0], ras, decs)

    def test_write_hdf5(self):
        catd = cd.catData()
        nt.assert_raises(TypeError, catd.write_hdf5 , 'srcd.h5', clobber=True)
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        catd.write_hdf5('srcd.h5', clobber=True)
        nt.assert_raises(IOError, catd.write_hdf5, 'srcd.h5')
        
    def test_read_hdf5(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        catd.write_hdf5('srcd.h5', clobber=True)
        catd.read_hdf5('srcd.h5')
        nt.assert_raises(IOError, catd.read_hdf5, 'src.h5')
        
    def test_calc_corrflux(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs) 
        beam = bt.get_fitsbeam(beamfits, 151e6)
        catd.calc_corrflux(beam, 'xx')
        catd.calc_corrflux(beam, 'yy')
        nt.assert_raises(AttributeError, catd.calc_corrflux, fitsfile1_xx, 'xx')

        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        catd.calc_corrflux(beam, 'xx')
        catd.calc_corrflux(beam, 'yy')
        nt.assert_raises(KeyError, catd.calc_corrflux, beam,  'xy')
