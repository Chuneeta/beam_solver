import h5py
import os
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
import catdata as cd
import beam_utils as bt
import healpy

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
        ura, udec = catd.get_unique(ras, decs)
        np.testing.assert_almost_equal(ura, ras)
        np.testing.assert_almost_equal(udec, decs)
        ura, udec = catd.get_unique(ras[0], decs[0])
        np.testing.assert_almost_equal(ura, [ras[0]])
        np.testing.assert_almost_equal(udec, [decs[0]])
        ura, udec = catd.get_unique([30.01713089, 30.01713089, 30.01713089], [-30.88211818, -30.88211818, -30.88211818])
        np.testing.assert_almost_equal(ura, [30.01713089])
        np.testing.assert_almost_equal(udec, [-30.88211818])
        ura, udec = catd.get_unique([30.01713089, 30.01713089, 30.01713089], [-30.88211818, -30.88211818, -30.85])	
        np.testing.assert_almost_equal(ura, [30.01713089], 2)
        np.testing.assert_almost_equal(udec, [-30.88211818], 2)

        # checking for errors
        nt.assert_raises(AssertionError, catd.get_unique, ras, decs[0:4])
       	nt.assert_raises(TypeError, catd.get_unique, 'a', decs[0])

    def test_gen_catalog(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        nt.assert_equal(catd.Npols, 1)
        nt.assert_equal(catd.Nfits, len(fitsfiles_xx))
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.ha_array.shape, (len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.data_array.shape, (1, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.err_array.shape, (1, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.pos_array.shape, (2, len(ras)))
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras, decs]))
        catd.gen_catalog(fitsfiles_yy, ras, decs)
        nt.assert_equal(catd.Npols, 1)
        nt.assert_equal(catd.Nfits, len(fitsfiles_xx))
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.ha_array.shape, (len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.data_array.shape, (1, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.err_array.shape, (1, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.pos_array.shape, (2, len(ras)))	
        catd.gen_catalog(np.array(fitsfiles_xx), ras, decs)        
        catd.gen_catalog(fitsfiles_xx, np.array(ras), np.array(decs))
        srcdict = catd.gen_catalog(fitsfiles_xx, ras, decs, return_data=True)
        nt.assert_equal(len(srcdict.keys()), len(ras))
        ras0 = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
        decs0 = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835]
        fitsfile = [os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')]
        srcdict = catd.gen_catalog(fitsfile, ras0, decs0, return_data=True)
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras0, decs0]))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.], [0.5], [0.6]]]))	
        keys = srcdict.keys()
        np.testing.assert_almost_equal(srcdict[(74.26, -52.02)]['data'], np.array([[0.75]]))

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
        nt.assert_equal(catd.Npols, 2)
        nt.assert_equal(catd.Nfits, len(fitsfiles_xx))
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.ha_array.shape, (len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.data_array.shape, (2, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.err_array.shape, (2, len(ras), len(fitsfiles_xx)))
        nt.assert_equal(catd.pos_array.shape, (2, len(ras)))
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
        ras0 = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
        decs0 = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835]
        fitsfile = [os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')]
        catd.gen_catalog(fitsfile, ras0, decs0)
        catd.write_hdf5('srcd.h5', clobber=True)
        catd.read_hdf5('srcd.h5')
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras0, decs0]))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.], [0.5], [0.6]]]))
        nt.assert_raises(IOError, catd.read_hdf5, 'src.h5')
        
    def test_calc_catalog_flux(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs) 
        beam = bt.get_fitsbeam(beamfits, 151e6)
        catd.calc_catalog_flux(beam, 'xx')
        catd.calc_catalog_flux(beam, 'yy')
        nt.assert_raises(AttributeError, catd.calc_catalog_flux, fitsfile1_xx, 'xx')
        ras0 = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
        decs0 = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835]
        fitsfile = [os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')]
        catd.gen_catalog(fitsfile, ras0, decs0)
        catalog_flux = catd.calc_catalog_flux(beam, 'xx')
        azalt = catd.azalt_array
        simulated_fluxs = [0.75, 0.5, 1, 0.5, 0.6]
        catalog_flux0 = []
        for i in range(azalt.shape[1]):
            beamval = healpy.get_interp_val(beam, np.pi/2 - (azalt[1, i, 0] * np.pi/180.), azalt[0, i, 0] * np.pi/180.)
            catalog_flux0.append(simulated_fluxs[i] / beamval.real)
        np.testing.assert_almost_equal(catalog_flux, catalog_flux0, 3)
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        catd.calc_catalog_flux(beam, 'xx')
        catd.calc_catalog_flux(beam, 'yy')
        nt.assert_raises(KeyError, catd.calc_catalog_flux, beam,  'xy')
