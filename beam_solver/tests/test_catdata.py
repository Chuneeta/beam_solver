import h5py
import os
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
from beam_solver import catdata as cd
from beam_solver import beam_utils as bt
from beam_solver import fits_utils as ft

beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
fitsfile = os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')
outfile = fitsfile.replace('.fits', '.mod.fits')
h5file = os.path.join(DATA_PATH, 'srcd.h5')

ras = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
decs = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835]

class Test_catData():
    def test_get_unique(self):
        catd = cd.catData()
        ura, udec = catd.get_unique(ras, decs)
        np.testing.assert_almost_equal(ura, ras)
        np.testing.assert_almost_equal(udec, decs)
        ura, udec = catd.get_unique([30.01713089, 30.01713089, 30.01713089], [-30.88211818, -30.88211818, -30.85])	
        np.testing.assert_almost_equal(ura, [30.01713089], 2)
        np.testing.assert_almost_equal(udec, [-30.88211818], 2)

    def test_length_radec(self):
        catd = cd.catData()
        nt.assert_raises(AssertionError, catd.get_unique, ras, decs[0:4])

    def test_check_jd(self):
        catd = cd.catData()
        nt.assert_raises(AssertionError, catd._get_jds, [fitsfile])
        
    def test_get_jd(self):
        ft.add_keyword(fitsfile, 'JD', 2458115.23736, outfile, overwrite=True)
        catd = cd.catData()
        jds = catd._get_jds([outfile])
        nt.assert_equal(jds, [2458115.23736])

    def test_gen_catalog(self):
        catd = cd.catData()
        catd.gen_catalog([outfile], ras, decs)
        nt.assert_equal(catd.Npols, 1)
        nt.assert_equal(catd.Nfits, 1)
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), 1))
        nt.assert_equal(catd.data_array.shape, (1, len(ras), 1))
        nt.assert_equal(catd.err_array.shape, (1, len(ras), 1))
        nt.assert_equal(catd.pos_array.shape, (2, len(ras)))
        nt.assert_equal(catd.ha_array.shape, (len(ras), 1))

    def test_catalog_vals(self):
        catd = cd.catData()
        catd.gen_catalog([outfile], ras, decs)
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras, decs]))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.0], [0.5], [0.6]]]))

    def test_return_dict(self):
        catd = cd.catData()
        srcdict = catd.gen_catalog([outfile], ras, decs, return_data=True)
        nt.assert_equal(len(srcdict.keys()), len(ras))
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras, decs]))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.], [0.5], [0.6]]]))

    def test_write_hdf5(self):
        catd = cd.catData()
        catd.gen_catalog([outfile], ras, decs)
        catd.write_hdf5(h5file, clobber=True)
        nt.assert_true(os.path.exists(h5file))
        nt.assert_raises(IOError, catd.write_hdf5, h5file)

    def test_read_hdf5(self):
        catd = cd.catData()
        catd.gen_catalog([outfile], ras, decs)
        catd.write_hdf5(h5file, clobber=True)
        catd.read_hdf5(h5file)
        nt.assert_equal(catd.Nsrcs, len(ras))
        np.testing.assert_almost_equal(catd.pos_array, np.array([ras, decs]))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.], [0.5], [0.6]]]))
        
    def test_no_hdf5(self):
        catd = cd.catData()
        nt.assert_raises(IOError, catd.read_hdf5, 'src.h5')

    def test_calc_catalog_flux(self):
        catd = cd.catData()
        catd.gen_catalog([outfile], ras, decs) 
        beam = bt.get_fitsbeam(beamfits, 151e6)
        catalog_flux = catd.calc_catalog_flux(beam, 'xx')
        nt.assert_almost_equal(catalog_flux[2], 1.000, 3)
