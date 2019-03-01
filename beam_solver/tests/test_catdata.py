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

    def test_get_lstha(self):
        catd = cd.catData()
        jd = 2458115.23736 + 5 / 24. / 60
        lst, ha = catd._get_lstha(2458115.23736, 22.47460079)
        np.testing.assert_allclose(ha, 0, atol=1e-1)

    def test_get_azalt(self):
        catd = cd.catData()
        az, alt = catd._get_azalt(-30.721388888888885, 0)
        nt.assert_almost_equal(az, np.pi/2)
        nt.assert_almost_equal(alt, np.pi/2)

    def test_generate_srcdict(self):
        catd = cd.catData()
        srcdict = catd._generate_srcdict(ras, decs, [outfile])
        keys = srcdict.keys()
        nt.assert_equal(len(keys), len(ras))
        data = srcdict[keys[0]]['data']
        nt.assert_equal(data.shape, (1,))
        np.testing.assert_almost_equal(data, np.array([0.75]))

    def test_srcdict_catdata(self):
        catd = cd.catData()
        srcdict = catd._generate_srcdict(ras, decs, [outfile])
        catd._srcdict_catdata(srcdict)
        nt.assert_equal(catd.data_array.shape, (1, len(ras), 1))
        np.testing.assert_almost_equal(catd.pos_array, srcdict.keys())

    def test_combine_srcdict(self):
        catd1 = cd.catData()
        srcdict1 = catd1._generate_srcdict(ras, decs, [outfile])
        catd2 = cd.catData()
        srcdict2 = catd2._generate_srcdict(ras, decs, [outfile])
        catd = cd.catData()
        srcdict = catd._combine_srcdict(srcdict1, srcdict2)
        keys = srcdict.keys()
        data = srcdict[keys[0]]['data']
        nt.assert_equal(data.shape, (2, 1))
        np.testing.assert_almost_equal(data, np.array([[0.75], [0.75]]))

    def test_gen_catalog_1pol(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, fitsfiles_xx=[outfile])
        nt.assert_equal(catd.pols, ['xx'])
        nt.assert_equal(catd.Nfits, 1)
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), 1))
        nt.assert_equal(catd.data_array.shape, (1, len(ras), 1))
        nt.assert_equal(catd.err_array.shape, (1, len(ras), 1))
        nt.assert_equal(catd.ha_array.shape, (len(ras), 1))

    def test_gen_catalog_2pol(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, fitsfiles_xx=[outfile], fitsfiles_yy=[outfile], pols=['xx', 'yy'])
        nt.assert_equal(catd.pols, ['xx', 'yy'])
        nt.assert_equal(catd.Nfits, 1)
        nt.assert_equal(catd.Nsrcs, len(ras))
        nt.assert_equal(catd.azalt_array.shape, (2, len(ras), 1))
        nt.assert_equal(catd.data_array.shape, (2, len(ras), 1))
        nt.assert_equal(catd.err_array.shape, (2, len(ras), 1))
        nt.assert_equal(catd.ha_array.shape, (len(ras), 1))

    def test_catalog_vals(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, [outfile])
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.0], [0.5], [0.6]]]))
    
    def test_return_dict(self):
        catd = cd.catData()
        srcdict = catd.gen_catalog(ras, decs, [outfile], return_data=True)
        nt.assert_equal(len(srcdict.keys()), len(ras))
        np.testing.assert_almost_equal(srcdict.keys(), catd.pos_array)

    def test_write_hdf5(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, [outfile])
        catd.write_hdf5(h5file, clobber=True)
        nt.assert_true(os.path.exists(h5file))
        nt.assert_raises(IOError, catd.write_hdf5, h5file)

    def test_read_hdf5(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, [outfile])
        catd.write_hdf5(h5file, clobber=True)
        catd.read_hdf5(h5file)
        nt.assert_equal(catd.Nsrcs, len(ras))
        np.testing.assert_almost_equal(catd.data_array, np.array([[[0.75], [0.5], [1.], [0.5], [0.6]]]))
        
    def test_no_hdf5(self):
        catd = cd.catData()
        nt.assert_raises(IOError, catd.read_hdf5, 'src.h5')

    def test_calc_catalog_flux(self):
        catd = cd.catData()
        catd.gen_catalog(ras, decs, [outfile]) 
        beam = bt.get_fitsbeam(beamfits, 151e6)
        catalog_flux = catd.calc_catalog_flux(beam, 'xx')
        nt.assert_almost_equal(catalog_flux[2], 1.000, 3)
