import h5py
import os
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
from beam_solver import catdata as cd
from beam_solver import beam_utils as bt
from beam_solver import fits_utils as ft
import copy

beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
fitsfile = os.path.join(DATA_PATH, '2458115.23736.test.xx.fits')
outfile = fitsfile.replace('.fits', '.mod.fits')
h5file = os.path.join(DATA_PATH, 'srcd.h5')

ras = [74.26237654, 41.91116875, 22.47460079, 9.8393989, 356.25426296]
decs = [-52.0209015, -43.2292595, -30.27372862, -17.40763737, -0.3692835]

def create_catdata(data, ha, nsrcs, npoints, pols=['xx']):
    catd = cd.catData()
    #catd.azalt_array = azalt
    catd.data_array = data
    catd.ha_array = ha
    catd.Nfits = npoints
    catd.Nsrcs = nsrcs
    catd.pols = pols
    return catd

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
        
    def test_get_resolution(self):
        catd = cd.catData()
        npoints = catd._get_resolution(60)
        x = np.linspace(0, np.pi, 60)
        nt.assert_almost_equal(npoints, x[1] - x[0])

    def test_get_npoints(self):
        catd = cd.catData()
        npix = 91
        x = np.linspace(0, np.pi, npix)
        azs = np.array([[0, np.pi/2, np.pi],[0, np.pi/2, np.pi]])
        alts = np.array([[0, np.pi/2, np.pi], [0, np.pi/2, np.pi]])
        catd.azalt_array = np.zeros((2, azs.shape[0], azs.shape[1]))
        catd.azalt_array[0, :, :] = azs
        catd.azalt_array[1, :, :] = alts
        npoints = catd._get_npoints(npix)
        nt.assert_almost_equal(npoints, int(np.pi/ (x[1] - x[0])) + 1)

    def test_get_npoints(self):
        catd = cd.catData()
        ha = np.array([[np.pi/2, np.pi/4, 1e-5, -np.pi/4, -np.pi/2]])
        catd.ha_array = ha
        npoints = catd._get_npoints(0.01)
        nt.assert_equal(npoints, int(np.pi/0.01))
 
    def test_interpolate_data(self):
        catd = cd.catData()
        x = np.arange(0, 5)
        y = np.array([0., 0.5, 1., 1.5, 2.])
        f = catd._interpolate_data(x, y, kind='linear')
        nt.assert_equal(f(0), 0.)        
        np.testing.assert_almost_equal(f(np.array([0, 1.5])), np.array([0., 0.75]))
        y = np.array([0., 0.5, 1., -0.55, -0.1])
        f = catd._interpolate_data(x, y, kind='cubic')
        nt.assert_true( -0.55 < f(np.array([2.5])) < 1.)

    def test_interpolate_catalog(self):
        data = np.array([[[0., 0.5, 1., 1.5, 2.]]])
        ha = np.array([[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]])
        catd = create_catdata(data, ha, 1, 5)
        catd.pos_array = np.array([(30.01, -30.43)])
        catd_copy = copy.deepcopy(catd) 
        catd_copy.interpolate_catalog(dha = np.pi/8)
        f = catd._interpolate_data(ha[0, :], catd.data_array[0, 0, :], kind='cubic')        
        np.testing.assert_almost_equal(catd_copy.ha_array[0, :], np.linspace(-np.pi/2, np.pi/2, 9))
        np.testing.assert_almost_equal(catd_copy.data_array[0, 0, :], f(np.linspace(-np.pi/2, np.pi/2, 9)))
        nt.assert_equal(catd_copy.Nfits, 9)
        nt.assert_equal(len(catd_copy.azalt_array[0, 0, :]), 9) 
 
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
