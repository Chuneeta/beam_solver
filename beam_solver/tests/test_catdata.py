import h5py
import os
from beam_solver import gencat as gc
from beam_solver import catdata as cd
from beam_solver import catbeam as cb
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
import copy

# fitsfiles
DATA_PATH = '/Users/Ridhima/Documents/ucb_projects/beam_characterization/beam_solver/beam_solver/data'
beamfits = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits') 
fitsfile1_xx = os.path.join(DATA_PATH, '2458115.23736.xx.fits')
fitsfile2_xx = os.path.join(DATA_PATH, '2458115.24482.xx.fits')
fitsfiles_xx = [fitsfile1_xx, fitsfile2_xx]

# right ascension and declination values
ras = [30.01713089, 27.72922349, 36.75248962, 34.2415497, 78.3776346, 74.03785837]
decs = [-30.88211818, -29.53377208, -30.63958257, -29.93990039, -30.48595805, -30.08651873]

# generating catData object
cat = gc.genCatBase(fitsfiles_xx, ras=ras, decs=decs)
srcd = cat.gen_catalog()

# HDF5 file name
outfile = os.path.join(DATA_PATH, 'srcd.h5')

class Test_catData():
    def test_write_hdf5(self):
        srcd.write_hdf5(outfile, clobber=True)
        nt.assert_raises(IOError, srcd.write_hdf5, outfile)

    def test_get_data(self):
        catd = cd.catData()
        catd.read_hdf5(outfile)

        # extracts data for a specific key (ra, dec, pol)
        key = ((ras[0], decs[0]), 'xx')
        catd.get_pflux(key)
        catd.get_tflux(key)
        key = ((ras[0], decs[0]), 'yy')
        catd.get_pflux(key)
        nt.assert_raises(ValueError, catd.get_pcorr, key)
        nt.assert_raises(ValueError, catd.get_tcorr, key)
        catd.get_ha(ras[0], decs[0])
        catd.get_azalt(ras[0], decs[0])
        key = [(ras[0], decs[0]), 'xx']
        catd.get_pflux(key)
        key = ([ras[0], decs[0]], 'xx')
        catd.get_pflux(key)

        # checks key for extracting data
        key = (('45', decs[0]), 'xx')
        nt.assert_raises(ValueError, catd.get_pflux, key)
        key = ((ras[0], '-30.0'), 'xx')
        nt.assert_raises(ValueError, catd.get_pflux, key)
        key = ((ras[0], decs[0]), 'xy')
        nt.assert_raises(ValueError, catd.get_pflux, key)
        key = ((ras[0], decs[1]), 'xx') 
        nt.assert_raises(KeyError, catd.get_pflux, key)        
        key = ((25.5678, decs[0]), 'xx')
        nt.assert_raises(KeyError, catd.get_pflux, key)
        key = ((ras[0], 30.5678), 'xx')
        nt.assert_raises(KeyError, catd.get_pflux, key)
        key = ((ras[0], decs[0]), 4)
        nt.assert_raises(ValueError, catd.get_pflux, key)
        key = (ras[0], 4)
        nt.assert_raises(ValueError, catd.get_pflux, key)
        key = ras[0]
        nt.assert_raises(ValueError, catd.get_pflux, key)

    def test_calc_corrflux(self):
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.calc_corrflux(beam=bm)

        # checks for polarizations
        catd = cd.catData()
        catd.read_hdf5(outfile)
        nt.assert_raises(ValueError, catd.calc_corrflux, beam=bm, pol=['yy'])

    def test_check(self):
        # checks attributes
        catd = cd.catData()
        catd.read_hdf5(outfile)
        delattr(catd, 'pflux_array')
        nt.assert_raises(AssertionError, catd.check)

        # checks numpy arrays
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.pflux_array = 'hello'
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.pflux_array = catd.pflux_array.astype(int)
        catd.check()
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.pflux_array = np.array(['hello'])
        nt.assert_raises(ValueError, catd.check)
        
        # checks extras
        catd = cd.catData()
        catd.read_hdf5(outfile)
        nfits = catd.Nfits
        catd.Nfits = float(nfits)
        catd.check()
        catd.Nfits = 'a'
        nt.assert_raises(ValueError, catd.check)

        # check optional attributes
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.calc_corrflux(beam=bm)
        catd.beam_size = 'a'
        nt.assert_raises(AssertionError, catd.check)
	catd.calc_corrflux(beam=bm)
	catd.beam_normalization = 45
	nt.assert_raises(AssertionError, catd.check)
	catd.calc_corrflux(beam=bm)
        catd.beam_type = 'a'
        nt.assert_raises(ValueError, catd.check)
	catd.calc_corrflux(beam=bm)
	catd.pcorr_array = 'a'
	nt.assert_raises(AssertionError, catd.check)
	catd.calc_corrflux(beam=bm)
        catd.tcorr_array = 'a'
        nt.assert_raises(AssertionError, catd.check)
        catd.calc_corrflux(beam=bm)
        catd.pcorr_array = np.chararray(catd.pcorr_array.shape)
        nt.assert_raises(ValueError, catd.check)
	catd.calc_corrflux(beam=bm)
        catd.tcorr_array = np.chararray(catd.tcorr_array.shape)
        nt.assert_raises(ValueError, catd.check)
        
	# checks shape of numpy array
        # ras and decs
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.ras = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check) 
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.decs = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)

        # data arrays
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.tflux_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.tflux_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check) 
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.pcorr_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.tcorr_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.azalt_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.ha_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.jd_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.freq_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.lst_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
        catd = cd.catData()
        catd.read_hdf5(outfile)
        catd.rms_array = np.array([1., 2., 3., 4,])
        nt.assert_raises(AssertionError, catd.check)
