import h5py
import os
import numpy as np
import nose.tools as nt
from beam_solver import catdata as cd
from beam_solver import beam_utils as bt
from beam_solver import beamsolve as bs
from beam_solver.data import DATA_PATH


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

class Test_BeamSolveBase():
    def test_construct_linear_sys(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'xx')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_linear_sys(mflux=corrflux)
        srcdict = catd.gen_catalog(fitsfiles_xx, ras, decs, return_data=True)
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'yy')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_linear_sys(mflux=corrflux)
        catd.gen_catalog(fitsfiles_yy, ras, decs)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'yy')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_linear_sys(mflux=corrflux)
        nt.assert_raises(IndexError, bms.construct_linear_sys)
        nt.assert_raises(TypeError, bms.construct_linear_sys, 1.00)
        nt.assert_raises(TypeError, bms.construct_linear_sys, '1.00')
        nt.assert_raises(TypeError, bms.construct_linear_sys, ['1.00'])

    def test_construct_nonlinear_sys(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'xx')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_nonlinear_sys(mflux=corrflux, bvals=np.zeros((60, 60)))
        bms.construct_nonlinear_sys(mflux=corrflux, bvals=np.zeros((60, 60)), constrain=True)
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, mflux=corrflux, bvals=np.zeros((30, 30))) 
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, [], np.zeros((60, 60)))
        nt.assert_raises(TypeError, bms.construct_nonlinear_sys, 1.00, np.zeros((60, 60)))
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, '1.00', np.zeros((60, 60)))
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, ['1.00'], np.zeros((60, 60)))

    def test_solve(self):
        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, ras, decs)
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'xx')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_linear_sys(mflux=corrflux)
        bms.solve(solver='Linear')
        bms.construct_nonlinear_sys(mflux=corrflux, bvals=np.zeros((60, 60)))
        bms.solve(solver='LinProduct')
        nt.assert_raises(AssertionError, bms.solve, solver='Linear')

        catd = cd.catData()
        catd.gen_catalog(fitsfiles_xx, [78.3776346, 74.03785837], [-30.48595805, -30.08651873])
        beam = bt.get_fitsbeam(beamfits, 151e6)
        corrflux = catd.calc_corrflux(beam, 'xx')
        bms = bs.BeamSolveBase(cat=catd)
        bms.construct_nonlinear_sys(mflux=corrflux, bvals=np.zeros((60, 60)), constrain=True)
        nt.assert_raises(KeyError, bms.solve, solver='LinProduct')

class Test_BeamSolveCross():
    def test_construct_linear_sys(self):
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        beam_xx = bt.get_fitsbeam(beamfits, 151e6, 'xx')
        beam_yy = bt.get_fitsbeam(beamfits, 151e6, 'yy')
        corrflux_xx = catd.calc_corrflux(beam_xx, 'xx')
        corrflux_yy = catd.calc_corrflux(beam_yy, 'yy')
        bms = bs.BeamSolveCross(cat=catd)
        bms.construct_linear_sys(mflux_xx=corrflux_xx, mflux_yy=corrflux_yy)
        nt.assert_raises(IndexError, bms.construct_linear_sys, mflux_xx=[], mflux_yy=corrflux_yy)
        nt.assert_raises(IndexError, bms.construct_linear_sys, mflux_xx=corrflux_xx, mflux_yy=[])
        
    def test_construct_nonlinear_sys(self):
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        beam_xx = bt.get_fitsbeam(beamfits, 151e6, 'xx')
        beam_yy = bt.get_fitsbeam(beamfits, 151e6, 'yy')
        corrflux_xx = catd.calc_corrflux(beam_xx, 'xx')
        corrflux_yy = catd.calc_corrflux(beam_yy, 'yy')
        bms = bs.BeamSolveCross(cat=catd)
        bms.construct_nonlinear_sys(mflux_xx=corrflux_xx, mflux_yy=corrflux_yy, bvals=np.zeros((60, 60)))
        bms.construct_nonlinear_sys(mflux_xx=corrflux_xx, mflux_yy=corrflux_yy, bvals=np.zeros((60, 60)), constrain=True) 
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, mflux_xx=[], mflux_yy=corrflux_yy, bvals=np.zeros((60, 60)))
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, mflux_xx=corrflux_xx, mflux_yy=[], bvals=np.zeros((60, 60)))
        nt.assert_raises(IndexError, bms.construct_nonlinear_sys, mflux_xx=corrflux_xx, mflux_yy=corrflux_yy, bvals=np.zeros((30, 30)))

    def test_solve(self):
        catd = cd.catData()
        catd.gen_polcatalog(fitsfiles_xx, fitsfiles_yy, ras, decs)
        beam_xx = bt.get_fitsbeam(beamfits, 151e6, 'xx')
        beam_yy = bt.get_fitsbeam(beamfits, 151e6, 'yy')
        corrflux_xx = catd.calc_corrflux(beam_xx, 'xx')
        corrflux_yy = catd.calc_corrflux(beam_yy, 'yy')
        bms = bs.BeamSolveCross(cat=catd)
        bms.construct_linear_sys(mflux_xx=corrflux_xx, mflux_yy=corrflux_yy)
        bms.solve(solver='Linear')
        nt.assert_raises(AttributeError, bms.solve, solver='LinProduct')
        bms.construct_nonlinear_sys(mflux_xx=corrflux_xx, mflux_yy=corrflux_yy, bvals=np.zeros((60, 60)), constrain=True)
        bms.solve(solver='LinProduct')
        nt.assert_raises(AssertionError, bms.solve, solver='Linear')
