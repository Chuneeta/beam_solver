import os
from beam_solver import gencat as gc
from beam_solver import catdata as cd
from beam_solver import catbeam as cb
from beam_solver import beamsolve as bs
from beam_solver.data import DATA_PATH
import numpy as np
import nose.tools as nt
import copy

# fitsfiles
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

class Test_BeamSolveBase():
    def test_init(self):
        # checks input to the beamSolve class
        bms = bs.BeamSolveBase(cat=srcd)
        bms = bs.BeamSolveBase(cat=srcd, bm_pix='60')
        nt.assert_raises(ValueError, bs.BeamSolveBase, srcd, 'a')       
        
    def test_mk_key(self):
        pixel = 0 
        srcid = 0
        timeid = 0
        bms = bs.BeamSolveBase(cat=srcd)
        bms._mk_key(pixel, srcid, timeid)
        bms._mk_key('0', srcid, timeid)
        bms._mk_key(pixel, '0', timeid)
        bms._mk_key(pixel, srcid, '0')
        bms._mk_key(pixel, '0', timeid)
        nt.assert_raises(ValueError, bms._mk_key, 'a', srcid, timeid)
        nt.assert_raises(ValueError, bms._mk_key, pixel, 'a', timeid)
        nt.assert_raises(ValueError, bms._mk_key, pixel, srcid, 'a')
        
    def test_unravel_key(self):
        bms = bs.BeamSolveBase(cat=srcd)
        bms.unravel_pix(60, (0, 0))
        bms.unravel_pix('60', (0, 0))
        bms.unravel_pix(60, [0, 0])
        bms.unravel_pix(60, ('0', 0))
        bms.unravel_pix(60, (0, '0'))
        nt.assert_raises(ValueError, bms.unravel_pix, 'a', (0, 0))
        nt.assert_raises(ValueError, bms.unravel_pix, 60, ('a', 0))
        nt.assert_raises(ValueError, bms.unravel_pix, 60, (0, 'a'))

    def test_get_weights(self):
        bms = bs.BeamSolveBase(cat=srcd)
        azalts = srcd.azalt_array[:, 0, :]
        bms.get_weights(azalts)
        nt.assert_raises(ValueError, bms.get_weights, azalts[0:1, :])

    def test_construct_linear_sys(self):
        bms = bs.BeamSolveBase(cat=srcd)
        nt.assert_raises(ValueError, bms.construct_linear_sys)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        bms.construct_linear_sys(flux_type='total')
        nt.assert_raises(ValueError, bms.construct_linear_sys, flux_type='a')
        srcd1 = copy.deepcopy(srcd)
        srcd1.pcorr_array = np.ones((5,))
        bms = bs.BeamSolveBase(cat=srcd1)
        nt.assert_raises(AssertionError, bms.construct_linear_sys)

    def test_solve(self):
        bms = bs.BeamSolveBase(cat=srcd)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        bms.solve(solver='Linear')
        nt.assert_raises(ValueError, bms.solve, 'Log')

    def eval_sol(self):
        bms = bs.BeamSolveBase(cat=srcd)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        sol = bms.solve(solver='Linear')
        bms.eval_sol(sol)
        nt.assert_raises(ValueError, bms.eval_sol, np.array([2,.0, 3.0, 4.0]))

    def test_get_A(self):
        bms = bs.BeamSolveBase(cat=srcd)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        sol = bms.solve(solver='Linear')
        bms.get_A()
      
    def test_svd(self):
        bms = bs.BeamSolveBase(cat=srcd)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        sol = bms.solve(solver='Linear')
        A = bms.get_A()
        bms.svd(A)

    def test_remove_degen(self):
        bms = bs.BeamSolveBase(cat=srcd)
        cbeam = cb.catBeamFits()
        bm = cbeam.generate_beam(beamfits, 150e6, pol=['xx'])
        srcd.calc_corrflux(beam=bm)
        bms.construct_linear_sys()
        sol = bms.solve(solver='Linear')
        bms.remove_degen(sol)
        nt.assert_raises(ValueError, bms.remove_degen, sol, 'a')
